// tests/cuda_emu/cuda_emu.hpp - a CPU emulation of the CUDA execution model, just enough to run the hyper-connection
// read's device code (src/kernels/cuda/fused_gr_common.cuh, hc_q8.cuh) on a machine without a GPU.
//
// WHAT IT IS.  Every CUDA thread is a fiber (its own stack, a hand-written x86-64 context switch), a block is 256 of them,
// and the grid is run as a pool of `resident` blocks that are dispatched in any order the test asks for (ascending,
// descending, shuffled).  __syncthreads, __shfl_xor_sync and the spin loops of the kernels are cooperative waits, so a
// kernel whose blocks wait on each other (the fused hyper-connection read) runs here exactly as far as it would on a GPU
// that keeps only `resident` blocks on its SMs: a kernel that needs more resident blocks than it is given deadlocks, and
// the emulator reports it (no fiber finished for a very long time) instead of hanging.
//
// WHAT IT IS NOT.  One OS thread runs everything, so memory is sequentially consistent: a missing __threadfence cannot
// be found here, only the logic (indexing, partitioning, ordering of the blocks' protocol, the arithmetic).  Float math
// is the host's: fmaf and the conversions are exact IEEE, __expf is expf (the GPU's is the hardware approximation: a
// result that goes through it differs from a GPU run, but two kernels that both use it differ from each other exactly
// as they do on the GPU).  `__shared__` data is declared through STRATA_SHARED so that every block has its own.
//
// Include this BEFORE the device fragments, in a translation unit that is plain C++ (not nvcc).
#pragma once

#pragma GCC diagnostic ignored "-Wparentheses"   // STRATA_SHARED declares `int (&last) = ...`

#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(__x86_64__) || !defined(__linux__)
#error "cuda_emu.hpp needs x86-64 Linux (hand-written context switch)"
#endif

// ------------------------------------------------------------------------------------------------------------------
// the fiber context switch: save the callee-saved registers on the old stack, load the new stack pointer, restore
extern "C" void emu_ctx_swap(void** save_sp, void* load_sp);
asm(R"(
.text
.globl emu_ctx_swap
.type emu_ctx_swap,@function
emu_ctx_swap:
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    movq %rsp, (%rdi)
    movq %rsi, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
.size emu_ctx_swap,.-emu_ctx_swap
)");

namespace emu {

struct uint3_ { unsigned x = 0, y = 0, z = 0; };
struct dim3 : uint3_ {
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) { x = x_; y = y_; z = z_; }
};

struct Block;
struct Fiber {
    void* sp = nullptr;
    std::unique_ptr<char[]> stack;
    Block* blk = nullptr;
    uint3_ tid;
    int lane = 0, warp = 0;
    bool done = false;
};
struct Warp {
    int alive = 32, arrived = 0;
    unsigned gen = 0;
    uint64_t slot[32] = {};
};
struct Free { void operator()(void* p) const { std::free(p); } };
struct Block {
    uint3_ bid;
    std::vector<Fiber> fibers;
    std::vector<Warp> warps;
    int alive = 0, arrived = 0;
    unsigned gen = 0;
    std::map<int, std::unique_ptr<void, Free>> smem;   // __COUNTER__ of the STRATA_SHARED site -> this block's copy
};

// the scheduler's state: one OS thread, so plain globals
inline Fiber* g_cur = nullptr;
inline void* g_sched_sp = nullptr;
inline uint3_ g_blockDim, g_gridDim;
inline std::function<void()> g_body;
inline unsigned long long g_switches = 0, g_since_done = 0;
inline unsigned long long g_deadlock_switches = 200000000ull;   // switches without any fiber finishing: a deadlock
inline unsigned long long g_cycles = 0;

inline void yield() {
    ++g_switches; ++g_since_done;
    Fiber* f = g_cur;
    emu_ctx_swap(&f->sp, g_sched_sp);
}
template <class Cond> inline void wait_until(Cond cond) {
    while (!cond()) yield();
}

inline std::string g_error;   // the first exception a thread threw (__trap); launch() rethrows it

inline void fiber_entry() {
    try {
        g_body();
    } catch (const std::exception& e) {
        if (g_error.empty()) g_error = e.what();
    }
    Fiber* f = g_cur;
    Block* b = f->blk;
    f->done = true;
    g_since_done = 0;
    // a finished thread no longer holds up its warp's or its block's barrier
    Warp& w = b->warps[(size_t) f->warp];
    --w.alive; --b->alive;
    if (w.arrived > 0 && w.arrived >= w.alive) { w.arrived = 0; ++w.gen; }
    if (b->arrived > 0 && b->arrived >= b->alive) { b->arrived = 0; ++b->gen; }
    void* dummy = nullptr;
    emu_ctx_swap(&dummy, g_sched_sp);
    std::abort();   // never resumed
}

inline void syncthreads() {
    Block* b = g_cur->blk;
    if (++b->arrived >= b->alive) { b->arrived = 0; ++b->gen; return; }
    const unsigned g = b->gen;
    wait_until([&] { return b->gen != g; });
}
inline void warp_barrier(Warp& w) {
    if (++w.arrived >= w.alive) { w.arrived = 0; ++w.gen; return; }
    const unsigned g = w.gen;
    wait_until([&] { return w.gen != g; });
}
template <class T> inline T shfl_xor(T v, int off) {
    static_assert(sizeof(T) <= 8, "shuffle payload");
    Fiber* f = g_cur;
    Warp& w = f->blk->warps[(size_t) f->warp];
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(T));
    w.slot[f->lane] = bits;
    warp_barrier(w);
    const int src = (f->lane ^ off) & 31;
    const uint64_t o = w.slot[src];
    warp_barrier(w);
    T r;
    std::memcpy(&r, &o, sizeof(T));
    return r;
}
inline void* smem_get(size_t bytes, int key) {
    Block* b = g_cur->blk;
    auto it = b->smem.find(key);
    if (it == b->smem.end()) {
        void* p = std::aligned_alloc(64, (bytes + 63) / 64 * 64);
        std::memset(p, 0xFF, bytes);   // an uninitialised read is a NaN, not a lucky zero
        it = b->smem.emplace(key, std::unique_ptr<void, Free>(p)).first;
    }
    return it->second.get();
}

enum class Order { Ascending, Descending, Shuffled };
struct LaunchOpts {
    int resident = 8;                  // blocks on the "GPU" at once
    Order order = Order::Ascending;    // the order the hardware dispatches the blocks in
    unsigned seed = 1;
};

/// Run `body` once per CUDA thread of a grid x block launch.  Returns when every block has finished.
template <class F> void launch(dim3 grid, dim3 block, F body, const LaunchOpts& opt = {}) {
    g_blockDim = block; g_gridDim = grid;
    g_body = [body]() mutable { body(); };
    const unsigned nblocks = grid.x * grid.y * grid.z, nthreads = block.x * block.y * block.z;
    std::vector<unsigned> idx(nblocks);
    for (unsigned i = 0; i < nblocks; ++i) idx[i] = i;
    if (opt.order == Order::Descending) std::reverse(idx.begin(), idx.end());
    if (opt.order == Order::Shuffled) { std::mt19937 r(opt.seed); std::shuffle(idx.begin(), idx.end(), r); }
    size_t next = 0;
    std::vector<std::unique_ptr<Block>> res;
    constexpr size_t kStack = 64 * 1024;
    g_since_done = 0;
    g_error.clear();
    while (next < idx.size() || !res.empty()) {
        while (next < idx.size() && (int) res.size() < opt.resident) {
            const unsigned lin = idx[next++];
            auto b = std::make_unique<Block>();
            b->bid.x = lin % grid.x; b->bid.y = (lin / grid.x) % grid.y; b->bid.z = lin / (grid.x * grid.y);
            b->fibers.resize(nthreads);
            b->warps.resize((nthreads + 31) / 32);
            b->alive = (int) nthreads;
            for (unsigned t = 0; t < nthreads; ++t) {
                Fiber& f = b->fibers[t];
                f.blk = b.get();
                f.tid.x = t % block.x; f.tid.y = (t / block.x) % block.y; f.tid.z = t / (block.x * block.y);
                f.lane = (int) (t & 31); f.warp = (int) (t >> 5);
                f.stack.reset(new char[kStack]);
                // initial frame: six saved registers, the entry as the return address, a fake caller above it
                uintptr_t top = ((uintptr_t) f.stack.get() + kStack) & ~(uintptr_t) 15;
                void** sp = (void**) top;
                *--sp = nullptr;                          // fake return address of fiber_entry
                *--sp = (void*) &fiber_entry;             // popped by `ret`
                for (int r = 0; r < 6; ++r) *--sp = nullptr;   // rbp rbx r12..r15
                f.sp = sp;
            }
            res.push_back(std::move(b));
        }
        for (auto& b : res) {
            for (Fiber& f : b->fibers) {
                if (f.done) continue;
                g_cur = &f;
                emu_ctx_swap(&g_sched_sp, f.sp);
                if (g_since_done > g_deadlock_switches) {
                    std::fprintf(stderr, "cuda_emu: no thread finished in %llu switches: deadlock (resident %d, next block %zu of %zu)\n",
                                 g_since_done, opt.resident, next, idx.size());
                    throw std::runtime_error("cuda_emu: deadlock");
                }
            }
        }
        if (!g_error.empty()) throw std::runtime_error(g_error);
        for (size_t i = 0; i < res.size();) {
            if (res[i]->alive == 0) res.erase(res.begin() + (long) i);
            else ++i;
        }
    }
    g_cur = nullptr;
}

}  // namespace emu

// ------------------------------------------------------------------------------------------------------------------
// the CUDA names the device fragments use
#define STRATA_CUDA_EMU 1
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define threadIdx (emu::g_cur->tid)
#define blockIdx (emu::g_cur->blk->bid)
#define blockDim (emu::g_blockDim)
#define gridDim (emu::g_gridDim)
#define __syncthreads() emu::syncthreads()
#define __threadfence() ((void) 0)
#define STRATA_SHARED(type, name, dims) \
    type(&name) dims = *reinterpret_cast<type(*) dims>(emu::smem_get(sizeof(type dims), __COUNTER__))

template <class T> inline T __shfl_xor_sync(unsigned, T v, int off) { return emu::shfl_xor(v, off); }
template <class T> inline T __ldcg(const T* p) { return *p; }
inline unsigned atomicAdd(unsigned* p, unsigned v) { const unsigned o = *p; *p = o + v; return o; }
inline int atomicAdd(int* p, int v) { const int o = *p; *p = o + v; return o; }
inline unsigned atomicExch(unsigned* p, unsigned v) { const unsigned o = *p; *p = v; return o; }
inline float __expf(float x) { return std::exp(x); }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline long long clock64() { return (long long) (emu::g_cycles += 1000); }
inline void __nanosleep(unsigned) { emu::yield(); }
[[noreturn]] inline void __trap() { throw std::runtime_error("cuda_emu: __trap()"); }

struct float4 { float x, y, z, w; };
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
struct uint2 { unsigned x, y; };
struct uint4 { unsigned x, y, z, w; };
inline float __uint_as_float(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }
struct half { uint16_t bits = 0; half() = default; half(float f) : bits(strata::kernels::f16_from_f32(f)) {} };
struct half2 { half x, y; };
inline half2 make_half2(half a, half b) { return {a, b}; }
inline half __ushort_as_half(unsigned short b) { half h; h.bits = b; return h; }
inline float __half2float(half h) { return strata::kernels::f32_from_f16(h.bits); }
