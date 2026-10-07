// src/kernels/hc_q8_parity.cu - GPU checks of the hyper-connection read from Q8_0 projections (STRATA_HC_Q8=1) and of its
// one-launch form (STRATA_HC_FUSED=1).  Synthetic weights and activations, no model, a few seconds, under 70 MiB of
// VRAM (--bench: about 150 MiB, it keeps six layers' weights so that the reads come from DRAM, not from L2).
//
//   hc_q8_parity                  the checks below; exit 0 = all passed
//   hc_q8_parity --bench [T]      then time one read (T tokens, default 3) in a graph of 96 reads, the way a window runs them:
//                                 the default BF16 read, the Q8_0 read, and both with the counters set (STRATA_HC_FUSED=1's form;
//                                 the environment variable STRATA_HC_FUSED_ONE_LAUNCH=1 makes that form the one-launch kernels)
//
//   1. the Q8_0 read (two launches) against a double-precision reference computed from the Q8_0 values: the weights and the
//      formula are right on this GPU (tolerance covers the GPU's exp and fp32 accumulation order)
//   2. the fused one-launch read against the two launches, bit for bit: 1..8 tokens, with and without the pending write, inject
//      rows BF16 / Q8_0 / none, R_out in place and not, with the q8_1 image (STRATA_QFUSE); the block counters are zero after
//   3. the same through a CUDA graph, replayed with new inputs (counters must reset), 96 reads back to back
//   4. the fused read with most of every SM taken by another kernel (it must still finish: its blocks start in ticket order)
//   2b the same for the BF16 read (the default weights): the one-launch form against the staged read's three launches
//   5. the default BF16 read with the pack's BF16 copy of the same weights, against the Q8_0 read: printed (the numeric
//      difference STRATA_HC_Q8 makes in one read), not a pass/fail
#include "strata/kernels/fused_gr.hpp"

#include "hc_q8_ref.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::kernels;
using hcref::D;
using hcref::HC;
using hcref::LR;
using hcref::N;

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
int failures = 0;
#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

template <class V> V* upload(const std::vector<V>& h) {
    V* d = nullptr;
    check(cudaMalloc((void**) &d, h.size() * sizeof(V)), "cudaMalloc");
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(V), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <class V> std::vector<V> download(const V* d, size_t n) {
    std::vector<V> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(V), cudaMemcpyDeviceToHost), "download");
    return h;
}

__global__ void hog_kernel(long long cycles) {
    extern __shared__ char sm[];
    if (threadIdx.x == 0) sm[0] = 1;
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}

/// one set of weights on the device (several sets in a bench, to read from DRAM)
struct Weights {
    uint8_t *q8_down = nullptr, *q8_up = nullptr, *q8_inj = nullptr;
    uint16_t *bf_down = nullptr, *bf_up = nullptr, *bf_inj = nullptr;   // the pack's BF16 copy of the same values (inject: exact)
    void load(const hcref::Fixture& f) {
        q8_down = upload(f.q8_down); q8_up = upload(f.q8_up); q8_inj = upload(f.q8_inj);
        bf_down = upload(hcref::Fixture::bf16_copy(f.dq_down)); bf_up = upload(hcref::Fixture::bf16_copy(f.dq_up)); bf_inj = upload(f.w_inj);
    }
    void release() {
        for (void* p : {(void*) q8_down, (void*) q8_up, (void*) q8_inj, (void*) bf_down, (void*) bf_up, (void*) bf_inj}) cudaFree(p);
    }
};

struct Opts {
    bool apply = true, inject = true, inject_q8 = false, in_place = false, qfuse = false;
    bool q8 = true, fused = false;   // q8 false: the default BF16 read
};
struct Out {
    std::vector<float> mixed, Rout, rs, inject, lo;
    std::vector<uint8_t> q81;
    bool counters_zero = true;
};

/// The buffers of one read of T tokens, and the call.
struct Read {
    int T;
    const hcref::Fixture& f;
    float *R, *Rout, *bo, *inj_prev, *w_norm, *lo, *rs, *inj_out, *mixed, *xn;
    uint8_t* q81;
    unsigned *qcnt, *sync;
    cudaStream_t st;   // every copy and memset below is ordered with the reads on this stream (the legacy default stream is not
                       // ordered with a non-blocking one: a late memset would poison a finished read's output)
    Read(const hcref::Fixture& fx, cudaStream_t stream) : T(fx.T), f(fx), st(stream) {
        R = upload(f.R); bo = upload(f.bo); inj_prev = upload(f.inj_prev); w_norm = upload(f.w_norm);
        auto z = [](size_t n) { float* p = nullptr; check(cudaMalloc((void**) &p, n * 4), "malloc"); check(cudaMemset(p, 0, n * 4), "memset"); return p; };
        Rout = z((size_t) T * D); lo = z((size_t) T * LR); rs = z((size_t) T * HC); inj_out = z((size_t) T * HC);
        mixed = z((size_t) T * N); xn = z((size_t) T * D * 2);
        check(cudaMalloc((void**) &q81, (size_t) T * (N / 32) * 36), "malloc");
        check(cudaMalloc((void**) &qcnt, (N / 32) * 4), "malloc"); check(cudaMemset(qcnt, 0, (N / 32) * 4), "memset");
        check(cudaMalloc((void**) &sync, kFusedGrSyncWords * 4), "malloc"); check(cudaMemset(sync, 0, kFusedGrSyncWords * 4), "memset");
    }
    ~Read() {
        for (void* p : {(void*) R, (void*) Rout, (void*) bo, (void*) inj_prev, (void*) w_norm, (void*) lo, (void*) rs, (void*) inj_out,
                        (void*) mixed, (void*) xn, (void*) q81, (void*) qcnt, (void*) sync}) cudaFree(p);
    }
    void reset_inputs() {   // R is rewritten in place by a read with in_place
        check(cudaMemcpyAsync(R, f.R.data(), f.R.size() * 4, cudaMemcpyHostToDevice, st), "reset R");
        check(cudaMemsetAsync(q81, 0x5a, (size_t) T * (N / 32) * 36, st), "poison q81");
        check(cudaMemsetAsync(Rout, 0, (size_t) T * D * 4, st), "clear R_out");
        check(cudaStreamSynchronize(st), "reset sync");
    }
    void call(const Weights& w, const Opts& o, cudaStream_t st) {
        FusedGrArgs a[kFusedGrMaxT];
        for (int t = 0; t < T; ++t) {
            a[t].R = R + (size_t) t * D;
            a[t].R_out = o.in_place ? R + (size_t) t * D : Rout + (size_t) t * D;
            a[t].apply = o.apply;
            a[t].bo_prev = bo + (size_t) t * N;
            a[t].inj_prev = inj_prev + (size_t) t * HC;
            a[t].w_norm = w_norm;
            a[t].w_down = w.bf_down; a[t].w_up = w.bf_up;
            a[t].w_inject = o.inject ? w.bf_inj : nullptr;
            if (o.q8) {
                a[t].q8_down = w.q8_down; a[t].q8_up = w.q8_up;
                a[t].q8_inject = o.inject && o.inject_q8 ? w.q8_inj : nullptr;
            }
            if (o.fused) a[t].hc_sync = sync;
            a[t].eps = f.eps;
            a[t].lo = lo + (size_t) t * LR; a[t].rs = rs + (size_t) t * HC;
            a[t].inject_out = inj_out + (size_t) t * HC; a[t].mixed = mixed + (size_t) t * N;
            if (o.qfuse) { a[t].q8_mixed = q81 + (size_t) t * (N / 32) * 36; a[t].q8_cnt = qcnt; }
        }
        fused_gr_read_multi(a, T, xn, st);
    }
    Out fetch(const Opts& o) {
        check(cudaDeviceSynchronize(), "sync");
        Out r;
        r.mixed = download(mixed, (size_t) T * N);
        r.Rout = download(o.in_place ? R : Rout, (size_t) T * D);
        r.rs = download(rs, (size_t) T * HC);
        r.inject = download(inj_out, (size_t) T * HC);
        r.lo = download(lo, (size_t) T * LR);
        r.q81 = download(q81, (size_t) T * (N / 32) * 36);
        for (unsigned v : download(sync, kFusedGrSyncWords)) if (v != 0) r.counters_zero = false;
        for (unsigned v : download(qcnt, N / 32)) if (v != 0) r.counters_zero = false;
        return r;
    }
};

bool equal(const Out& a, const Out& b, const Opts& o) {
    return hcref::same_bits(a.mixed, b.mixed) && (!o.apply || hcref::same_bits(a.Rout, b.Rout)) && hcref::same_bits(a.rs, b.rs) &&
           (!o.inject || hcref::same_bits(a.inject, b.inject)) && (!o.qfuse || a.q81 == b.q81) &&
           (o.q8 || hcref::same_bits(a.lo, b.lo));   // (the Q8_0 read keeps `lo` in shared memory unless it is the one-launch form)
}

double ms_per(cudaGraphExec_t g, cudaStream_t st, int reps) {
    for (int i = 0; i < 3; ++i) check(cudaGraphLaunch(g, st), "warm");
    check(cudaStreamSynchronize(st), "warm sync");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, st);
    for (int i = 0; i < reps; ++i) check(cudaGraphLaunch(g, st), "launch");
    cudaEventRecord(e1, st);
    check(cudaEventSynchronize(e1), "timing");
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / (float) reps;
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    int bench_T = 3;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") { bench = true; if (i + 1 < argc && argv[i + 1][0] != '-') bench_T = std::atoi(argv[++i]); }
    }
    int dev = 0, sms = 0, cc_major = 0, cc_minor = 0;
    check(cudaGetDevice(&dev), "device");
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::printf("device %d: %s, sm_%d%d, %d SMs\n", dev, prop.name, cc_major, cc_minor, sms);
    if (cc_major < 7) { std::printf("compute capability < 7.0: the fused read is not supported there; nothing to check\n"); return 0; }
    fused_gr_check();   // the default read's variant check, as the engine does at start

    cudaStream_t st;
    check(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");

    // ---- 1 + 2 ------------------------------------------------------------------------------------------------------
    for (int T = 1; T <= kFusedGrMaxT; ++T) {
        hcref::Fixture f;
        f.build(T, 900u + (unsigned) T);
        Weights w;
        w.load(f);
        Read rd(f, st);
        struct V { bool apply, inject, inject_q8, in_place, qfuse; };
        const V vs[] = {{true, true, false, false, false}, {true, true, false, true, true}, {false, true, true, false, false},
                        {false, false, false, false, true}, {true, false, false, true, false}};
        for (size_t vi = 0; vi < sizeof(vs) / sizeof(vs[0]); ++vi) {
            Opts o;
            o.apply = vs[vi].apply; o.inject = vs[vi].inject; o.inject_q8 = vs[vi].inject_q8; o.in_place = vs[vi].in_place;
            o.qfuse = vs[vi].qfuse; o.q8 = true; o.fused = false;
            rd.reset_inputs();
            rd.call(w, o, st);
            const Out two = rd.fetch(o);
            if (vi == 0 || vi == 2) {   // 1. against the double reference
                const hcref::Ref r = hcref::reference(f, o.apply, o.inject, o.inject_q8, f.dq_down, f.dq_up);
                const double em = hcref::worst(two.mixed, r.mixed), er = o.apply ? hcref::worst(two.Rout, r.Rout) : 0.0, es = hcref::worst(two.rs, r.rs);
                const double ei = o.inject ? hcref::worst(two.inject, r.inject) : 0.0;
                std::printf("1  T=%d inject=%s: Q8_0 read vs double reference: mixed %.2e R_out %.2e rs %.2e inject %.2e\n", T,
                            !o.inject ? "none" : o.inject_q8 ? "q8_0" : "bf16", em, er, es, ei);
                CHECK(em < 3e-5); CHECK(er < 1e-6); CHECK(es < 1e-6); CHECK(ei < 3e-5);
            }
            o.fused = true;
            for (int rep = 0; rep < 2; ++rep) {   // twice: the second launch finds the counters as the first left them
                rd.reset_inputs();
                rd.call(w, o, st);
                const Out fused = rd.fetch(o);
                const bool same = equal(two, fused, o);
                std::printf("2  T=%d apply=%d inject=%d%s in_place=%d qfuse=%d rep %d: fused vs two launches %s%s\n", T, o.apply, o.inject,
                            o.inject_q8 ? "(q8)" : "", o.in_place, o.qfuse, rep, same ? "bit-identical" : "DIFFERS",
                            fused.counters_zero ? "" : ", COUNTERS NOT ZERO");
                CHECK(same); CHECK(fused.counters_zero);
            }
        }
        // 2b. the BF16 read (the default weights): the one-launch form against the staged read's three launches (up to 4 rows;
        // more rows keep the three launches and must come out the same, trivially)
        {
            const V vb[] = {{true, true, false, false, false}, {true, true, false, true, true}, {false, false, false, false, true}};
            for (size_t vi = 0; vi < sizeof(vb) / sizeof(vb[0]); ++vi) {
                Opts o;
                o.apply = vb[vi].apply; o.inject = vb[vi].inject; o.in_place = vb[vi].in_place; o.qfuse = vb[vi].qfuse;
                o.q8 = false; o.fused = false;
                rd.reset_inputs();
                rd.call(w, o, st);
                const Out three = rd.fetch(o);
                o.fused = true;
                for (int rep = 0; rep < 2; ++rep) {
                    rd.reset_inputs();
                    rd.call(w, o, st);
                    const Out one = rd.fetch(o);
                    const bool same = equal(three, one, o);
                    std::printf("2b T=%d BF16 apply=%d inject=%d in_place=%d qfuse=%d rep %d: one launch vs three %s%s\n", T, o.apply, o.inject,
                                o.in_place, o.qfuse, rep, same ? "bit-identical" : "DIFFERS", one.counters_zero ? "" : ", COUNTERS NOT ZERO");
                    CHECK(same); CHECK(one.counters_zero);
                }
            }
        }
        // 3. a captured graph of 96 fused reads, replayed with new inputs
        for (int q8 = 1; q8 >= 0; --q8) {
            Opts o;   // apply, bf16 inject, not in place
            o.q8 = q8 != 0;
            o.fused = true;
            cudaGraph_t g = nullptr;
            cudaGraphExec_t ge = nullptr;
            check(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
            for (int i = 0; i < 96; ++i) rd.call(w, o, st);
            check(cudaStreamEndCapture(st, &g), "end capture");
            check(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0), "instantiate");
            for (int replay = 0; replay < 3; ++replay) {
                std::vector<float> r2 = f.R;
                for (size_t i = 0; i < r2.size(); ++i) r2[i] = f.R[i] * (0.5f + 0.25f * (float) replay) + 0.01f * (float) (i % 13);
                check(cudaMemcpyAsync(rd.R, r2.data(), r2.size() * 4, cudaMemcpyHostToDevice, st), "new R");
                check(cudaStreamSynchronize(st), "new R sync");
                check(cudaGraphLaunch(ge, st), "graph launch");
                const Out got = rd.fetch(o);
                Opts ot = o; ot.fused = false;
                check(cudaMemcpyAsync(rd.R, r2.data(), r2.size() * 4, cudaMemcpyHostToDevice, st), "new R");
                check(cudaStreamSynchronize(st), "new R sync");
                rd.call(w, ot, st);
                const Out want = rd.fetch(ot);
                const bool same = equal(got, want, o);
                if (T == 1 || T == 3 || T == 8)
                    std::printf("3  T=%d graph of 96 fused %s reads, replay %d: %s%s\n", T, q8 ? "Q8_0" : "BF16", replay, same ? "bit-identical to the multi-launch read" : "DIFFERS",
                                got.counters_zero ? "" : ", COUNTERS NOT ZERO");
                CHECK(same); CHECK(got.counters_zero);
            }
            cudaGraphExecDestroy(ge);
            cudaGraphDestroy(g);
        }
        // 4. most of every SM taken by another kernel
        for (int q8 = 1; q8 >= 0 && (T == 3 || T == 8 || T == 4); --q8) {
            Opts o;
            o.q8 = q8 != 0;
            o.fused = true;
            Opts ot = o; ot.fused = false;
            rd.reset_inputs();
            rd.call(w, ot, st);
            const Out want = rd.fetch(ot);
            cudaStream_t other;
            check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "stream 2");
            const int smem = 56 * 1024;   // leaves room for one fused block per SM (not for 2.6 per SM: 68 resident blocks < 176 down tasks)
            check(cudaFuncSetAttribute(hog_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "hog smem");
            const long long cycles = 60000000ll;   // ~30-40 ms
            hog_kernel<<<sms, 1024, smem, other>>>(cycles);
            check(cudaGetLastError(), "hog launch");
            rd.reset_inputs();
            const auto t0 = std::chrono::steady_clock::now();
            rd.call(w, o, st);
            const Out got = rd.fetch(o);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            check(cudaStreamSynchronize(other), "hog sync");
            cudaStreamDestroy(other);
            const bool same = equal(got, want, o);
            std::printf("4  T=%d with a kernel holding 56 KiB and 1024 threads on every SM: fused %s read %s, %.1f ms%s\n", T, q8 ? "Q8_0" : "BF16", same ? "bit-identical" : "DIFFERS", ms,
                        got.counters_zero ? "" : ", COUNTERS NOT ZERO");
            CHECK(same); CHECK(got.counters_zero);
        }
        // 5. the default BF16 read with the pack's rounding of the same weights, against the Q8_0 read
        if (T == 1 || T == 3 || T == 8) {
            Opts oq, ob;
            oq.inject = true; ob.inject = true; ob.q8 = false;
            rd.reset_inputs(); rd.call(w, oq, st); const Out q = rd.fetch(oq);
            rd.reset_inputs(); rd.call(w, ob, st); const Out b = rd.fetch(ob);
            std::vector<double> qm(q.mixed.begin(), q.mixed.end()), qi(q.inject.begin(), q.inject.end());
            std::printf("5  T=%d BF16-copy read vs Q8_0 read (the numeric effect of STRATA_HC_Q8 on one read, worst / max): mixed %.2e  inject %.2e\n",
                        T, hcref::worst(b.mixed, qm), hcref::worst(b.inject, qi));
        }
        w.release();
    }

    // ---- bench ------------------------------------------------------------------------------------------------------
    if (bench) {
        constexpr int SETS = 6, READS = 96;
        hcref::Fixture f;
        f.build(bench_T, 4242u);
        Weights w[SETS];
        for (int i = 0; i < SETS; ++i) w[i].load(f);
        Read rd(f, st);
        struct Mode { const char* name; Opts o; double bytes; } modes[4];
        constexpr double kParams = 320.0 * 10240.0;   // one projection matrix
        modes[0] = {"BF16 default read (3 launches)", Opts{}, 2.0 * kParams * 2.0};
        modes[0].o.q8 = false;
        modes[1] = {"Q8_0 read, two launches", Opts{}, 2.0 * kParams * 34.0 / 32.0};
        modes[2] = {"Q8_0 read, STRATA_HC_FUSED", Opts{}, modes[1].bytes};
        modes[2].o.fused = true;
        modes[3] = {"BF16 read, STRATA_HC_FUSED", Opts{}, modes[0].bytes};
        modes[3].o.q8 = false;
        modes[3].o.fused = true;
        std::printf("bench: T=%d, %d reads per graph, %d weight sets cycled (DRAM, not L2)\n", bench_T, READS, SETS);
        for (auto& m : modes) {
            cudaGraph_t g = nullptr;
            cudaGraphExec_t ge = nullptr;
            check(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
            for (int i = 0; i < READS; ++i) rd.call(w[i % SETS], m.o, st);
            check(cudaStreamEndCapture(st, &g), "end capture");
            check(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0), "instantiate");
            const double ms = ms_per(ge, st, 20);
            std::printf("   %-34s %7.1f us per read   (%.0f GB/s of weights)\n", m.name, 1000.0 * ms / READS, m.bytes / (ms / READS * 1e-3) / 1e9);
            cudaGraphExecDestroy(ge);
            cudaGraphDestroy(g);
        }
        for (auto& s : w) s.release();
    }

    cudaStreamDestroy(st);
    if (failures) { std::printf("hc_q8_parity: %d failure(s)\n", failures); return 1; }
    std::printf("hc_q8_parity: all checks passed\n");
    return 0;
}
