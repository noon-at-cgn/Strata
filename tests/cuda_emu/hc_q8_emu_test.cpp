// tests/cuda_emu/hc_q8_emu_test.cpp - the hyper-connection read from Q8_0 projections, run on the CPU (no GPU needed).
//
// The device code under test is the production code: src/kernels/cuda/fused_gr_common.cuh and hc_q8.cuh are included here
// unchanged and executed by tests/cuda_emu/cuda_emu.hpp (every CUDA thread a fiber, blocks dispatched in a chosen order).
//
//   A. gr_down_q8_kernel + gr_up_q8_kernel (STRATA_HC_Q8=1's two launches) against a double-precision reference computed
//      from the raw Q8_0 bytes: the formula, the chunking and the indexing of the existing read are right.
//   B. gr_q8_fused_kernel (STRATA_HC_Q8_FUSED=1, one launch) against those two launches, BIT FOR BIT, for 1..8 tokens, with
//      and without the pending write, with and without the inject rows (BF16 and Q8_0), with the S26 q8_1 image, with R_out
//      in place and not, and under several block dispatch orders and residency limits (including one block at a time):
//      the fused read changes the launch structure, not one bit of the result; and the counters are zero afterwards.
//   E. the same for the BF16 read (gr_hc_fused_kernel, STRATA_HC_FUSED=1 without STRATA_HC_Q8): bit-identical to the staged read's
//      three launches (gr_norm_split_kernel, gr_down_staged_kernel, gr_up_multi_kernel), and that read to a double reference.
//   C. a kernel that waits for blocks that cannot start is reported as a deadlock, not hung (the emulator's guard).
//   D. the ticket is what makes the fused read safe: the same kernel with tasks taken from blockIdx is a deadlock when the up
//      blocks are dispatched first and the GPU has room for two blocks; the ticketed one finishes.
//
// The CPU cannot show timing, fences, or the GPU's exp: GPU runs of the same checks are src/kernels/hc_q8_parity.cpp.
#include "cuda_emu.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "hc_q8_ref.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace strata::kernels {
namespace {
#include "fused_gr_common.cuh"
#include "hc_q8.cuh"
#include "hc_bf16.cuh"

int failures = 0;
#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

using hcref::Fixture;

struct Out {
    std::vector<float> mixed, Rout, rs, inject, lo;
    std::vector<uint8_t> q81;
    bool ok = true;
};

enum class Kind { TwoLaunch, Fused };
struct Case {
    bool apply = true, inject = true, inject_q8 = false, in_place = false, qfuse = false;
    Kind kind = Kind::TwoLaunch;
    emu::LaunchOpts opt;
};

// the fused kernel with its tasks taken from blockIdx instead of a ticket: the design this test must be able to tell from the real one
template <int T>
__global__ void gr_q8_fused_noticket_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg, unsigned* sync_) {
    const unsigned task = blockIdx.x;
    if (task < (unsigned) Q8F_DOWN) gr_q8f_down<T>(m, part, ssg, sync_, (int) (task % Q8_NRG), (int) (task / Q8_NRG));
    else gr_q8f_up<T>(m, sync_, (int) (task - Q8F_DOWN));
}

template <int T> void launch_two(const GrMulti& m, float* part, float* ssg) {
    emu::launch(emu::dim3(Q8_RG + 1, Q8_NKC), emu::dim3(THREADS), [&] { gr_down_q8_kernel<T>(m, part, ssg); });
    emu::launch(emu::dim3(UPM_BLOCKS), emu::dim3(THREADS), [&] { gr_up_q8_kernel<T>(m, part, ssg); });
}
template <int T> void launch_fused(const GrMulti& m, float* part, float* ssg, unsigned* sync, const emu::LaunchOpts& o) {
    emu::launch(emu::dim3(Q8F_GRID), emu::dim3(THREADS), [&] { gr_q8_fused_kernel<T>(m, part, ssg, sync); }, o);
}

template <int T> Out run_t(const Fixture& f, const Case& c) {
    std::vector<float> R = f.R;
    std::vector<float> Rout((size_t) T * D, -7.0f), lo((size_t) T * LR, -7.0f), rs((size_t) T * HC, -7.0f),
        inj_out((size_t) T * HC, -7.0f), mixed((size_t) T * N, -7.0f), xn((size_t) T * D * 2, 0.0f);
    std::vector<uint8_t> q81((size_t) T * (N / 32) * 36, 0x5a);
    std::vector<unsigned> qcnt(N / 32, 0), sync(kFusedGrSyncWords, 0);
    GrMulti m{};
    m.T = T;
    m.xn = xn.data();
    for (int t = 0; t < T; ++t) {
        FusedGrArgs& a = m.a[t];
        a.R = R.data() + (size_t) t * D;
        a.R_out = c.in_place ? R.data() + (size_t) t * D : Rout.data() + (size_t) t * D;
        a.apply = c.apply;
        a.bo_prev = f.bo.data() + (size_t) t * N;
        a.inj_prev = f.inj_prev.data() + (size_t) t * HC;
        a.w_norm = f.w_norm.data();
        a.w_down = reinterpret_cast<const uint16_t*>(f.q8_down.data());   // non-null, never read by the Q8_0 read
        a.w_up = reinterpret_cast<const uint16_t*>(f.q8_up.data());
        a.w_inject = c.inject ? f.w_inj.data() : nullptr;
        a.q8_down = f.q8_down.data();
        a.q8_up = f.q8_up.data();
        a.q8_inject = c.inject && c.inject_q8 ? f.q8_inj.data() : nullptr;
        a.eps = f.eps;
        a.lo = lo.data() + (size_t) t * LR;
        a.rs = rs.data() + (size_t) t * HC;
        a.inject_out = inj_out.data() + (size_t) t * HC;
        a.mixed = mixed.data() + (size_t) t * N;
        if (c.qfuse) { a.q8_mixed = q81.data() + (size_t) t * (N / 32) * 36; a.q8_cnt = qcnt.data(); }
    }
    float* part = xn.data();
    float* ssg = xn.data() + (size_t) T * Q8_NKC * PR;
    if (c.kind == Kind::TwoLaunch) launch_two<T>(m, part, ssg);
    else launch_fused<T>(m, part, ssg, sync.data(), c.opt);
    Out o;
    o.mixed = mixed; o.Rout = c.in_place ? R : Rout; o.rs = rs; o.inject = inj_out; o.lo = lo; o.q81 = q81;
    if (c.kind == Kind::Fused)
        for (unsigned w : sync) if (w != 0) o.ok = false;
    if (c.qfuse)
        for (unsigned w : qcnt) if (w != 0) o.ok = false;
    return o;
}
Out run(const Fixture& f, const Case& c) {
    switch (f.T) {
        case 1: return run_t<1>(f, c);
        case 2: return run_t<2>(f, c);
        case 3: return run_t<3>(f, c);
        case 4: return run_t<4>(f, c);
        case 5: return run_t<5>(f, c);
        case 6: return run_t<6>(f, c);
        case 7: return run_t<7>(f, c);
        default: return run_t<8>(f, c);
    }
}


// ---- the BF16 read: the staged variant's three launches against the one launch
struct Bf16Weights { std::vector<uint16_t> down, up; };
template <int T> Out run_bf16_t(const Fixture& f, const Bf16Weights& w, const Case& c) {
    std::vector<float> R = f.R;
    std::vector<float> Rout((size_t) T * D, -7.0f), lo((size_t) T * LR, -7.0f), rs((size_t) T * HC, -7.0f),
        inj_out((size_t) T * HC, -7.0f), mixed((size_t) T * N, -7.0f), xn((size_t) T * D, 0.0f);
    std::vector<uint8_t> q81((size_t) T * (N / 32) * 36, 0x5a);
    std::vector<unsigned> qcnt(N / 32, 0), sync(kFusedGrSyncWords, 0);
    GrMulti m{};
    m.T = T;
    m.xn = xn.data();
    for (int t = 0; t < T; ++t) {
        FusedGrArgs& a = m.a[t];
        a.R = R.data() + (size_t) t * D;
        a.R_out = c.in_place ? R.data() + (size_t) t * D : Rout.data() + (size_t) t * D;
        a.apply = c.apply;
        a.bo_prev = f.bo.data() + (size_t) t * N;
        a.inj_prev = f.inj_prev.data() + (size_t) t * HC;
        a.w_norm = f.w_norm.data();
        a.w_down = w.down.data(); a.w_up = w.up.data();
        a.w_inject = c.inject ? f.w_inj.data() : nullptr;
        a.eps = f.eps;
        a.lo = lo.data() + (size_t) t * LR;
        a.rs = rs.data() + (size_t) t * HC;
        a.inject_out = inj_out.data() + (size_t) t * HC;
        a.mixed = mixed.data() + (size_t) t * N;
        if (c.qfuse) { a.q8_mixed = q81.data() + (size_t) t * (N / 32) * 36; a.q8_cnt = qcnt.data(); }
    }
    const size_t dyn = (size_t) T * 2 * H_TILE * sizeof(float);
    if (c.kind == Kind::TwoLaunch) {   // the staged variant: three launches
        emu::LaunchOpts o0;
        o0.dyn_smem = dyn;
        emu::launch(emu::dim3(T, HC), emu::dim3(THREADS), [&] { gr_norm_split_kernel(m); });
        emu::launch(emu::dim3(DOWN_BLOCKS + 1), emu::dim3(THREADS), [&] { gr_down_staged_kernel<T, true>(m); }, o0);
        emu::launch(emu::dim3(UPM_BLOCKS), emu::dim3(THREADS), [&] { gr_up_multi_kernel<T, true>(m); });
    } else {
        m.a[0].hc_sync = sync.data();
        emu::LaunchOpts o = c.opt;
        o.dyn_smem = dyn;
        emu::launch(emu::dim3(kF1Grid<T>), emu::dim3(THREADS), [&] { gr_hc_fused_kernel<T>(m, sync.data()); }, o);
    }
    Out o;
    o.mixed = mixed; o.Rout = c.in_place ? R : Rout; o.rs = rs; o.inject = inj_out; o.lo = lo; o.q81 = q81;
    if (c.kind == Kind::Fused)
        for (unsigned v : sync) if (v != 0) o.ok = false;
    if (c.qfuse)
        for (unsigned v : qcnt) if (v != 0) o.ok = false;
    return o;
}
Out run_bf16(const Fixture& f, const Bf16Weights& w, const Case& c) {
    switch (f.T) {
        case 1: return run_bf16_t<1>(f, w, c);
        case 2: return run_bf16_t<2>(f, w, c);
        case 3: return run_bf16_t<3>(f, w, c);
        default: return run_bf16_t<4>(f, w, c);
    }
}

using hcref::same_bits;
using hcref::worst;
hcref::Ref reference(const Fixture& f, const Case& c) {
    return hcref::reference(f, c.apply, c.inject, c.inject_q8, f.dq_down, f.dq_up);
}

}  // namespace
}  // namespace strata::kernels

using namespace strata::kernels;

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;   // only C and D (seconds)
    const auto t_start = std::chrono::steady_clock::now();
    auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };
    std::printf("shapes: %d down tasks + %d up tasks = %d blocks of %d threads\n", Q8F_DOWN, Q8F_UP, Q8F_GRID, THREADS);

    // ---- A: the two-launch Q8_0 read against the double reference --------------------------------------------------
    for (int T : {1, 3}) {
        if (quick) break;
        Fixture f;
        f.build(T, 100u + (unsigned) T);
        for (int inj = 0; inj < 3; ++inj) {
            Case c;
            c.apply = true; c.inject = inj > 0; c.inject_q8 = inj == 2;
            const Out o = run(f, c);
            const hcref::Ref r = reference(f, c);
            const double e_mixed = worst(o.mixed, r.mixed), e_R = worst(o.Rout, r.Rout), e_rs = worst(o.rs, r.rs);
            const double e_inj = c.inject ? worst(o.inject, r.inject) : 0.0;
            std::printf("A  T=%d inject=%s: worst error / max|ref|  mixed %.2e  R_out %.2e  rs %.2e  inject %.2e   [%.1f s]\n", T,
                        inj == 0 ? "none" : inj == 1 ? "bf16" : "q8_0", e_mixed, e_R, e_rs, e_inj, secs());
            CHECK(e_mixed < 5e-6); CHECK(e_R < 1e-6); CHECK(e_rs < 1e-6); CHECK(e_inj < 5e-6);
        }
    }

    // ---- B: fused == two launches, bit for bit ---------------------------------------------------------------------
    struct Variant { bool apply, inject, inject_q8, in_place, qfuse; };
    const Variant variants[] = {
        {true, true, false, false, false}, {true, true, false, true, true}, {false, true, true, false, false},
        {false, false, false, false, true}, {true, false, false, true, false},
    };
    for (int T = 1; T <= kFusedGrMaxT; ++T) {
        if (quick) break;
        Fixture f;
        f.build(T, 500u + (unsigned) T);
        for (size_t vi = 0; vi < sizeof(variants) / sizeof(variants[0]); ++vi) {
            if (T != 1 && T != 3 && T != 8 && vi > 1) continue;   // keep the run short: every T gets variants 0 and 1, T 1/3/8 all of them
            const Variant& v = variants[vi];
            Case base;
            base.apply = v.apply; base.inject = v.inject; base.inject_q8 = v.inject_q8; base.in_place = v.in_place; base.qfuse = v.qfuse;
            base.kind = Kind::TwoLaunch;
            const Out ref = run(f, base);
            struct Disp { emu::Order order; int resident; };
            const Disp disps[] = {{emu::Order::Ascending, 12}, {emu::Order::Descending, 5}, {emu::Order::Shuffled, 3}, {emu::Order::Ascending, 1}};
            for (size_t di = 0; di < sizeof(disps) / sizeof(disps[0]); ++di) {
                if (di >= 2 && !(T == 3 || T == 8)) continue;   // shuffled order and one resident block: T 3 and 8
                Case c = base;
                c.kind = Kind::Fused;
                c.opt.order = disps[di].order; c.opt.resident = disps[di].resident; c.opt.seed = 7u + (unsigned) di;
                const Out o = run(f, c);
                const bool same = same_bits(o.mixed, ref.mixed) && same_bits(o.Rout, ref.Rout) && same_bits(o.rs, ref.rs) &&
                                  (!v.inject || same_bits(o.inject, ref.inject)) && (!v.qfuse || o.q81 == ref.q81);
                // lo is the fused kernel's hand-off buffer (the two-launch read keeps it in shared memory): against the reference
                std::printf("B  T=%d apply=%d inject=%d%s in_place=%d qfuse=%d order=%d resident=%d: %s%s  [%.1f s]\n", T, v.apply,
                            v.inject, v.inject_q8 ? "(q8)" : "", v.in_place, v.qfuse, (int) disps[di].order, disps[di].resident,
                            same ? "bit-identical" : "DIFFERS", o.ok ? "" : " COUNTERS NOT ZERO", secs());
                CHECK(same); CHECK(o.ok);
            }
            if (vi == 0) {   // lo: finite, and the silu of the reference's dot to float rounding
                Case c = base; c.kind = Kind::Fused;
                const Out o = run(f, c);
                const hcref::Ref r = reference(f, c);
                const double e = worst(o.lo, r.lo);
                CHECK(e < 5e-6);
                if (T == 1 || T == 8) std::printf("B  T=%d lo handed to the up tasks: worst error / max|ref| %.2e\n", T, e);
            }
        }
    }

    // ---- E: the BF16 read, one launch against the staged variant's three ---------------------------------------------
    for (int T = 1; T <= F1_MAX_T; ++T) {
        Fixture f;
        f.build(T, 700u + (unsigned) T);
        Bf16Weights w{Fixture::bf16_copy(f.dq_down), Fixture::bf16_copy(f.dq_up)};
        std::vector<float> bd(w.down.size()), bu(w.up.size());
        for (size_t i = 0; i < bd.size(); ++i) bd[i] = strata::kernels::f32_from_bf16(w.down[i]);
        for (size_t i = 0; i < bu.size(); ++i) bu[i] = strata::kernels::f32_from_bf16(w.up[i]);
        struct EV { bool apply, inject, in_place, qfuse; };
        const EV evs[] = {{true, true, false, false}, {true, true, true, true}, {false, false, false, true}};
        for (size_t vi = 0; vi < sizeof(evs) / sizeof(evs[0]); ++vi) {
            if (quick && (T != 3 || vi != 1)) continue;
            Case base;
            base.apply = evs[vi].apply; base.inject = evs[vi].inject; base.in_place = evs[vi].in_place; base.qfuse = evs[vi].qfuse;
            base.kind = Kind::TwoLaunch;
            const Out ref = run_bf16(f, w, base);
            if (vi == 0) {   // the staged read against the double reference (BF16 weights as floats)
                const hcref::Ref r = hcref::reference(f, base.apply, base.inject, false, bd, bu);
                const double em = worst(ref.mixed, r.mixed), el = worst(ref.lo, r.lo), er = worst(ref.rs, r.rs);
                std::printf("E  T=%d staged BF16 read vs double reference: mixed %.2e lo %.2e rs %.2e\n", T, em, el, er);
                CHECK(em < 5e-6); CHECK(el < 5e-6); CHECK(er < 1e-6);
            }
            struct Disp { emu::Order order; int resident; };
            const Disp disps[] = {{emu::Order::Ascending, 12}, {emu::Order::Descending, 4}, {emu::Order::Shuffled, 3}, {emu::Order::Ascending, 1}};
            for (size_t di = 0; di < sizeof(disps) / sizeof(disps[0]); ++di) {
                if (di >= 2 && T != 3) continue;
                Case c = base;
                c.kind = Kind::Fused;
                c.opt.order = disps[di].order; c.opt.resident = disps[di].resident; c.opt.seed = 11u + (unsigned) di;
                const Out o = run_bf16(f, w, c);
                const bool same = same_bits(o.mixed, ref.mixed) && same_bits(o.Rout, ref.Rout) && same_bits(o.rs, ref.rs) &&
                                  same_bits(o.lo, ref.lo) && (!base.inject || same_bits(o.inject, ref.inject)) &&
                                  (!base.qfuse || o.q81 == ref.q81);
                std::printf("E  T=%d apply=%d inject=%d in_place=%d qfuse=%d order=%d resident=%d: one launch vs three: %s%s  [%.1f s]\n", T,
                            base.apply, base.inject, base.in_place, base.qfuse, (int) disps[di].order, disps[di].resident,
                            same ? "bit-identical" : "DIFFERS", o.ok ? "" : " COUNTERS NOT ZERO", secs());
                CHECK(same); CHECK(o.ok);
            }
        }
    }

    // ---- C: the guard against a kernel that waits for blocks that will never run ------------------------------------
    {
        emu::g_deadlock_switches = 2000000ull;
        bool caught = false;
        try {
            std::vector<unsigned> flag(1, 0);
            // every block waits for a flag nobody sets
            emu::launch(emu::dim3(2), emu::dim3(32), [&] { emu::wait_until([&] { return flag[0] != 0; }); }, emu::LaunchOpts{2});
        } catch (const std::runtime_error&) { caught = true; }
        std::printf("C  a wait that can never be satisfied is reported: %s\n", caught ? "ok" : "NOT DETECTED");
        CHECK(caught);
    }

    // ---- D: why the ticket: with the task taken from blockIdx, a GPU that dispatches the up blocks first (and holds only as
    // many blocks as it has room for) never finishes - the emulator must see that, and must see the real kernel finish
    {
        Fixture f;
        f.build(2, 77u);
        auto try_kernel = [&](bool ticket, emu::Order order, int resident) {
            std::vector<float> xn((size_t) 2 * D * 2, 0.0f), lo(2 * LR), rs(2 * HC), inj(2 * HC), mixed(2 * N), Rout(2 * D);
            std::vector<float> R = f.R;
            std::vector<unsigned> sync(kFusedGrSyncWords, 0);
            GrMulti m{};
            m.T = 2; m.xn = xn.data();
            for (int t = 0; t < 2; ++t) {
                FusedGrArgs& a = m.a[t];
                a.R = R.data() + (size_t) t * D; a.R_out = Rout.data() + (size_t) t * D; a.apply = true;
                a.bo_prev = f.bo.data() + (size_t) t * N; a.inj_prev = f.inj_prev.data() + (size_t) t * HC;
                a.w_norm = f.w_norm.data(); a.w_down = reinterpret_cast<const uint16_t*>(f.q8_down.data());
                a.w_up = reinterpret_cast<const uint16_t*>(f.q8_up.data()); a.w_inject = f.w_inj.data();
                a.q8_down = f.q8_down.data(); a.q8_up = f.q8_up.data(); a.eps = f.eps;
                a.lo = lo.data() + (size_t) t * LR; a.rs = rs.data() + (size_t) t * HC;
                a.inject_out = inj.data() + (size_t) t * HC; a.mixed = mixed.data() + (size_t) t * N;
            }
            float* part = xn.data();
            float* ssg = xn.data() + (size_t) 2 * Q8_NKC * PR;
            emu::LaunchOpts o;
            o.order = order; o.resident = resident;
            emu::g_deadlock_switches = 3000000ull;
            try {
                if (ticket) emu::launch(emu::dim3(Q8F_GRID), emu::dim3(THREADS), [&] { gr_q8_fused_kernel<2>(m, part, ssg, sync.data()); }, o);
                else emu::launch(emu::dim3(Q8F_GRID), emu::dim3(THREADS), [&] { gr_q8_fused_noticket_kernel<2>(m, part, ssg, sync.data()); }, o);
            } catch (const std::runtime_error&) { return false; }
            return true;
        };
        const bool real = try_kernel(true, emu::Order::Descending, 2);
        const bool naive = try_kernel(false, emu::Order::Descending, 2);
        std::printf("D  up blocks dispatched first, 2 blocks resident: ticketed kernel %s, blockIdx-ordered kernel %s\n",
                    real ? "finishes" : "HANGS", naive ? "finishes (the check cannot tell)" : "hangs (as it must)");
        CHECK(real); CHECK(!naive);
        emu::g_deadlock_switches = 200000000ull;
    }

    if (failures) { std::printf("hc_q8_emu_test: %d failure(s)\n", failures); return 1; }
    std::printf("hc_q8_emu_test: ok (%.1f s)\n", secs());
    return 0;
}
