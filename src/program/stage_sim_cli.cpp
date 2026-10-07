// stage_sim: what --pipeline-windows 2 should give on a layer split, from the per-stage ms of a serve log.
//   stage_sim [--preset brief|recording] [--gpu0 ms --cpu0 ms --gpu1 ms --cpu1 ms] [--layers0 N --layers1 N]
//             [--fixed ms] [--undo ms] [--d share] [--jitter 0..1] [--stretch f] [--tokens per_window] [--windows N] [--h a,b,c]
// Prints the predicted ms/window (and the gain over serial, and tok/s with --tokens) per pool policy and on-path share
// h.  Compare the row for the measured on-path share (O/N of the `strata pipeline:` line) with its ms/window.
#include "strata/program/stage_sim.hpp"

#include <cstdlib>
#include <cstring>
#include <sstream>

using namespace strata::program;

int main(int argc, char** argv) {
    std::string preset = "recording";
    double v[4] = {-1, -1, -1, -1};   // gpu0 cpu0 gpu1 cpu1
    int n0 = 24, n1 = 24, n = 4000;
    double fixed = -1, undo = -1, d = 0.15, tokens = 0, jitter = 0.3, stretch = 1.0;
    std::vector<double> hs = {0.3, 0.4, 0.5, 0.7};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "stage_sim: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--preset") preset = next();
        else if (a == "--gpu0") v[0] = std::atof(next());
        else if (a == "--cpu0") v[1] = std::atof(next());
        else if (a == "--gpu1") v[2] = std::atof(next());
        else if (a == "--cpu1") v[3] = std::atof(next());
        else if (a == "--layers0") n0 = std::max(1, std::atoi(next()));
        else if (a == "--layers1") n1 = std::max(1, std::atoi(next()));
        else if (a == "--fixed") fixed = std::atof(next());
        else if (a == "--undo") undo = std::atof(next());
        else if (a == "--d") d = std::atof(next());
        else if (a == "--tokens") tokens = std::atof(next());
        else if (a == "--jitter") jitter = std::clamp(std::atof(next()), 0.0, 1.0);
        else if (a == "--stretch") stretch = std::max(1.0, std::atof(next()));
        else if (a == "--windows") n = std::max(10, std::atoi(next()));
        else if (a == "--h") {
            hs.clear();
            std::stringstream ss(next());
            for (std::string t; std::getline(ss, t, ',');) hs.push_back(std::atof(t.c_str()));
        } else {
            std::fprintf(stderr, "usage: stage_sim [--preset brief|recording] [--gpu0 --cpu0 --gpu1 --cpu1 ms (stage totals)] "
                                 "[--layers0 N --layers1 N] [--fixed ms] [--undo ms] [--d share] [--jitter 0..1] [--stretch f>=1] "
                                 "[--tokens n] [--windows N] [--h a,b,c]\n");
            return 2;
        }
    }
    StageSimParams p = preset == "brief" ? stage_sim_brief() : stage_sim_recording();
    if (preset != "brief" && preset != "recording") { std::fprintf(stderr, "stage_sim: unknown preset %s\n", preset.c_str()); return 2; }
    if (v[0] >= 0 || v[1] >= 0 || v[2] >= 0 || v[3] >= 0 || n0 != 24 || n1 != 24) {
        const double f = p.fixed_ms;
        p = stage_sim_uniform(n0, v[0] >= 0 ? v[0] : p.gpu_total(0), v[1] >= 0 ? v[1] : p.cpu_total(0),
                              n1, v[2] >= 0 ? v[2] : p.gpu_total(1), v[3] >= 0 ? v[3] : p.cpu_total(1));
        p.fixed_ms = f;
    }
    if (fixed >= 0) p.fixed_ms = fixed;
    if (undo >= 0) p.undo_ms = undo;
    p.jitter = jitter;
    p.overlap_stretch = stretch;
    stage_sim_print_table(stdout, p, d, tokens, n, hs);
    return 0;
}
