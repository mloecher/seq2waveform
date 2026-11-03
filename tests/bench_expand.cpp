// Time parsing, waveform expansion and k-space without Python.
// Usage: bench_expand <file.seq>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "seq2waveform/parser.hpp"
#include "seq2waveform/sample.hpp"
#include "seq2waveform/waveform.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fputs("usage: bench_expand <file.seq>\n", stderr);
        return 2;
    }
    const std::string path = argv[1];

    try {
        auto t = Clock::now();
        const s2w::Sequence seq = s2w::parse_file(path);
        std::printf("parse:             %8.1f ms  (%zu blocks)\n", ms_since(t), seq.blocks.size());

        t = Clock::now();
        const s2w::Waveforms w = s2w::waveforms(seq);
        std::printf(
            "waveforms() full:  %8.1f ms  (gx %zu gy %zu gz %zu corners, rf %zu pulses, adc %zu "
            "events)\n",
            ms_since(t), w.gx.size(), w.gy.size(), w.gz.size(), w.rf.size(), w.adc.size());

        t = Clock::now();
        const std::vector<s2w::Ps> adc = s2w::adc_times(seq);
        std::printf("adc_times():       %8.1f ms  (%zu samples)\n", ms_since(t), adc.size());

        double gtotal = 0.0;
        for (const char ax : {'x', 'y', 'z'}) {
            t = Clock::now();
            const s2w::PiecewiseLinear g = s2w::gradient(seq, ax);
            const double dt = ms_since(t);
            gtotal += dt;
            std::printf("gradient('%c'):     %8.1f ms  (%zu corners)\n", ax, dt, g.size());
        }
        std::printf("  (3 gradients:    %8.1f ms)\n", gtotal);

        t = Clock::now();
        const std::vector<s2w::RfMeta> rf = s2w::rf_events_meta(seq);
        std::printf("rf_events_meta():  %8.1f ms  (%zu events)\n", ms_since(t), rf.size());

        // --- the k-space pipeline, phase by phase -------------------------------------------
        const s2w::SequenceSource src(seq);

        t = Clock::now();
        const s2w::Sampler sampler = src.sampler();
        std::printf("\nsampler() build:   %8.1f ms\n", ms_since(t));

        t = Clock::now();
        const std::vector<s2w::SampleSpec> specs = src.adc_specs();
        std::size_t n_samples = 0;
        for (const s2w::SampleSpec& s : specs) {
            n_samples += s.times.size();
        }
        std::printf("adc_specs():       %8.1f ms  (%zu events, %zu samples)\n", ms_since(t),
                    specs.size(), n_samples);

        // Where does adc_specs() lose to a flat build? Same 21.5 M values as adc_times(), but
        // into one vector per event instead of one for the run. This isolates that.
        t = Clock::now();
        const std::vector<s2w::AdcEvent> evs = s2w::adc_events(seq);
        std::size_t ev_samples = 0;
        for (const s2w::AdcEvent& e : evs) {
            ev_samples += e.t.size();
        }
        std::printf("  adc_events():    %8.1f ms  (%zu vectors, %zu samples)\n", ms_since(t),
                    evs.size(), ev_samples);

        // kspace_flat zero-fills three result arrays it then overwrites entirely. Is that free
        // (fresh zero pages from the OS) or is it a real pass?
        t = Clock::now();
        {
            std::vector<double> z0(n_samples, 0.0), z1(n_samples, 0.0), z2(n_samples, 0.0);
            std::printf("  3x zero-fill:    %8.1f ms  (%.0f MB)\n", ms_since(t),
                        3.0 * static_cast<double>(n_samples) * 8.0 / 1e6);
        }

        t = Clock::now();
        std::vector<s2w::Trace> traces;
        traces.reserve(specs.size());
        for (const s2w::SampleSpec& s : specs) {
            traces.push_back(sampler.kspace(s.times, s.reference));
        }
        const double kspace_ms = ms_since(t);
        std::size_t produced = 0;
        for (const s2w::Trace& tr : traces) {
            produced += tr.x.size();
        }
        std::printf("kspace sampling:   %8.1f ms  (%zu k-space samples)\n", kspace_ms, produced);

        t = Clock::now();
        const s2w::FlatTrace flat = src.adc_kspace_flat();
        std::printf("adc_kspace_flat(): %8.1f ms  (%zu samples)\n", ms_since(t), flat.t.size());

        t = Clock::now();
        const s2w::Trace dense = src.gradients_dense(1e-4);
        std::printf("gradients_dense(): %8.1f ms  (%zu samples)\n", ms_since(t), dense.x.size());

        if (kspace_ms > 0.0) {
            std::printf("\nthroughput (sampling only): %.1f M samples/s\n",
                        static_cast<double>(n_samples) / (kspace_ms * 1e3));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
