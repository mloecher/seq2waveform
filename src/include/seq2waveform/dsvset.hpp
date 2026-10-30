// A Siemens DSP-Simulator measurement set: three gradient channels, the RFD envelope and the INF.
//
// The DSV-side analogue of a `Sequence`, in the Siemens frame. It is a `WaveformSource`, so the
// whole sampling and k-space pipeline runs on it unchanged -- which is the point: it lets a DSV+INF
// set stand in for a `.seq`, and predict trajectories for sequences that have no `.seq` at all.
//
// The gradients, the ADC and the excitations all share the ~1020 us lead-in, so k-space -- an
// integral *difference* -- comes out right with no lead-in juggling.
//
// Ported from the `impl WaveformSource for DsvSet` block in the Rust `dsv.rs`. It lives in its own
// header rather than in `dsv.hpp` because it needs `sample.hpp` and `inf.hpp`, which the channel
// reader does not; Rust can keep them in one file, C++ would rather not make `dsv.hpp` heavy.

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "seq2waveform/dsv.hpp"
#include "seq2waveform/inf.hpp"
#include "seq2waveform/sample.hpp"

namespace s2w {

/// Three gradient channels, the RFD envelope and the INF, behind the `WaveformSource` interface.
class DsvSet : public WaveformSource {
public:
    /// Build from already-decoded channels and a parsed INF.
    DsvSet(DsvChannel grx, DsvChannel gry, DsvChannel grz, DsvChannel rfd, Inf inf, double gamma);

    /// Read a measurement set from the four `.dsv` channel files plus the INF on disk.
    static DsvSet from_paths(const std::string& grx, const std::string& gry,
                             const std::string& grz, const std::string& rfd,
                             const std::string& inf, double gamma);

    /// The gradient channel for `axis`, converted from the DSV's mT/m to Hz/m.
    ///
    /// `Hz/m = mT/m * 1e-3 (T/m) * gamma (Hz/T)`. The times are already interval-centered in the
    /// Siemens frame, so there is no shift.
    PiecewiseLinear gradient(char axis) const override;

    /// ADC events with Siemens's `MaxAdcSegmentLength` chunking undone, so the grouping matches
    /// the `.seq`'s per-event ADC. freq and phase are 0: the INF does not record them.
    std::vector<AdcEvent> adc_events() const override;

    /// Excitations as `(center, freq_hz, phase_rad)`, sorted by center.
    ///
    /// The INF has no excite/refocus type, so every RF event is treated as an excitation. The
    /// center is the window middle for a symmetric pulse (exact) and the sub-raster RFD peak for
    /// an off-center one (min-phase, SLR, adiabatic). Warns on stderr when any pulse is
    /// peak-off-center, because that center is raster-limited where the middle is not.
    std::vector<std::tuple<Ps, double, double>> excitations() const override;

    /// SYNC events as triggers. `channel` and `trig_type` are 0: the SYNC column carries neither.
    std::vector<TriggerEvent> trigger_events() const override;

    /// The span of the gradient channels, which covers the whole run (Siemens frame, lead-in
    /// included).
    double duration() const override;

    /// The gyromagnetic ratio (Hz/T) this set uses. Divide an Hz/m output by `gamma * 1e-3` for
    /// mT/m, or by `gamma` for T/m.
    double gamma() const { return gamma_; }

    const DsvChannel& channel(char axis) const;
    const DsvChannel& rfd() const { return rfd_; }
    const Inf& inf() const { return inf_; }

private:
    DsvChannel grx_;
    DsvChannel gry_;
    DsvChannel grz_;
    DsvChannel rfd_;
    Inf inf_;
    double gamma_;
};

}  // namespace s2w
