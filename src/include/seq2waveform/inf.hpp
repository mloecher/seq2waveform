// Siemens INF log reader. ADC headers give acquisition segments; event-block tables
// give RF, frequency/phase and SYNC events. Non-numeric table rows are ignored.

#pragma once

#include <cstddef>
#include <istream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "seq2waveform/dsv.hpp"
#include "seq2waveform/time.hpp"

namespace s2w {

/// One ADC acquisition segment (one `adc MeasHeader` block).
struct AdcSegment {
    /// Segment start, seconds (Siemens frame).
    double start = 0.0;
    /// Segment duration, seconds.
    double duration = 0.0;
    /// Samples in this segment.
    std::size_t num_samples = 0;

    /// Dwell time, seconds.
    double dwell() const { return duration / static_cast<double>(num_samples); }

    /// Center-of-dwell sample times: `start + dwell*((0..N)+0.5)`.
    std::vector<Ps> sample_times() const;

    /// Append those same times to `out`, so a caller concatenating many segments fills one
    /// reserved array instead of allocating a vector per segment and copying it out.
    void append_sample_times(std::vector<Ps>& out) const;
};

/// A SYNC-column event, e.g. an external trigger.
struct SyncEvent {
    std::string name;
    /// Time, seconds (Siemens frame: block start + Rel.Time).
    double time = 0.0;
    double duration = 0.0;
};

/// An RF event from the TXFlip/Dur column, with freq/phase from the block's Freq/Phase row.
struct RfEvent {
    /// Pulse start, seconds (Siemens frame; the rf delay point = block start + Rel.Time).
    double start = 0.0;
    /// Pulse duration, seconds.
    double duration = 0.0;
    double flip_deg = 0.0;
    /// Frequency offset, Hz (from the Freq/Phase `zSet` cell).
    double freq = 0.0;
    double phase_deg = 0.0;

    /// Excitation center (s). See `center_and_offset`.
    double center(const DsvChannel& rfd) const;

    /// `(center_s, peak - middle)`.
    ///
    /// Symmetric pulses (sinc/gauss, Pulseq `centerpos = 0.5`): the RFD peak is at the window
    /// middle, so the center is `start + duration/2`, exactly. Asymmetric pulses
    /// (min-phase/SLR/adiabatic): the peak is off-middle and the excitation is *there*, so the
    /// sub-raster RFD peak is used instead, as `calcRfCenter` does. The returned `peak - middle`
    /// lets the caller flag off-center pulses, whose center is then raster-limited.
    std::pair<double, double> center_and_offset(const DsvChannel& rfd) const;

private:
    /// Sub-raster peak time of the |B1| RFD envelope within the window. A flat max plateau returns
    /// its midpoint; a sharp single peak is refined by parabolic interpolation of the raster
    /// samples. Empty when there is no usable RFD.
    std::optional<double> rfd_peak(const DsvChannel& rfd) const;
};

/// One data row of an event-block table.
struct EventRow {
    /// Time relative to the block start, seconds.
    double rel_time = 0.0;
    /// The trimmed column cells.
    std::vector<std::string> cells;
};

/// One event-block table (`* EventBlock`).
struct EventBlock {
    /// Block start, seconds (Siemens frame).
    double start = 0.0;
    std::vector<EventRow> rows;
};

/// A single parse of an INF file.
struct Inf {
    std::vector<AdcSegment> adc;
    std::vector<EventBlock> blocks;
    /// Event-table header names, recorded from the first `Rel.Time` row.
    std::vector<std::string> columns;

    /// All ADC sample times, concatenated in segment order.
    std::vector<Ps> adc_sample_times() const;

    /// ADC split into LOGICAL events: consecutive `MeasHeader` segments that are contiguous in
    /// time with the same dwell are merged back into one.
    ///
    /// Siemens splits a long acquisition by `MaxAdcSegmentLength` (5000-sample chunks, say); this
    /// restores the single logical ADC so the grouping matches the `.seq`'s per-event ADC. The
    /// merge is exact: contiguous same-dwell chunks reproduce the original times.
    std::vector<AdcSegment> adc_events() const;

    /// SYNC-column events (triggers and the like). Filter by `name` in the caller.
    std::vector<SyncEvent> triggers() const;

    /// RF events from the TXFlip/Dur column, with freq/phase taken from the nearest preceding
    /// Freq/Phase (`zSet`) row in the same block.
    std::vector<RfEvent> rf_events() const;

    /// Index of the first column whose header starts with `prefix`.
    std::optional<std::size_t> col(const std::string& prefix) const;

    /// The Pulseq-to-Siemens lead-in (s): the start of the *second* event block.
    ///
    /// Block 0 is the Siemens `GLOBzSet` setup; block 1 is the first Pulseq block, and its Siemens
    /// start time IS the lead-in (~1020 us). Add it to a Pulseq-frame time to reach the Siemens
    /// frame, where the DSV waveforms live. Never fit it.
    double lead_in() const;
};

/// Parse an INF from any stream (single pass).
Inf read_inf_stream(std::istream& in);

/// Parse an INF file from disk.
Inf read_inf_file(const std::string& path);

}  // namespace s2w
