// Core tests for the Siemens INF reader, ported one-for-one from the Rust unit tests in
// crates/seq2waveform/src/dsv/inf.rs, with SAMPLE_INF included verbatim.
//
// Two things here are easy to get subtly wrong and expensive to notice later:
//
// * The MaxAdcSegmentLength merge. Siemens splits a long readout into chunks; merging consecutive
//   same-dwell contiguous chunks restores the logical ADC that matches the `.seq`. The test
//   asserts the merged sample times equal the concatenation of the originals, which is the claim
//   that the merge is exact rather than merely plausible.
// * The RF center. A symmetric pulse's center is the window middle, exactly; an asymmetric one's
//   is the sub-raster RFD peak. Getting that backwards shifts every k-space reference by up to
//   half a pulse.
//
// No test framework on purpose: one translation unit, plain checks, non-zero exit on failure.

#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "seq2waveform/detail/inf_detail.hpp"
#include "seq2waveform/dsv.hpp"
#include "seq2waveform/inf.hpp"
#include "seq2waveform/time.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

void close(double a, double b, const char* what, double tol = 1e-12) {
    if (!(std::fabs(a - b) < tol)) {
        std::printf("FAIL: %s (%.17g vs %.17g)\n", what, a, b);
        ++failures;
    }
}

// One ADC block (start 100us, dur 16us, 4 samples => 4us dwell), then one event block
// (start 9140us) with a Freq/Phase set row, an RF pulse row, and a SYNC trigger row.
const char* kSampleInf =
    "[INFO]\n"
    "adc MeasHeader 100 16\n"
    "ulScanCounter                 : 1\n"
    "ushSamplesInScan              : 4               ;# of samples\n"
    "[INFO]\n"
    "* EventBlock 9140 200\n"
    "|=================================================|\n"
    "| Rel.Time | TXFlip/Dur | Freq/Phase | SYNC/Dur |\n"
    "|-------------------------------------------------|\n"
    "|      190 |                  | [sX zSet]: -32000/ 115.20 |                       |\n"
    "|      200 | [sX]: 130.8/6000 |                           |                       |\n"
    "|        0 |                  |                           | [undef]: EXTRIG0/  10 |\n";

s2w::Inf parse_inf(const char* text) {
    std::istringstream in(text);
    return s2w::read_inf_stream(in);
}

s2w::AdcSegment seg(double start, std::size_t n, double dwell) {
    s2w::AdcSegment s;
    s.start = start;
    s.duration = static_cast<double>(n) * dwell;
    s.num_samples = n;
    return s;
}

void parses_adc_segments() {
    const s2w::Inf inf = parse_inf(kSampleInf);
    check(inf.adc.size() == 1, "one ADC segment");
    if (!inf.adc.empty()) {
        check(inf.adc[0].num_samples == 4, "four samples");
        close(inf.adc[0].dwell(), 4e-6, "dwell is duration / samples");
    }
    check(inf.adc_sample_times().size() == 4, "four sample times");
}

void merges_contiguous_adc_segments_into_logical_events() {
    const double d = 2e-6;
    // two contiguous chunks (one long readout split by MaxAdcSegmentLength), then a gap, then a
    // separate readout.
    const s2w::AdcSegment a = seg(1e-3, 5000, d);
    const s2w::AdcSegment b = seg(1e-3 + 5000.0 * d, 200, d);          // starts where a ends
    const s2w::AdcSegment c = seg(1e-3 + 5200.0 * d + 50e-6, 100, d);  // 50 us gap -> new event
    const std::vector<s2w::AdcSegment> merged = s2w::detail::merge_contiguous({a, b, c});
    check(merged.size() == 2, "three chunks merge into two events");
    if (merged.size() == 2) {
        check(merged[0].num_samples == 5200, "the first two chunks joined");
        check(merged[1].num_samples == 100, "the gapped chunk stayed separate");
        // the merged times match the concatenation of the originals
        std::vector<s2w::Ps> concat = a.sample_times();
        const std::vector<s2w::Ps> bt = b.sample_times();
        concat.insert(concat.end(), bt.begin(), bt.end());
        const std::vector<s2w::Ps> got = merged[0].sample_times();
        check(got.size() == concat.size(), "merged sample count matches");
        bool same = got.size() == concat.size();
        for (std::size_t i = 0; i < got.size() && i < concat.size(); ++i) {
            same = same && std::fabs(got[i].secs() - concat[i].secs()) < 1e-12;
        }
        check(same, "the merge is exact, not just plausible");
    }
}

void a_dwell_change_starts_a_new_event() {
    // No Rust counterpart. Contiguity alone must not merge: two readouts that abut but sample at
    // different rates are different events, and merging them would silently resample one.
    const s2w::AdcSegment a = seg(0.0, 100, 2e-6);
    const s2w::AdcSegment b = seg(200e-6, 100, 4e-6);  // abuts a, but twice the dwell
    const std::vector<s2w::AdcSegment> merged = s2w::detail::merge_contiguous({a, b});
    check(merged.size() == 2, "a dwell change starts a new event");
}

void extracts_triggers() {
    const s2w::Inf inf = parse_inf(kSampleInf);
    const std::vector<s2w::SyncEvent> trig = inf.triggers();
    check(trig.size() == 1, "one trigger");
    if (!trig.empty()) {
        check(trig[0].name == "EXTRIG0", "trigger name");
        close(trig[0].time, 9140e-6, "time is block start + Rel.Time");
        close(trig[0].duration, 10e-6, "duration is us -> s");
    }
}

void extracts_rf_events_with_freq_phase() {
    const s2w::Inf inf = parse_inf(kSampleInf);
    const std::vector<s2w::RfEvent> rf = inf.rf_events();
    check(rf.size() == 1, "one RF event");
    if (!rf.empty()) {
        close(rf[0].start, 9340e-6, "start is block 9140 + Rel.Time 200");
        close(rf[0].duration, 6000e-6, "duration is us -> s");
        close(rf[0].flip_deg, 130.8, "flip angle", 1e-9);
        close(rf[0].freq, -32000.0, "freq comes from the preceding zSet row", 1e-9);
        close(rf[0].phase_deg, 115.20, "phase comes from the same row", 1e-9);
    }
}

s2w::DsvChannel rfd_with(const std::vector<double>& times, const std::vector<double>& values) {
    s2w::DsvChannel ch;
    ch.raster_time = 1e-6;
    ch.unit = "Volt";
    for (const double t : times) {
        ch.waveform.t.push_back(s2w::Ps::from_secs(t));
    }
    ch.waveform.v = values;
    ch.num_samples = 9;
    return ch;
}

void rf_center_is_the_window_middle() {
    const s2w::DsvChannel rfd = rfd_with({0.5e-6, 4.5e-6, 8.5e-6}, {0.0, 5.0, 0.0});
    s2w::RfEvent e;
    e.start = 0.0;
    e.duration = 9e-6;
    const auto co = e.center_and_offset(rfd);
    close(co.first, 4.5e-6, "symmetric: the peak is the middle, so use the exact middle");
    check(std::fabs(co.second) < 1e-6, "not flagged off-center");
}

void rf_center_uses_peak_for_asymmetric() {
    // a min-phase-like envelope: a long ramp up to a sharp peak near the END of the window
    const s2w::DsvChannel rfd = rfd_with({0.5e-6, 7.5e-6, 8.5e-6}, {0.0, 5.0, 0.0});
    s2w::RfEvent e;
    e.start = 0.0;
    e.duration = 9e-6;
    const auto co = e.center_and_offset(rfd);
    // the window middle is 4.5 us but the peak is ~7.5 us -> center near the peak, flagged
    check(std::fabs(co.first - 7.5e-6) < 1e-6, "asymmetric: center moves to the peak");
    check(std::fabs(co.second) > 1e-6, "flagged off-center");
}

void split_cell_grammar() {
    // No Rust counterpart; this is the whole table-cell grammar in one place.
    const auto a = s2w::detail::split_cell("[sX]: 130.8/6000");
    check(a.has_value() && a->first == "130.8" && a->second == "6000", "label prefix is stripped");
    const auto b = s2w::detail::split_cell(" 42 / 7 ");
    check(b.has_value() && b->first == "42" && b->second == "7", "no label is fine");
    check(!s2w::detail::split_cell("no slash here").has_value(), "a cell without '/' is empty");
    check(!s2w::detail::split_cell("[sX]: / 10").has_value(), "a blank first part is empty");
    const auto c = s2w::detail::split_cell("[undef]: EXTRIG0/  10");
    check(c.has_value() && c->first == "EXTRIG0" && c->second == "10", "a name is a valid part");
}

void lead_in_is_the_second_block_start() {
    // Block 0 is the Siemens GLOBzSet setup; block 1 is the first Pulseq block, and its start IS
    // the lead-in. With only one block there is nothing to report.
    const s2w::Inf one = parse_inf(kSampleInf);
    check(one.blocks.size() == 1, "the sample INF has one event block");
    close(one.lead_in(), 0.0, "a single-block INF has no lead-in");

    const char* two =
        "* EventBlock 0 20\n"
        "| Rel.Time | SYNC/Dur |\n"
        "|        0 | [undef]: EXTRIG0/ 10 |\n"
        "* EventBlock 1020 200\n"
        "|        0 | [undef]: EXTRIG0/ 10 |\n";
    const s2w::Inf inf = parse_inf(two);
    check(inf.blocks.size() == 2, "two event blocks");
    close(inf.lead_in(), 1020e-6, "the lead-in is the second block's start");
}

}  // namespace

int main() {
    parses_adc_segments();
    merges_contiguous_adc_segments_into_logical_events();
    a_dwell_change_starts_a_new_event();
    extracts_triggers();
    extracts_rf_events_with_freq_phase();
    rf_center_is_the_window_middle();
    rf_center_uses_peak_for_asymmetric();
    split_cell_grammar();
    lead_in_is_the_second_block_start();

    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
