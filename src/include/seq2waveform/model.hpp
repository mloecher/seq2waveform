// Parsed Pulseq sequence data. Blocks reference event libraries by id; 0 means no event.
// Units: seconds, Hz for RF, Hz/m for gradients, radians for phase.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace s2w {

/// Pulseq file format version.
struct Version {
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t revision = 0;
};

inline bool operator==(const Version& a, const Version& b) {
    return a.major == b.major && a.minor == b.minor && a.revision == b.revision;
}
inline bool operator!=(const Version& a, const Version& b) { return !(a == b); }

/// Intended use of an RF pulse (the `use` field, stored in v1.5+).
enum class RfUse { Excitation, Refocusing, Inversion, Saturation, Preparation, Other, Undefined };

/// Map the single-character code stored in the file (the initial of the use name). Anything
/// unrecognised, a missing field included, is `Undefined`.
RfUse rf_use_from_char(char c);

/// Lowercase use name, e.g. "excitation", "undefined".
const char* rf_use_name(RfUse u);

/// An RF event (`[RF]` library). Amplitude in Hz, times in seconds.
struct Rf {
    double amplitude = 0.0;
    std::uint32_t mag_shape = 0;
    std::uint32_t phase_shape = 0;
    std::uint32_t time_shape = 0;
    double center = 0.0;
    double delay = 0.0;
    double freq_ppm = 0.0;
    double phase_ppm = 0.0;
    double freq = 0.0;
    double phase = 0.0;
    RfUse use = RfUse::Undefined;
};

/// A trapezoid gradient (`[TRAP]`). Amplitude Hz/m, ramp/flat/delay in seconds.
struct Trap {
    double amplitude = 0.0;
    double rise = 0.0;
    double flat = 0.0;
    double fall = 0.0;
    double delay = 0.0;
};

/// An arbitrary (shaped) gradient (`[GRADIENTS]`). Amplitude Hz/m.
struct ArbitraryGrad {
    double amplitude = 0.0;
    double first = 0.0;
    double last = 0.0;
    std::uint32_t amp_shape = 0;
    std::uint32_t time_shape = 0;
    double delay = 0.0;
};

/// A gradient library entry: a trapezoid or an arbitrary shape.
using Gradient = std::variant<Trap, ArbitraryGrad>;

/// An ADC event (`[ADC]`). dwell/delay in seconds.
struct Adc {
    std::uint32_t num = 0;
    double dwell = 0.0;
    double delay = 0.0;
    double freq_ppm = 0.0;
    double phase_ppm = 0.0;
    double freq = 0.0;
    double phase = 0.0;
    std::uint32_t phase_shape = 0;
};

/// A trigger event (`extension TRIGGERS`). delay/duration in seconds.
struct Trigger {
    std::uint32_t trig_type = 0;
    std::uint32_t channel = 0;
    double delay = 0.0;
    double duration = 0.0;
};

/// One node of an extension list (`[EXTENSIONS]`): a linked list of typed references.
struct ExtNode {
    std::uint32_t type_id = 0;
    std::uint32_t ref_id = 0;
    std::uint32_t next_id = 0;
};

/// A shape from `[SHAPES]`, stored in its on-disk (possibly compressed) form.
struct Shape {
    std::size_t num_samples = 0;
    std::vector<double> data;

    /// Decompress: run-length-encoded derivative, then cumulative sum.
    ///
    /// A run of a repeated derivative value `v` is stored as `v, v, k`, meaning `v` repeated
    /// `k + 2` times. When `data.size() == num_samples` the shape is stored uncompressed (the data
    /// are the samples) and is returned without a cumulative sum. The result's length is whatever
    /// the run lengths expand to, which is not forced to `num_samples`.
    std::vector<double> decompressed() const;
};

/// One sequence block (a row of `[BLOCKS]`). Event fields are library ids (0 = none).
struct Block {
    std::uint32_t num = 0;
    /// Block duration, seconds.
    double duration = 0.0;
    std::uint32_t rf = 0;
    std::uint32_t gx = 0;
    std::uint32_t gy = 0;
    std::uint32_t gz = 0;
    std::uint32_t adc = 0;
    std::uint32_t ext = 0;
};

/// How a centers-sampled arbitrary gradient (`time_shape_id = 0`) becomes piecewise-linear corners.
enum class GradMode {
    /// Reconstruct raster-edge vertices (`restoreAdditionalShapeSamples`) -- matches Pulseq/h5
    /// exactly. Can staircase on fast waveforms.
    Pulseq,
    /// Connect the stored center samples directly (with `first`/`last` at the edges). Smooths
    /// arbitrary shapes to the DSV, but leaves trapezoids sharp.
    Centers,
    /// Resample the whole channel at raster centers to match the DSV representation.
    /// Rounds trapezoid corners and smooths arbitrary shapes.
    Dsv,
};

/// A parsed Pulseq sequence: the block list plus the event libraries.
struct Sequence {
    Version version;
    /// How centers-sampled arbitrary gradients are rendered (default `GradMode::Pulseq`).
    GradMode grad_mode = GradMode::Pulseq;
    /// Raw `[DEFINITIONS]` key -> value string; parse on demand via `definition_f64`.
    std::map<std::string, std::string> definitions;
    std::vector<Block> blocks;
    std::map<std::uint32_t, Rf> rf;
    std::map<std::uint32_t, Gradient> gradients;
    std::map<std::uint32_t, Adc> adc;
    std::map<std::uint32_t, Shape> shapes;
    std::map<std::uint32_t, Trigger> triggers;
    std::map<std::uint32_t, ExtNode> ext_nodes;
    /// Extension type id -> name (e.g. 1 -> "TRIGGERS"), from `extension <NAME> <id>` headers.
    std::map<std::uint32_t, std::string> ext_type_names;
    std::map<std::string, std::string> signature;

    /// First whitespace-separated value of a `[DEFINITIONS]` entry, if it parses as a number.
    std::optional<double> definition_f64(const std::string& key) const;
};

}  // namespace s2w
