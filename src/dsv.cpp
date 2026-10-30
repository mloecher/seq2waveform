#include "seq2waveform/dsv.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "seq2waveform/detail/dsv_detail.hpp"
#include "seq2waveform/error.hpp"
#include "seq2waveform/scan.hpp"
#include "seq2waveform/time.hpp"

namespace s2w {
namespace detail {

std::pair<PiecewiseLinear, std::size_t> rle_to_piecewise(const std::vector<std::int64_t>& rle,
                                                         double raster, double vertfactor) {
    std::vector<Ps> t;
    std::vector<double> v;
    std::int64_t acc = 0;                  // cumulative sum = value * VERTFACTOR at sample k
    std::size_t k = 0;                     // current sample index
    std::optional<std::int64_t> prev;      // previous delta
    std::optional<std::size_t> last_idx;   // index of the last emitted corner

    std::size_t i = 0;
    const std::size_t n_rle = rle.size();
    while (i < n_rle) {
        // Expand the next RLE token into (delta, repeats). `d, d, k` means d repeated k + 2 times;
        // a delta not followed by a copy of itself stands alone.
        std::int64_t d = 0;
        std::size_t reps = 1;
        if (i + 2 < n_rle && rle[i + 1] == rle[i]) {
            d = rle[i];
            // Rust: `(rle[i + 2] + 2).max(0) as usize`. The clamp keeps a corrupt count from
            // overflowing on the way in, as in the shape decoder.
            std::int64_t r = rle[i + 2];
            constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
            r = r > kMax - 2 ? kMax - 2 : r;
            reps = static_cast<std::size_t>(r + 2 < 0 ? 0 : r + 2);
            i += 3;
        } else {
            d = rle[i];
            i += 1;
        }

        for (std::size_t rep = 0; rep < reps; ++rep) {
            acc += d;
            if (k == 0) {
                t.push_back(Ps::from_secs(0.5 * raster));
                v.push_back(static_cast<double>(acc) / vertfactor);
                last_idx = 0;
            } else if (!prev.has_value() || *prev != d) {
                // The slope changed at sample k-1, so that sample is a corner; its value is the
                // accumulator *before* this delta, i.e. acc - d.
                const std::size_t idx = k - 1;
                if (!last_idx.has_value() || *last_idx != idx) {
                    t.push_back(Ps::from_secs((static_cast<double>(idx) + 0.5) * raster));
                    v.push_back(static_cast<double>(acc - d) / vertfactor);
                    last_idx = idx;
                }
            }
            prev = d;
            ++k;
        }
    }

    // Always keep the final sample, so the channel's span is right even when it ends mid-run.
    if (k > 0) {
        const std::size_t idx = k - 1;
        if (!last_idx.has_value() || *last_idx != idx) {
            t.push_back(Ps::from_secs((static_cast<double>(idx) + 0.5) * raster));
            v.push_back(static_cast<double>(acc) / vertfactor);
        }
    }

    PiecewiseLinear w;
    w.t = std::move(t);
    w.v = std::move(v);
    return {std::move(w), k};
}

}  // namespace detail

namespace {

enum class Section { None, Definitions, Values, Other };

/// Look up a `[DEFINITIONS]` key and parse the whole value as a double.
double def_f64(const std::map<std::string, std::string>& defs, const char* key) {
    const auto it = defs.find(key);
    double out = 0.0;
    if (it == defs.end() || !parse_f64(it->second, out)) {
        throw DsvError(std::string("missing or invalid ") + key);
    }
    return out;
}

/// Rust's `{:?}` on a string, for the bad-value-line message. See parser.cpp's copy.
std::string debug_quote(std::string_view s) {
    std::string out = "\"";
    for (const char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += c;
        }
    }
    out += '"';
    return out;
}

}  // namespace

std::vector<double> DsvChannel::dense() const {
    // One forward sweep over the corners, in fixed-size chunks of query times, rather than a
    // `value_at` binary search per sample. The sample times are strictly increasing, so the sweep
    // is the right shape; on `cones` GRX that is 63.5 M samples over 2.6 M corners, where the
    // searches cost ~21 comparisons each. Chunking keeps the times buffer at 64 KiB instead of
    // materialising 508 MB of them.
    constexpr std::size_t kChunk = 8192;
    std::vector<double> out(num_samples, 0.0);
    std::vector<Ps> times(kChunk);
    for (std::size_t base = 0; base < num_samples; base += kChunk) {
        const std::size_t len = std::min(kChunk, num_samples - base);
        for (std::size_t i = 0; i < len; ++i) {
            times[i] = Ps::from_secs((static_cast<double>(base + i) + 0.5) * raster_time);
        }
        waveform.sample_at_sorted_into(times.data(), len, out.data() + base);
    }
    return out;
}

DsvChannel read_dsv_text(std::string_view text) {
    std::map<std::string, std::string> definitions;
    std::vector<std::int64_t> rle;
    Section section = Section::None;

    // One int64 per value line at most, so the newline count is a tight upper bound. Without it
    // the delta stream regrows a dozen times on a 23 MB channel; with it, one allocation.
    rle.reserve(static_cast<std::size_t>(std::count(text.begin(), text.end(), '\n')));

    // Split on '\n' in place rather than with std::getline, which allocates and copies a
    // std::string per line -- a DSV channel is one integer per line and millions of lines long, so
    // that was most of the read. Same segmentation as parse_text.
    std::size_t pos = 0;
    bool done = false;  // the `;` terminator in [VALUES] ends the read, as Rust's `break` does
    while (!done && pos < text.size()) {
        const std::size_t nl = text.find('\n', pos);
        const std::size_t stop = nl == std::string_view::npos ? text.size() : nl;
        const std::string_view t = trim(text.substr(pos, stop - pos));  // also drops a CRLF '\r'
        pos = nl == std::string_view::npos ? text.size() : nl + 1;
        if (t.empty()) {
            continue;
        }
        if (t.front() == '[' && t.back() == ']') {
            if (t == "[DEFINITIONS]") {
                section = Section::Definitions;
            } else if (t == "[VALUES]") {
                section = Section::Values;
            } else {
                section = Section::Other;
            }
            continue;
        }
        switch (section) {
            case Section::Definitions: {
                // `KEY=value`, both trimmed. A line without '=' is skipped, not an error.
                const std::size_t eq = t.find('=');
                if (eq != std::string_view::npos) {
                    definitions[std::string(trim(t.substr(0, eq)))] =
                        std::string(trim(t.substr(eq + 1)));
                }
                break;
            }
            case Section::Values: {
                if (t.front() == ';') {
                    done = true;  // the terminator comment: stop reading entirely
                    break;
                }
                std::int64_t d = 0;
                if (!parse_i64(t, d)) {
                    throw DsvError("non-integer value line: " + debug_quote(t));
                }
                rle.push_back(d);
                break;
            }
            case Section::None:
            case Section::Other:
                break;
        }
    }

    const double vertfactor = def_f64(definitions, "VERTFACTOR");
    const double horidelta_us = def_f64(definitions, "HORIDELTA");
    const auto unit_it = definitions.find("VERTUNITNAME");

    DsvChannel ch;
    ch.raster_time = horidelta_us * 1e-6;
    ch.unit = unit_it == definitions.end() ? std::string() : unit_it->second;
    auto decoded = detail::rle_to_piecewise(rle, ch.raster_time, vertfactor);
    ch.waveform = std::move(decoded.first);
    ch.num_samples = decoded.second;
    ch.definitions = std::move(definitions);
    return ch;
}

DsvChannel read_dsv_channel(std::istream& in) {
    // Read the whole stream in 64 KiB blocks, then decode in place -- the same trade the `.seq`
    // and INF readers make: holding one channel costs far less than a per-line allocation.
    std::string text;
    char buf[1 << 16];
    while (in.read(buf, sizeof buf) || in.gcount() > 0) {
        text.append(buf, static_cast<std::size_t>(in.gcount()));
    }
    return read_dsv_text(text);
}

DsvChannel read_dsv_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("I/O error: cannot open " + path);
    }
    return read_dsv_channel(in);
}

}  // namespace s2w
