#include "seq2waveform/parser.hpp"

#include <array>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "seq2waveform/error.hpp"
#include "seq2waveform/scan.hpp"

namespace s2w {
namespace {

enum class Section {
    None,
    Version,
    Definitions,
    Blocks,
    Rf,
    Gradients,
    Trap,
    Adc,
    Extensions,
    TriggerExt,
    Shapes,
    Signature,
    Other,
};

/// A `[BLOCKS]` row before its duration is scaled: the raster count is only meaningful once
/// `BlockDurationRaster` is known, and `[DEFINITIONS]` may come after `[BLOCKS]`.
struct RawBlock {
    std::uint32_t num = 0;
    double duration_raster = 0.0;
    std::array<std::uint32_t, 6> events{};
};

/// The `[SHAPES]` entry being built -- a shape spans several lines.
struct PendingShape {
    std::uint32_t id = 0;
    std::size_t num_samples = 0;
    std::vector<double> data;
};

// --- field helpers ---------------------------------------------------------

/// Rust's `{:?}` on a string: double-quoted with the usual escapes. Enough of it for the ASCII
/// content of a `.seq` file, so a bad-shape message reads identically from either port.
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

/// Field `i` as a double, or a `ParseError`. Rust's `pf`, message wording included -- it reaches
/// Python verbatim, and the parity suite compares it.
double pf(const std::vector<std::string_view>& f, std::size_t i, const char* what,
          std::size_t line) {
    double v = 0.0;
    if (i >= f.size() || !parse_f64(f[i], v)) {
        throw ParseError(line, std::string("missing/invalid ") + what);
    }
    return v;
}

/// Field `i` as a u32, or a `ParseError`. Rust's `pu`.
std::uint32_t pu(const std::vector<std::string_view>& f, std::size_t i, const char* what,
                 std::size_t line) {
    std::uint32_t v = 0;
    if (i >= f.size() || !parse_u32(f[i], v)) {
        throw ParseError(line, std::string("missing/invalid ") + what);
    }
    return v;
}

// --- section row parsers ---------------------------------------------------

void parse_version(std::string_view t, Version& v) {
    const std::vector<std::string_view> f = fields(t);
    if (f.empty()) {
        return;
    }
    std::uint32_t val = 0;
    if (f.size() < 2 || !parse_u32(f[1], val)) {
        val = 0;  // Rust: unwrap_or(0) -- an unreadable value is not an error here
    }
    if (f[0] == "major") {
        v.major = val;
    } else if (f[0] == "minor") {
        v.minor = val;
    } else if (f[0] == "revision") {
        v.revision = val;
    }
}

void parse_definition(std::string_view t, std::map<std::string, std::string>& defs) {
    // Key, then the rest of the line trimmed. A bare key stores an empty value; a repeated key
    // overwrites, as Rust's HashMap::insert does.
    const std::pair<std::string_view, std::string_view> kv = split_first_space(t);
    defs[std::string(kv.first)] = std::string(kv.second);
}

/// `NUM DUR RF GX GY GZ ADC EXT`
RawBlock parse_block(std::string_view t, std::size_t line) {
    const std::vector<std::string_view> f = fields(t);
    RawBlock b;
    b.num = pu(f, 0, "block num", line);
    b.duration_raster = pf(f, 1, "block duration", line);
    for (std::size_t j = 0; j < b.events.size(); ++j) {
        b.events[j] = pu(f, 2 + j, "block event id", line);
    }
    return b;
}

/// `id ampl mag phase time_shape center delay freqPPM phasePPM freq phase use`
std::pair<std::uint32_t, Rf> parse_rf(std::string_view t, std::size_t line) {
    const std::vector<std::string_view> f = fields(t);
    const std::uint32_t id = pu(f, 0, "rf id", line);
    Rf rf;
    rf.amplitude = pf(f, 1, "rf amplitude", line);
    rf.mag_shape = pu(f, 2, "rf mag_shape", line);
    rf.phase_shape = pu(f, 3, "rf phase_shape", line);
    rf.time_shape = pu(f, 4, "rf time_shape", line);
    rf.center = pf(f, 5, "rf center", line) * 1e-6;  // us -> s
    rf.delay = pf(f, 6, "rf delay", line) * 1e-6;
    rf.freq_ppm = pf(f, 7, "rf freqPPM", line);
    rf.phase_ppm = pf(f, 8, "rf phasePPM", line);
    rf.freq = pf(f, 9, "rf freq", line);
    rf.phase = pf(f, 10, "rf phase", line);
    // A missing or empty `use` field becomes 'u' -> Undefined, as in Rust. v1.4 files have no
    // such column, so this is the common case, not an error.
    rf.use = rf_use_from_char(f.size() > 11 && !f[11].empty() ? f[11].front() : 'u');
    return {id, rf};
}

/// `id amplitude first last amp_shape time_shape delay`
std::pair<std::uint32_t, Gradient> parse_grad(std::string_view t, std::size_t line) {
    const std::vector<std::string_view> f = fields(t);
    const std::uint32_t id = pu(f, 0, "grad id", line);
    ArbitraryGrad g;
    g.amplitude = pf(f, 1, "grad amplitude", line);
    g.first = pf(f, 2, "grad first", line);
    g.last = pf(f, 3, "grad last", line);
    g.amp_shape = pu(f, 4, "grad amp_shape", line);
    g.time_shape = pu(f, 5, "grad time_shape", line);
    g.delay = pf(f, 6, "grad delay", line) * 1e-6;
    return {id, Gradient(g)};
}

/// `id amplitude rise flat fall delay`
std::pair<std::uint32_t, Gradient> parse_trap(std::string_view t, std::size_t line) {
    const std::vector<std::string_view> f = fields(t);
    const std::uint32_t id = pu(f, 0, "trap id", line);
    Trap g;
    g.amplitude = pf(f, 1, "trap amplitude", line);
    g.rise = pf(f, 2, "trap rise", line) * 1e-6;
    g.flat = pf(f, 3, "trap flat", line) * 1e-6;
    g.fall = pf(f, 4, "trap fall", line) * 1e-6;
    g.delay = pf(f, 5, "trap delay", line) * 1e-6;
    return {id, Gradient(g)};
}

/// `id num dwell delay freqPPM phasePPM freq phase phase_id`
std::pair<std::uint32_t, Adc> parse_adc(std::string_view t, std::size_t line) {
    const std::vector<std::string_view> f = fields(t);
    const std::uint32_t id = pu(f, 0, "adc id", line);
    Adc a;
    a.num = pu(f, 1, "adc num", line);
    a.dwell = pf(f, 2, "adc dwell", line) * 1e-9;  // ns -> s
    a.delay = pf(f, 3, "adc delay", line) * 1e-6;  // us -> s
    a.freq_ppm = pf(f, 4, "adc freqPPM", line);
    a.phase_ppm = pf(f, 5, "adc phasePPM", line);
    a.freq = pf(f, 6, "adc freq", line);
    a.phase = pf(f, 7, "adc phase", line);
    a.phase_shape = pu(f, 8, "adc phase_id", line);
    return {id, a};
}

/// `id type ref next_id`
std::pair<std::uint32_t, ExtNode> parse_ext_node(std::string_view t, std::size_t line) {
    const std::vector<std::string_view> f = fields(t);
    const std::uint32_t id = pu(f, 0, "ext id", line);
    ExtNode node;
    node.type_id = pu(f, 1, "ext type", line);
    node.ref_id = pu(f, 2, "ext ref", line);
    node.next_id = pu(f, 3, "ext next_id", line);
    return {id, node};
}

/// `id type channel delay duration`
std::pair<std::uint32_t, Trigger> parse_trigger(std::string_view t, std::size_t line) {
    const std::vector<std::string_view> f = fields(t);
    const std::uint32_t id = pu(f, 0, "trigger id", line);
    Trigger trg;
    trg.trig_type = pu(f, 1, "trigger type", line);
    trg.channel = pu(f, 2, "trigger channel", line);
    trg.delay = pf(f, 3, "trigger delay", line) * 1e-6;
    trg.duration = pf(f, 4, "trigger duration", line) * 1e-6;
    return {id, trg};
}

// --- shapes (stateful across lines) ---------------------------------------

void finish_shape(std::optional<PendingShape>& cur, std::map<std::uint32_t, Shape>& shapes) {
    if (!cur) {
        return;
    }
    Shape s;
    s.num_samples = cur->num_samples;
    s.data = std::move(cur->data);
    shapes[cur->id] = std::move(s);
    cur.reset();
}

void parse_shape_line(std::string_view t, std::optional<PendingShape>& cur,
                      std::map<std::uint32_t, Shape>& shapes, std::size_t line) {
    // Note both prefixes are matched without requiring a space, as in Rust: "shape_id 1" and
    // "shape_id1" are both accepted, and the remainder is trimmed before parsing.
    if (starts_with(t, "shape_id")) {
        finish_shape(cur, shapes);  // close the previous shape
        std::uint32_t id = 0;
        if (!parse_u32(trim(t.substr(8)), id)) {
            throw ParseError(line, "bad shape_id: " + debug_quote(t));
        }
        cur = PendingShape{id, 0, {}};
    } else if (starts_with(t, "num_samples")) {
        std::size_t n = 0;
        if (!parse_usize(trim(t.substr(11)), n)) {
            throw ParseError(line, "bad num_samples: " + debug_quote(t));
        }
        if (cur) {
            cur->num_samples = n;  // outside a shape this is silently dropped, as in Rust
        }
    } else {
        double v = 0.0;
        if (!parse_f64(t, v)) {
            throw ParseError(line, "bad shape sample: " + debug_quote(t));
        }
        if (cur) {
            cur->data.push_back(v);
        }
    }
}

// --- dispatch --------------------------------------------------------------

/// Recognise a section header (`[X]`, or `extension <NAME> <id>`) and record extension-type
/// names. False for an ordinary content line. `t` must be non-empty and trimmed.
bool section_header(std::string_view t, Sequence& seq, Section& out) {
    if (t.front() == '[' && t.back() == ']') {
        if (t == "[VERSION]") {
            out = Section::Version;
        } else if (t == "[DEFINITIONS]") {
            out = Section::Definitions;
        } else if (t == "[BLOCKS]") {
            out = Section::Blocks;
        } else if (t == "[RF]") {
            out = Section::Rf;
        } else if (t == "[GRADIENTS]") {
            out = Section::Gradients;
        } else if (t == "[TRAP]") {
            out = Section::Trap;
        } else if (t == "[ADC]") {
            out = Section::Adc;
        } else if (t == "[EXTENSIONS]") {
            out = Section::Extensions;
        } else if (t == "[SHAPES]") {
            out = Section::Shapes;
        } else if (t == "[SIGNATURE]") {
            out = Section::Signature;
        } else {
            out = Section::Other;  // an unknown section: skip its rows
        }
        return true;
    }

    if (starts_with(t, "extension ")) {
        const std::vector<std::string_view> f = fields(t.substr(10));
        const std::string_view name = f.empty() ? std::string_view{} : f[0];
        std::uint32_t type_id = 0;
        if (f.size() < 2 || !parse_u32(f[1], type_id)) {
            type_id = 0;  // Rust: unwrap_or(0)
        }
        seq.ext_type_names[type_id] = std::string(name);
        out = name == "TRIGGERS" ? Section::TriggerExt : Section::Other;
        return true;
    }

    return false;
}

/// Route one content line to the current section's row parser.
void dispatch(Section section, std::string_view t, Sequence& seq, std::vector<RawBlock>& raw_blocks,
              std::optional<PendingShape>& cur_shape, std::size_t line) {
    switch (section) {
        case Section::Version:
            parse_version(t, seq.version);
            break;
        case Section::Definitions:
            parse_definition(t, seq.definitions);
            break;
        case Section::Signature:
            parse_definition(t, seq.signature);
            break;
        case Section::Blocks:
            raw_blocks.push_back(parse_block(t, line));
            break;
        case Section::Rf: {
            const std::pair<std::uint32_t, Rf> e = parse_rf(t, line);
            seq.rf.insert_or_assign(e.first, e.second);
            break;
        }
        case Section::Gradients: {
            const std::pair<std::uint32_t, Gradient> e = parse_grad(t, line);
            seq.gradients.insert_or_assign(e.first, e.second);
            break;
        }
        case Section::Trap: {
            const std::pair<std::uint32_t, Gradient> e = parse_trap(t, line);
            seq.gradients.insert_or_assign(e.first, e.second);
            break;
        }
        case Section::Adc: {
            const std::pair<std::uint32_t, Adc> e = parse_adc(t, line);
            seq.adc.insert_or_assign(e.first, e.second);
            break;
        }
        case Section::Extensions: {
            const std::pair<std::uint32_t, ExtNode> e = parse_ext_node(t, line);
            seq.ext_nodes.insert_or_assign(e.first, e.second);
            break;
        }
        case Section::TriggerExt: {
            const std::pair<std::uint32_t, Trigger> e = parse_trigger(t, line);
            seq.triggers.insert_or_assign(e.first, e.second);
            break;
        }
        case Section::Shapes:
            parse_shape_line(t, cur_shape, seq.shapes, line);
            break;
        case Section::None:
        case Section::Other:
            break;
    }
}

}  // namespace

Sequence parse_text(std::string_view text) {
    Sequence seq;
    Section section = Section::None;
    std::vector<RawBlock> raw_blocks;
    std::optional<PendingShape> cur_shape;

    // 1-based, and it counts blank and comment lines too -- Rust enumerates before skipping them,
    // so this is what makes an error point at the same line from either port.
    std::size_t line = 0;
    // Split on '\n' in place rather than with std::getline, which allocates and copies a
    // std::string per line -- on a large sequence that is over a million allocations, and it
    // dominated the parse. The segmentation matches both getline and Rust's `lines()`: a trailing
    // newline does not yield a final empty line, and a final line without one still counts.
    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::size_t nl = text.find('\n', pos);
        const std::size_t stop = nl == std::string_view::npos ? text.size() : nl;
        const std::string_view raw = text.substr(pos, stop - pos);
        pos = nl == std::string_view::npos ? text.size() : nl + 1;

        ++line;
        const std::string_view t = trim(raw);  // also drops a CRLF '\r'
        if (t.empty() || t.front() == '#') {
            continue;
        }

        Section next = Section::None;
        if (section_header(t, seq, next)) {
            // A repeated [SHAPES] header does not close the shape in progress, as in Rust.
            if (next != Section::Shapes) {
                finish_shape(cur_shape, seq.shapes);
            }
            section = next;
            continue;
        }

        dispatch(section, t, seq, raw_blocks, cur_shape, line);
    }
    finish_shape(cur_shape, seq.shapes);  // the file may end mid-shape

    // Block durations are stored in BlockDurationRaster units, resolved only now because
    // [DEFINITIONS] may appear after [BLOCKS]. Default 10 us.
    const double bdr = seq.definition_f64("BlockDurationRaster").value_or(10e-6);
    seq.blocks.reserve(raw_blocks.size());
    for (const RawBlock& r : raw_blocks) {
        Block b;
        b.num = r.num;
        b.duration = r.duration_raster * bdr;
        b.rf = r.events[0];
        b.gx = r.events[1];
        b.gy = r.events[2];
        b.gz = r.events[3];
        b.adc = r.events[4];
        b.ext = r.events[5];
        seq.blocks.push_back(b);
    }

    return seq;
}

Sequence parse(std::istream& in) {
    // Read the whole stream in 64 KiB blocks, then parse in place. Sequence files are a few tens
    // of MB at most, and holding one costs far less than a per-line allocation.
    std::string text;
    char buf[1 << 16];
    while (in.read(buf, sizeof buf) || in.gcount() > 0) {
        text.append(buf, static_cast<std::size_t>(in.gcount()));
    }
    return parse_text(text);
}

Sequence parse_file(const std::string& path) {
    // Binary mode: trim removes a CRLF '\r' anyway, and this keeps the byte stream identical on
    // every platform.
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("I/O error: cannot open " + path);
    }
    return parse(in);
}

}  // namespace s2w
