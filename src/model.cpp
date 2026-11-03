#include "seq2waveform/model.hpp"

#include <cmath>
#include <limits>
#include <string_view>

#include "seq2waveform/scan.hpp"

namespace s2w {

RfUse rf_use_from_char(char c) {
    switch (c) {
        case 'e':
            return RfUse::Excitation;
        case 'r':
            return RfUse::Refocusing;
        case 'i':
            return RfUse::Inversion;
        case 's':
            return RfUse::Saturation;
        case 'p':
            return RfUse::Preparation;
        case 'o':
            return RfUse::Other;
        default:
            return RfUse::Undefined;
    }
}

const char* rf_use_name(RfUse u) {
    switch (u) {
        case RfUse::Excitation:
            return "excitation";
        case RfUse::Refocusing:
            return "refocusing";
        case RfUse::Inversion:
            return "inversion";
        case RfUse::Saturation:
            return "saturation";
        case RfUse::Preparation:
            return "preparation";
        case RfUse::Other:
            return "other";
        case RfUse::Undefined:
            break;
    }
    return "undefined";
}

std::vector<double> Shape::decompressed() const {
    if (data.size() == num_samples) {
        return data;  // uncompressed: the data are the samples
    }

    // Expand the run-length-encoded derivative. `v, v, k` means v repeated k + 2 times; a value
    // not followed by a copy of itself stands alone.
    std::vector<double> deriv;
    deriv.reserve(num_samples);
    const std::size_t n = data.size();
    std::size_t i = 0;
    while (i < n) {
        const double v = data[i];
        if (i + 2 < n && data[i + 1] == v) {
            // Saturate before adding two to keep malformed repeat counts in range.
            std::int64_t k = saturating_i64(std::round(data[i + 2]));
            constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
            k = k > kMax - 2 ? kMax - 2 : k;
            const std::int64_t count = k + 2 < 0 ? 0 : k + 2;
            deriv.insert(deriv.end(), static_cast<std::size_t>(count), v);
            i += 3;
        } else {
            deriv.push_back(v);
            i += 1;
        }
    }

    // Sum in order to preserve floating-point rounding.
    double acc = 0.0;
    for (double& d : deriv) {
        acc += d;
        d = acc;
    }
    return deriv;
}

std::optional<double> Sequence::definition_f64(const std::string& key) const {
    const auto it = definitions.find(key);
    if (it == definitions.end()) {
        return std::nullopt;
    }
    const std::string_view first = split_first_space(trim(it->second)).first;
    double v = 0.0;
    if (!parse_f64(first, v)) {
        return std::nullopt;
    }
    return v;
}

}  // namespace s2w
