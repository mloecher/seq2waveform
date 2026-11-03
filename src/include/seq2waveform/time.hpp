// Integer picosecond times keep time differences exact across a sequence.
// Values remain double; int64 picoseconds cover about 106 days.

#pragma once

#include <cmath>
#include <cstdint>

namespace s2w {

/// Picosecond time, distinct from seconds and sample counts.
class Ps {
public:
    /// Picoseconds per second (1e12).
    static constexpr double kPerSec = 1e12;

    constexpr Ps() = default;
    explicit constexpr Ps(std::int64_t picoseconds) : raw_(picoseconds) {}

    /// The zero instant.
    static constexpr Ps zero() { return Ps(0); }

    /// Round seconds to the nearest picosecond, with ties away from zero.
    static Ps from_secs(double seconds) {
        return Ps(static_cast<std::int64_t>(std::llround(seconds * kPerSec)));
    }

    /// Convert to seconds. Subtract times first to retain precision for small intervals.
    constexpr double secs() const { return static_cast<double>(raw_) * 1e-12; }

    /// The interpolation fraction (self - a) / (b - a), from the raw integer differences so the
    /// picosecond unit cancels exactly -- one division, correct regardless of absolute time.
    constexpr double frac_between(Ps a, Ps b) const {
        return static_cast<double>(raw_ - a.raw_) / static_cast<double>(b.raw_ - a.raw_);
    }

    /// Raw picosecond count.
    constexpr std::int64_t raw() const { return raw_; }

    friend constexpr bool operator==(Ps a, Ps b) { return a.raw_ == b.raw_; }
    friend constexpr bool operator!=(Ps a, Ps b) { return a.raw_ != b.raw_; }
    friend constexpr bool operator<(Ps a, Ps b) { return a.raw_ < b.raw_; }
    friend constexpr bool operator<=(Ps a, Ps b) { return a.raw_ <= b.raw_; }
    friend constexpr bool operator>(Ps a, Ps b) { return a.raw_ > b.raw_; }
    friend constexpr bool operator>=(Ps a, Ps b) { return a.raw_ >= b.raw_; }

    friend constexpr Ps operator+(Ps a, Ps b) { return Ps(a.raw_ + b.raw_); }
    friend constexpr Ps operator-(Ps a, Ps b) { return Ps(a.raw_ - b.raw_); }
    friend constexpr Ps operator-(Ps a) { return Ps(-a.raw_); }
    friend constexpr Ps operator*(Ps a, std::int64_t k) { return Ps(a.raw_ * k); }

    Ps& operator+=(Ps rhs) {
        raw_ += rhs.raw_;
        return *this;
    }

private:
    std::int64_t raw_{0};
};

}  // namespace s2w
