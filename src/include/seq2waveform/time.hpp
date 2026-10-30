// Integer time in picoseconds.
//
// Absolute times reach tens of seconds, where a double resolves only a few femtoseconds -- so
// interpolating (t-t0)/(t1-t0) with large operands loses precision that depends on *where* in the
// sequence you are, and the same block played early vs late samples differently. int64 picoseconds
// make every difference exact at any magnitude (t-t0 is an exact integer, cast to double only once
// small). MRI rasters are exact in ps (10 us = 1e7, half-dwell 0.5 ns = 500) and int64 spans
// ~106 days. Only *time* is integer; values (Hz/m, 1/m, Hz, rad) stay double.

#pragma once

#include <cmath>
#include <cstdint>

namespace s2w {

/// A time, in picoseconds. A distinct type so it cannot be mixed up with a seconds-double, a
/// value, or a raw sample count -- the guarantee that keeps bit-exact delay sweeps bit-exact.
class Ps {
public:
    /// Picoseconds per second (1e12).
    static constexpr double kPerSec = 1e12;

    constexpr Ps() = default;
    explicit constexpr Ps(std::int64_t picoseconds) : raw_(picoseconds) {}

    /// The zero instant.
    static constexpr Ps zero() { return Ps(0); }

    /// Nearest picosecond to `seconds`. Exact for raster-aligned times. Rounds half away from
    /// zero, matching Rust's `(s * 1e12).round() as i64`, so the two ports agree bit-for-bit.
    static Ps from_secs(double seconds) {
        return Ps(static_cast<std::int64_t>(std::llround(seconds * kPerSec)));
    }

    /// This instant/duration in seconds. For a *difference* the operands cancel first (exact
    /// int64), so (t1 - t0).secs() is precise even when t0, t1 are large. Multiplies by 1e-12
    /// rather than dividing by 1e12 -- same result to ~1 ulp, far cheaper in hot loops.
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
