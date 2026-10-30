// Library-wide constants. The Rust crate keeps these at its root (lib.rs).

#pragma once

namespace s2w {

/// Default gyromagnetic ratio (Hz/T) for converting between Hz/m and mT/m
/// (`Hz/m = mT/m * gamma * 1e-3`).
///
/// 42_575_575 is the value the **Siemens DSP interpreter** uses, not the textbook 42.577e6. It is
/// the default so that DSV-derived and `.seq`-derived gradients share a consistent scale; override
/// it per nucleus or vendor.
constexpr double kScanGamma = 42575575.0;

}  // namespace s2w
