// Error types for the core library.
//
// Ported from the Rust `error.rs`, message wording included: the parity suite compares the
// exception text from both ports, which is a cheap end-to-end check that the parse loop stamped
// the same source line on the same failure.

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace s2w {

/// A malformed line in a text sequence file, carrying the 1-based source line number.
///
/// `what()` is formatted like Rust's `Display` for `Error::Parse`:
/// `"parse error at line {line}: {message}"`.
class ParseError : public std::runtime_error {
public:
    ParseError(std::size_t line, const std::string& message)
        : std::runtime_error("parse error at line " + std::to_string(line) + ": " + message),
          line_(line),
          message_(message) {}

    /// 1-based line number in the source file.
    std::size_t line() const { return line_; }

    /// The message on its own, without the line prefix.
    const std::string& message() const { return message_; }

private:
    std::size_t line_;
    std::string message_;
};

/// A malformed Siemens `.dsv` / `.inf` file. `what()` matches Rust's `Error::Dsv` Display:
/// `"DSV format error: {message}"`. These files carry no line numbers worth reporting -- a bad
/// value line is quoted in the message instead.
class DsvError : public std::runtime_error {
public:
    explicit DsvError(const std::string& message)
        : std::runtime_error("DSV format error: " + message), message_(message) {}

    /// The message on its own, without the prefix.
    const std::string& message() const { return message_; }

private:
    std::string message_;
};

}  // namespace s2w
