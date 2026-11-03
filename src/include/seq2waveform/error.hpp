// Parse errors with source details for callers and Python exceptions.

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace s2w {

/// A malformed sequence line. what(): "parse error at line {line}: {message}".
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

/// A malformed DSV/INF value. what(): "DSV format error: {message}".
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
