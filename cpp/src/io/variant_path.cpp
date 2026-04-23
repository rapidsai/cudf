/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "variant_path.hpp"

#include <cctype>
#include <charconv>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace cudf::io::parquet::detail {

namespace {

constexpr bool is_name_start(char c)
{
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

constexpr bool is_name_cont(char c) { return is_name_start(c) || (c >= '0' && c <= '9'); }

[[noreturn]] void throw_parse_error(std::string_view path, std::size_t pos, std::string_view msg)
{
  throw std::invalid_argument("invalid variant path \"" + std::string{path} + "\" at position " +
                              std::to_string(pos) + ": " + std::string{msg});
}

// Read an unquoted identifier starting at `pos`. Returns the new position after the name.
// Throws if no valid name is present at `pos`.
std::size_t read_unquoted_name(std::string_view path, std::size_t pos, std::string& out)
{
  auto const start = pos;
  auto const len   = path.size();
  if (pos >= len || !is_name_start(path[pos])) {
    throw_parse_error(path, pos, "expected field name (letter or '_')");
  }
  ++pos;
  while (pos < len && is_name_cont(path[pos])) {
    ++pos;
  }
  out.assign(path.data() + start, pos - start);
  return pos;
}

// Read a bracketed step starting AFTER the '['. Returns the new position after the closing ']'.
std::size_t read_bracket_step(std::string_view path,
                              std::size_t pos,
                              std::vector<variant_path_step>& out)
{
  auto const len = path.size();
  if (pos >= len) { throw_parse_error(path, pos, "unterminated '[' in variant path"); }

  char const nc = path[pos];
  if (nc == '*') { throw_parse_error(path, pos, "variant path wildcard '[*]' is not supported"); }
  if (nc == '-') { throw_parse_error(path, pos, "negative variant path index is not supported"); }
  if (nc == ']') { throw_parse_error(path, pos, "empty '[]' in variant path"); }

  if (nc == '\'' || nc == '"') {
    char const quote = nc;
    ++pos;
    auto const start = pos;
    while (pos < len && path[pos] != quote) {
      ++pos;
    }
    if (pos >= len) { throw_parse_error(path, pos, "unterminated quoted name"); }
    if (pos == start) { throw_parse_error(path, pos, "empty quoted name"); }
    std::string name{path.data() + start, pos - start};
    ++pos;  // consume closing quote
    if (pos >= len || path[pos] != ']') {
      throw_parse_error(path, pos, "expected ']' after quoted name");
    }
    ++pos;
    out.emplace_back(std::move(name));
    return pos;
  }

  if (nc >= '0' && nc <= '9') {
    auto const start = pos;
    while (pos < len && path[pos] >= '0' && path[pos] <= '9') {
      ++pos;
    }
    long long val = 0;
    auto [p, ec]  = std::from_chars(path.data() + start, path.data() + pos, val);
    if (ec != std::errc{} ||
        val > static_cast<long long>(std::numeric_limits<cudf::size_type>::max())) {
      throw_parse_error(path, start, "index out of range");
    }
    if (pos >= len || path[pos] != ']') {
      throw_parse_error(path, pos, "expected ']' after index");
    }
    ++pos;
    out.emplace_back(static_cast<cudf::size_type>(val));
    return pos;
  }

  throw_parse_error(path, pos, "unexpected character after '['");
}

}  // namespace

std::vector<variant_path_step> parse_variant_path(std::string_view path)
{
  std::vector<variant_path_step> steps;
  auto const len  = path.size();
  std::size_t pos = 0;

  // Optional leading '$'
  if (pos < len && path[pos] == '$') { ++pos; }

  bool first = true;
  while (pos < len) {
    char const c = path[pos];
    if (c == '.') {
      ++pos;
      std::string name;
      pos = read_unquoted_name(path, pos, name);
      steps.emplace_back(std::move(name));
    } else if (c == '[') {
      ++pos;
      pos = read_bracket_step(path, pos, steps);
    } else if (first && is_name_start(c)) {
      // Allow a bare leading name (e.g. "x" or "foo" with no leading '$').
      std::string name;
      pos = read_unquoted_name(path, pos, name);
      steps.emplace_back(std::move(name));
    } else {
      throw_parse_error(path, pos, "unexpected character in variant path");
    }
    first = false;
  }

  if (steps.empty()) { throw std::invalid_argument("variant path must contain at least one step"); }

  return steps;
}

}  // namespace cudf::io::parquet::detail
