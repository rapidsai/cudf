/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "variant_path.hpp"

#include <cudf/utilities/error.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace cudf::io::parquet::experimental::detail {

namespace {

// Dot-notation field names accept any byte except the structural characters '.' and '['.
[[nodiscard]] constexpr bool is_name_char(char c) { return c != '.' && c != '['; }

[[noreturn]] void throw_parse_error(std::string_view path, std::size_t pos, std::string_view msg)
{
  CUDF_FAIL("invalid variant path \"" + std::string{path} + "\" at position " +
              std::to_string(pos) + ": " + std::string{msg},
            std::invalid_argument);
}

std::size_t read_unquoted_name(std::string_view path, std::size_t pos, std::string& out)
{
  auto const start = pos;
  auto const len   = path.size();
  if (pos >= len || !is_name_char(path[pos])) {
    throw_parse_error(path, pos, "field name cannot be empty");
  }
  ++pos;
  while (pos < len && is_name_char(path[pos])) {
    ++pos;
  }
  out.assign(path.data() + start, pos - start);
  return pos;
}

}  // namespace

std::vector<std::string> parse_variant_path(std::string_view path)
{
  std::vector<std::string> steps;
  auto const len  = path.size();
  std::size_t pos = 0;

  // Optional leading '$'
  if (pos < len && path[pos] == '$') { ++pos; }

  bool first = true;
  while (pos < len) {
    char const c = path[pos];
    if (c == '.') {
      ++pos;
      if (pos >= len) { throw_parse_error(path, pos - 1, "trailing '.' with no field name"); }
      std::string name;
      pos = read_unquoted_name(path, pos, name);
      steps.emplace_back(std::move(name));
    } else if (c == '[') {
      // Array-index step: "[<non-negative integer>]".
      // The literal token (e.g. "[42]") is preserved so the GPU-side path walker can
      // distinguish array-index steps from object-key steps by inspecting the first byte.
      auto const tok_start = pos;
      ++pos;  // consume '['
      if (pos >= len) { throw_parse_error(path, tok_start, "unterminated '[' in variant path"); }
      char const nc = path[pos];
      if (nc == '*') {
        throw_parse_error(path, pos, "variant path wildcard '[*]' is not supported");
      }
      if (nc == '-') {
        throw_parse_error(path, pos, "negative variant path index is not supported");
      }
      if (nc == '\'' || nc == '"') {
        throw_parse_error(path, pos, "quoted names in '[...]' are not supported");
      }
      if (!(nc >= '0' && nc <= '9')) {
        throw_parse_error(path, pos, "expected non-negative integer after '['");
      }
      while (pos < len && path[pos] >= '0' && path[pos] <= '9') {
        ++pos;
      }
      if (pos >= len || path[pos] != ']') {
        throw_parse_error(path, pos, "expected ']' after index");
      }
      ++pos;  // consume ']'
      steps.emplace_back(path.data() + tok_start, pos - tok_start);
    } else if (first && is_name_char(c)) {
      // Allow a bare leading name (e.g. "x" or "foo" with no leading '$').
      std::string name;
      pos = read_unquoted_name(path, pos, name);
      steps.emplace_back(std::move(name));
    } else {
      throw_parse_error(path, pos, "unexpected character in variant path");
    }
    first = false;
  }

  CUDF_EXPECTS(!steps.empty(), "variant path is empty", std::invalid_argument);

  return steps;
}

}  // namespace cudf::io::parquet::experimental::detail
