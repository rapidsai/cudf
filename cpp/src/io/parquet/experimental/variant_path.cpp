/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "variant_path.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <charconv>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace cudf::io::parquet::experimental::detail {

namespace {

// Dot-notation field names accept any byte except the structural characters '.' and '['.
[[nodiscard]] constexpr bool is_name_char(char c) { return c != '.' && c != '['; }

[[noreturn]] void throw_parse_error(std::string_view path, std::size_t pos, std::string_view msg)
{
  CUDF_FAIL(std::format("invalid variant path \"{}\" at position {}: {}", path, pos, msg),
            std::invalid_argument);
}

// Reads a maximal run of name characters from the front of `tail`.
[[nodiscard]] std::string read_unquoted_name(std::string_view tail)
{
  std::size_t n = 0;
  while (n < tail.size() && is_name_char(tail[n])) {
    ++n;
  }
  return std::string{tail.substr(0, n)};
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
      if (pos >= len || !is_name_char(path[pos])) {
        throw_parse_error(path, pos - 1, "trailing '.' with no field name");
      }
      steps.emplace_back(read_unquoted_name(path.substr(pos)));
      pos += steps.back().size();
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
      auto const digits_start = pos;
      while (pos < len && path[pos] >= '0' && path[pos] <= '9') {
        ++pos;
      }
      // Reject indices that cannot be a valid array position (don't fit in cudf::size_type), so the
      // GPU-side path walker never has to parse an out-of-range value.
      cudf::size_type index_value = 0;
      [[maybe_unused]] auto const [ptr, ec] =
        std::from_chars(path.data() + digits_start, path.data() + pos, index_value);
      if (ec != std::errc{}) {
        throw_parse_error(path, digits_start, "variant path index is out of range");
      }
      if (pos >= len || path[pos] != ']') {
        throw_parse_error(path, pos, "expected ']' after index");
      }
      ++pos;  // consume ']'
      steps.emplace_back(path.data() + tok_start, pos - tok_start);
    } else if (first && is_name_char(c)) {
      // Allow a bare leading name (e.g. "x" or "foo" with no leading '$').
      steps.emplace_back(read_unquoted_name(path.substr(pos)));
      pos += steps.back().size();
    } else {
      // Neither a '.' step nor a valid leading name (e.g. a bracket step like "[0]" or "foo[1]")
      throw_parse_error(path, pos, "unexpected character in variant path");
    }

    first = false;
  }

  CUDF_EXPECTS(!steps.empty(), "variant path is empty", std::invalid_argument);

  return steps;
}

}  // namespace cudf::io::parquet::experimental::detail
