/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Reads a bracket step "[<non-negative integer>]" from the front of `tail`, where `tail` begins at
// the opening '['. `pos` is the position of that '[' within `path`, used only for error messages.
// The returned token keeps its brackets (e.g. "[42]"): the GPU-side path walker tells array-index
// steps apart from object-key steps by their leading '[' and re-parses the index from the token.
[[nodiscard]] std::string read_bracket_step(std::string_view path,
                                            std::string_view tail,
                                            std::size_t pos)
{
  // tail[0] is '['; the index digits start at tail[1].
  if (tail.size() < 2) { throw_parse_error(path, pos, "unterminated '[' in variant path"); }
  switch (tail[1]) {
    case '*': throw_parse_error(path, pos + 1, "variant path wildcard '[*]' is not supported");
    case '-': throw_parse_error(path, pos + 1, "negative variant path index is not supported");
    case '\'':
    case '"': throw_parse_error(path, pos + 1, "quoted names in '[...]' are not supported");
    default: break;
  }

  std::size_t n = 1;
  while (n < tail.size() && tail[n] >= '0' && tail[n] <= '9') {
    ++n;
  }
  if (n == 1) { throw_parse_error(path, pos + 1, "expected non-negative integer after '['"); }

  // Reject indices that cannot be a valid array position (don't fit in cudf::size_type), so the
  // GPU-side path walker never has to handle an out-of-range value.
  cudf::size_type index = 0;
  if (std::from_chars(tail.data() + 1, tail.data() + n, index).ec != std::errc{}) {
    throw_parse_error(path, pos + 1, "variant path index is out of range");
  }

  if (n >= tail.size() || tail[n] != ']') {
    throw_parse_error(path, pos + n, "expected ']' after index");
  }
  return std::string{tail.substr(0, n + 1)};  // include the closing ']'
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
    if (c == '[') {
      steps.emplace_back(read_bracket_step(path, path.substr(pos), pos));
    } else {
      if (c == '.') {
        ++pos;
        if (pos >= len || !is_name_char(path[pos])) {
          throw_parse_error(path, pos - 1, "trailing '.' with no field name");
        }
      } else if (!(first && is_name_char(c))) {
        // Neither a '.'/'[' step nor a valid leading name (e.g. a stray ']' or a name after a step)
        throw_parse_error(path, pos, "unexpected character in variant path");
      }
      steps.emplace_back(read_unquoted_name(path.substr(pos)));
    }
    pos += steps.back().size();
    first = false;
  }

  CUDF_EXPECTS(!steps.empty(), "variant path is empty", std::invalid_argument);

  return steps;
}

}  // namespace cudf::io::parquet::experimental::detail
