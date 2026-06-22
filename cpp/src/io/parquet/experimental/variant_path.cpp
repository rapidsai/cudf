/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "variant_path.hpp"

#include <cudf/utilities/error.hpp>

#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
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
    } else if (!(first && is_name_char(c))) {
      // Neither a '.' step nor a valid leading name (e.g. a bracket step like "[0]" or "foo[1]")
      throw_parse_error(path, pos, "unexpected character in variant path");
    }

    steps.emplace_back(read_unquoted_name(path.substr(pos)));
    pos += steps.back().size();
    first = false;
  }

  CUDF_EXPECTS(!steps.empty(), "variant path is empty", std::invalid_argument);

  return steps;
}

}  // namespace cudf::io::parquet::experimental::detail
