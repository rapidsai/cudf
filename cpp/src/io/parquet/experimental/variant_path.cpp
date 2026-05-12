/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "variant_path.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace cudf::io::parquet::experimental::detail {

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
      if (pos >= len) { throw_parse_error(path, pos - 1, "trailing '.' with no field name"); }
      std::string name;
      pos = read_unquoted_name(path, pos, name);
      steps.emplace_back(std::move(name));
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

}  // namespace cudf::io::parquet::experimental::detail
