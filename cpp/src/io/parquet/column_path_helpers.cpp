/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "column_path_helpers.hpp"

#include <algorithm>
#include <cctype>
#include <functional>
#include <string>
#include <string_view>

namespace cudf::io::parquet::detail {

std::string normalize_column_path(std::string_view col_path, bool case_sensitive_names)
{
  if (case_sensitive_names) { return std::string{col_path}; }
  auto normalized_path = std::string(col_path.size(), '\0');
  std::transform(col_path.begin(), col_path.end(), normalized_path.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  return normalized_path;
}

bool are_column_paths_equal(std::string_view lhs, std::string_view rhs, bool case_sensitive)
{
  if (lhs.size() != rhs.size()) { return false; }
  if (case_sensitive) { return lhs == rhs; }
  // Optimize by normalizing and comparing char-by-char instead of whole strings
  return std::equal(
    lhs.begin(), lhs.end(), rhs.begin(), [](unsigned char lhs_char, unsigned char rhs_char) {
      return std::equal_to<>{}(std::tolower(lhs_char), std::tolower(rhs_char));
    });
}

}  // namespace cudf::io::parquet::detail
