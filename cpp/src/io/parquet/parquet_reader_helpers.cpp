/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_reader_helpers.hpp"

#include <cudf/logger.hpp>
#include <cuda/numeric>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <format>
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

std::size_t column_path_hash::operator()(std::string_view path) const
{
  return std::hash<std::string>{}(normalize_column_path(path, case_sensitive_names));
}

bool column_path_equal::operator()(std::string_view lhs, std::string_view rhs) const
{
  return are_column_paths_equal(lhs, rhs, case_sensitive_names);
}

column_path_set make_column_path_set(bool case_sensitive_names, std::size_t bucket_hint)
{
  return column_path_set(
    bucket_hint, column_path_hash{case_sensitive_names}, column_path_equal{case_sensitive_names});
}

std::size_t derive_pass_read_limit(std::size_t chunk_read_limit)
{
  if (chunk_read_limit == 0) { return 0; }

  // Derive a heuristic pass limit (1.5x the chunk_read_limit) to reduce surprising OOMs
  auto const sum             = cuda::add_overflow(chunk_read_limit, chunk_read_limit / 2);
  auto const pass_read_limit = sum.overflow ? 0 : sum.value;

  CUDF_LOG_WARN(std::format(
    "Chunked Parquet reader: a chunk_read_limit ({} bytes) was provided without a "
    "pass_read_limit; defaulting pass_read_limit to {} bytes to bound input and decompression "
    "memory and reduce the risk of out-of-memory errors on large files. Use a constructor overload "
    "that accepts pass_read_limit to control this explicitly.",
    chunk_read_limit,
    pass_read_limit));

  return pass_read_limit;
}

}  // namespace cudf::io::parquet::detail
