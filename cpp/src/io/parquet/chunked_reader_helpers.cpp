/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "chunked_reader_helpers.hpp"

#include <cudf/logger.hpp>

#include <cuda/numeric>

#include <format>

namespace cudf::io::parquet::detail {

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
