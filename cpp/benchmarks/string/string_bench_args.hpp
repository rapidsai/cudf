/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cudf/types.hpp>

#include <benchmark/benchmark.h>

#include <limits>

/**
 * @brief Generate row count and row length argument ranges for a string benchmark.
 *
 * Generates a series of row count and row length arguments for string benchmarks.
 * Combinations of row count and row length that would exceed the maximum string character
 * column data length are not generated.
 *
 * @param b           Benchmark to update with row count and row length arguments.
 * @param min_rows    Minimum row count argument to generate.
 * @param max_rows    Maximum row count argument to generate.
 * @param rows_mult   Row count multiplier to generate intermediate row count arguments.
 * @param min_rowlen  Minimum row length argument to generate.
 * @param max_rowlen  Maximum row length argument to generate.
 * @param rowlen_mult Row length multiplier to generate intermediate row length arguments.
 */
inline void generate_string_bench_args(benchmark::internal::Benchmark* b,
                                       int min_rows,
                                       int max_rows,
                                       int rows_mult,
                                       int min_rowlen,
                                       int max_rowlen,
                                       int rowlen_mult)
{
  for (int row_count = min_rows; row_count <= max_rows; row_count *= rows_mult) {
    for (int rowlen = min_rowlen; rowlen <= max_rowlen; rowlen *= rowlen_mult) {
      // avoid generating combinations that exceed the cudf column limit
      size_t total_chars = static_cast<size_t>(row_count) * rowlen;
      if (total_chars < static_cast<size_t>(std::numeric_limits<cudf::size_type>::max())) {
        b->Args({row_count, rowlen});
      }
    }
  }
}
