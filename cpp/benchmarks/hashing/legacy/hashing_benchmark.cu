/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/cudf.h>

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>
#include <tests/utilities/legacy/column_wrapper.cuh>

#include <algorithm>
#include <numeric>

using cudf::test::column_wrapper;

class Hashing : public cudf::benchmark {};

template <class T>
void BM_hash_partition(benchmark::State& state) {
  auto const num_rows       = state.range(0);
  auto const num_cols       = state.range(1);
  auto const num_partitions = state.range(2);

  // Create input columns
  std::vector<column_wrapper<T>> input_columns(
    num_cols, {static_cast<cudf::size_type>(num_rows), [](cudf::size_type row) {
                 return static_cast<T>(row);
               }});
  std::vector<gdf_column*> raw_input(num_cols);
  std::transform(input_columns.begin(), input_columns.end(), raw_input.begin(), [](auto& col) {
    return col.get();
  });

  auto columns_to_hash = std::vector<cudf::size_type>(num_cols);
  std::iota(columns_to_hash.begin(), columns_to_hash.end(), 0);

  for (auto _ : state) {
    cuda_event_timer timer(state, true);

    // Allocate output memory inside loop for fair benchmark
    std::vector<column_wrapper<T>> output_columns(
      num_cols, {static_cast<cudf::size_type>(num_rows), [](cudf::size_type row) {
                   return static_cast<T>(row);
                 }});

    std::vector<gdf_column*> raw_output(num_cols);
    std::transform(output_columns.begin(), output_columns.end(), raw_output.begin(), [](auto& col) {
      return col.get();
    });

    std::vector<int> output_offsets(num_partitions);

    gdf_hash_partition(num_cols,
                       raw_input.data(),
                       columns_to_hash.data(),
                       columns_to_hash.size(),
                       num_partitions,
                       raw_output.data(),
                       output_offsets.data(),
                       GDF_HASH_MURMUR3);
  }
}

BENCHMARK_DEFINE_F(Hashing, gdf_hash_partition)
(::benchmark::State& state) { BM_hash_partition<double>(state); }

static void CustomRanges(benchmark::internal::Benchmark* b) {
  for (int columns = 1; columns <= 256; columns *= 16) {
    for (int partitions = 64; partitions <= 1024; partitions *= 2) {
      for (int rows = 1 << 17; rows <= 1 << 21; rows *= 2) { b->Args({rows, columns, partitions}); }
    }
  }
}

BENCHMARK_REGISTER_F(Hashing, gdf_hash_partition)
  ->Apply(CustomRanges)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();