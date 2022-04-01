/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/partitioning.hpp>

#include <algorithm>

class Hashing : public cudf::benchmark {
};

template <class T>
void BM_hash_partition(benchmark::State& state)
{
  auto const num_rows       = state.range(0);
  auto const num_cols       = state.range(1);
  auto const num_partitions = state.range(2);

  // Create owning columns
  auto input_table = create_sequence_table(cycle_dtypes({cudf::type_to_id<T>()}, num_cols),
                                           row_count{static_cast<cudf::size_type>(num_rows)});
  auto input       = cudf::table_view(*input_table);

  auto columns_to_hash = std::vector<cudf::size_type>(num_cols);
  std::iota(columns_to_hash.begin(), columns_to_hash.end(), 0);

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    auto output = cudf::hash_partition(input, columns_to_hash, num_partitions);
  }
}

BENCHMARK_DEFINE_F(Hashing, hash_partition)
(::benchmark::State& state) { BM_hash_partition<double>(state); }

static void CustomRanges(benchmark::internal::Benchmark* b)
{
  for (int columns = 1; columns <= 256; columns *= 16) {
    for (int partitions = 64; partitions <= 1024; partitions *= 2) {
      for (int rows = 1 << 17; rows <= 1 << 21; rows *= 2) {
        b->Args({rows, columns, partitions});
      }
    }
  }
}

BENCHMARK_REGISTER_F(Hashing, hash_partition)
  ->Apply(CustomRanges)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();
