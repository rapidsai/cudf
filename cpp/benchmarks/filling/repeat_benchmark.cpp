/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <benchmark/benchmark.h>

#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <random>

#include "../fixture/benchmark_fixture.hpp"
#include "../synchronization/synchronization.hpp"

class Repeat : public cudf::benchmark {
};

template <class TypeParam, bool nulls>
void BM_repeat(benchmark::State& state)
{
  using column_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;
  auto const n_rows    = static_cast<cudf::size_type>(state.range(0));
  auto const n_cols    = static_cast<cudf::size_type>(state.range(1));

  auto idx_begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto idx_end   = thrust::make_counting_iterator<cudf::size_type>(n_rows);

  std::vector<column_wrapper> columns;
  columns.reserve(n_rows);
  std::generate_n(std::back_inserter(columns), n_cols, [&]() {
    return nulls ? column_wrapper(
                     idx_begin,
                     idx_end,
                     thrust::make_transform_iterator(idx_begin, [](auto idx) { return true; }))
                 : column_wrapper(idx_begin, idx_end);
  });

  // repeat counts
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 3);

  std::vector<cudf::size_type> host_repeat_count(n_rows);
  std::generate(
    host_repeat_count.begin(), host_repeat_count.end(), [&] { return distribution(generator); });

  cudf::test::fixed_width_column_wrapper<cudf::size_type> repeat_count(host_repeat_count.begin(),
                                                                       host_repeat_count.end());

  // Create column views
  auto const column_views = std::vector<cudf::column_view>(columns.begin(), columns.end());

  // Create table view
  auto input = cudf::table_view(column_views);

  // warm up
  auto output = cudf::repeat(input, repeat_count);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf::repeat(input, repeat_count);
  }

  auto data_bytes =
    (input.num_columns() * input.num_rows() + output->num_columns() * output->num_rows()) *
    sizeof(TypeParam);
  auto null_bytes =
    nulls ? input.num_columns() * cudf::bitmask_allocation_size_bytes(input.num_rows()) +
              output->num_columns() * cudf::bitmask_allocation_size_bytes(output->num_rows())
          : 0;
  state.SetBytesProcessed(state.iterations() * (data_bytes + null_bytes));
}

#define REPEAT_BENCHMARK_DEFINE(name, type, nulls)                                                \
  BENCHMARK_DEFINE_F(Repeat, name)(::benchmark::State & state) { BM_repeat<type, nulls>(state); } \
  BENCHMARK_REGISTER_F(Repeat, name)                                                              \
    ->RangeMultiplier(8)                                                                          \
    ->Ranges({{1 << 10, 1 << 26}, {1, 8}})                                                        \
    ->UseManualTime()                                                                             \
    ->Unit(benchmark::kMillisecond);

REPEAT_BENCHMARK_DEFINE(double_nulls, double, true);
REPEAT_BENCHMARK_DEFINE(double_no_nulls, double, false);
