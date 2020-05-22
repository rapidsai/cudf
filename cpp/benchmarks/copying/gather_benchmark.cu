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

#include <benchmark/benchmark.h>

#include <cudf/copying.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/table_utilities.hpp>

#include <cudf/types.hpp>

#include <algorithm>
#include <random>

#include "../fixture/benchmark_fixture.hpp"
#include "../synchronization/synchronization.hpp"

class Gather : public cudf::benchmark {
};

template <class TypeParam, bool coalesce>
void BM_gather(benchmark::State& state)
{
  const cudf::size_type source_size{(cudf::size_type)state.range(0)};
  const cudf::size_type n_cols = (cudf::size_type)state.range(1);

  // Every element is valid
  auto data = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i; });

  // Gather indices
  std::vector<cudf::size_type> host_map_data(source_size);
  std::iota(host_map_data.begin(), host_map_data.end(), 0);

  if (coalesce) {
    std::reverse(host_map_data.begin(), host_map_data.end());
  } else {
    std::random_shuffle(host_map_data.begin(), host_map_data.end());
  }

  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map(host_map_data.begin(),
                                                                     host_map_data.end());

  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> source_column_wrappers;
  std::vector<cudf::column_view> source_columns(n_cols);

  std::generate_n(std::back_inserter(source_column_wrappers), n_cols, [=]() {
    return cudf::test::fixed_width_column_wrapper<TypeParam>(data, data + source_size);
  });
  std::transform(source_column_wrappers.begin(),
                 source_column_wrappers.end(),
                 source_columns.begin(),
                 [](auto const& col) { return static_cast<cudf::column_view>(col); });

  cudf::table_view source_table{source_columns};

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf::gather(source_table, gather_map);
  }

  state.SetBytesProcessed(state.iterations() * state.range(0) * n_cols * 2 * sizeof(TypeParam));
}

#define GBM_BENCHMARK_DEFINE(name, type, coalesce)             \
  BENCHMARK_DEFINE_F(Gather, name)(::benchmark::State & state) \
  {                                                            \
    BM_gather<type, coalesce>(state);                          \
  }                                                            \
  BENCHMARK_REGISTER_F(Gather, name)                           \
    ->RangeMultiplier(2)                                       \
    ->Ranges({{1 << 10, 1 << 26}, {1, 8}})                     \
    ->UseManualTime();

GBM_BENCHMARK_DEFINE(double_coalesce_x, double, true);
GBM_BENCHMARK_DEFINE(double_coalesce_o, double, false);
