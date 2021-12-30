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

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>
#include <cudf/transpose.hpp>
#include <cudf/types.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <memory>
#include <thrust/transform.h>

using cudf::test::fixed_width_column_wrapper;

static void BM_transpose(benchmark::State& state)
{
  auto count = state.range(0);

  auto data     = std::vector<int>(count, 0);
  auto validity = std::vector<bool>(count, 1);

  auto fwcw_iter = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), [&data, &validity](auto idx) {
      return fixed_width_column_wrapper<int>(data.begin(), data.end(), validity.begin());
    });

  auto input_columns = std::vector<fixed_width_column_wrapper<int>>(fwcw_iter, fwcw_iter + count);

  auto input_column_views =
    std::vector<cudf::column_view>(input_columns.begin(), input_columns.end());

  auto input = cudf::table_view(input_column_views);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto output = cudf::transpose(input);
  }
}

class Transpose : public cudf::benchmark {
};

#define TRANSPOSE_BM_BENCHMARK_DEFINE(name)                                                \
  BENCHMARK_DEFINE_F(Transpose, name)(::benchmark::State & state) { BM_transpose(state); } \
  BENCHMARK_REGISTER_F(Transpose, name)                                                    \
    ->RangeMultiplier(4)                                                                   \
    ->Range(4, 4 << 13)                                                                    \
    ->UseManualTime()                                                                      \
    ->Unit(benchmark::kMillisecond);

TRANSPOSE_BM_BENCHMARK_DEFINE(transpose_simple);
