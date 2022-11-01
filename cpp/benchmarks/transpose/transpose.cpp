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

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transpose.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

static void BM_transpose(benchmark::State& state)
{
  auto count = state.range(0);
  auto int_column_generator =
    thrust::make_transform_iterator(thrust::counting_iterator(0), [count](int i) {
      return cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, count, cudf::mask_state::ALL_VALID);
    });

  auto input_table = cudf::table(std::vector(int_column_generator, int_column_generator + count));
  auto input       = input_table.view();

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
