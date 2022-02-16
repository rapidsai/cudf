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

#include <benchmark/benchmark.h>
#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>

#include <thrust/reverse.h>
#include <thrust/shuffle.h>

#include <algorithm>

class Scatter : public cudf::benchmark {
};

template <class TypeParam, bool coalesce>
void BM_scatter(benchmark::State& state)
{
  const cudf::size_type source_size{(cudf::size_type)state.range(0)};
  const auto n_cols = (cudf::size_type)state.range(1);

  // Gather indices
  auto init = cudf::make_fixed_width_scalar<cudf::size_type>(static_cast<cudf::size_type>(0));
  auto scatter_map      = cudf::sequence(source_size, *init);
  auto scatter_map_view = scatter_map->mutable_view();

  if (coalesce) {
    thrust::reverse(thrust::device,
                    scatter_map_view.begin<cudf::size_type>(),
                    scatter_map_view.end<cudf::size_type>());
  } else {
    thrust::shuffle(thrust::device,
                    scatter_map_view.begin<cudf::size_type>(),
                    scatter_map_view.end<cudf::size_type>(),
                    thrust::default_random_engine());
  }

  // Every element is valid
  auto source_table =
    create_sequence_table({cudf::type_to_id<TypeParam>()}, n_cols, row_count{source_size});
  auto target_table =
    create_sequence_table({cudf::type_to_id<TypeParam>()}, n_cols, row_count{source_size});

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf::scatter(*source_table, scatter_map_view, *target_table);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) * n_cols * 2 *
                          sizeof(TypeParam));
}

#define SBM_BENCHMARK_DEFINE(name, type, coalesce)              \
  BENCHMARK_DEFINE_F(Scatter, name)(::benchmark::State & state) \
  {                                                             \
    BM_scatter<type, coalesce>(state);                          \
  }                                                             \
  BENCHMARK_REGISTER_F(Scatter, name)                           \
    ->RangeMultiplier(2)                                        \
    ->Ranges({{1 << 10, 1 << 25}, {1, 8}})                      \
    ->UseManualTime();

SBM_BENCHMARK_DEFINE(double_coalesce_x, double, true);
SBM_BENCHMARK_DEFINE(double_coalesce_o, double, false);
