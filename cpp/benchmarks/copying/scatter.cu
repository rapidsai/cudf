/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/types.hpp>

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/reverse.h>
#include <thrust/shuffle.h>

class Scatter : public cudf::benchmark {};

template <class TypeParam, bool coalesce>
void BM_scatter(benchmark::State& state)
{
  auto const source_size{static_cast<cudf::size_type>(state.range(0))};
  auto const n_cols{static_cast<cudf::size_type>(state.range(1))};

  // Gather indices
  auto scatter_map_table =
    create_sequence_table({cudf::type_to_id<cudf::size_type>()}, row_count{source_size});
  auto scatter_map = scatter_map_table->get_column(0).mutable_view();

  if (coalesce) {
    thrust::reverse(
      thrust::device, scatter_map.begin<cudf::size_type>(), scatter_map.end<cudf::size_type>());
  } else {
    thrust::shuffle(thrust::device,
                    scatter_map.begin<cudf::size_type>(),
                    scatter_map.end<cudf::size_type>(),
                    thrust::default_random_engine());
  }

  // Every element is valid
  auto source_table = create_sequence_table(cycle_dtypes({cudf::type_to_id<TypeParam>()}, n_cols),
                                            row_count{source_size});
  auto target_table = create_sequence_table(cycle_dtypes({cudf::type_to_id<TypeParam>()}, n_cols),
                                            row_count{source_size});

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf::scatter(*source_table, scatter_map, *target_table);
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

SBM_BENCHMARK_DEFINE(double_coalesced, double, true);
SBM_BENCHMARK_DEFINE(double_shuffled, double, false);
