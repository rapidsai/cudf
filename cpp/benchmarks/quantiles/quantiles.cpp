/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cudf/quantiles.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>

class Quantiles : public cudf::benchmark {};

static void BM_quantiles(benchmark::State& state, bool nulls)
{
  using Type = int;

  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  cudf::size_type const n_cols{(cudf::size_type)state.range(1)};
  cudf::size_type const n_quantiles{(cudf::size_type)state.range(2)};

  // Create columns with values in the range [0,100)
  data_profile profile = data_profile_builder().cardinality(0).distribution(
    cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, 100);
  profile.set_null_probability(nulls ? std::optional{0.01}
                                     : std::nullopt);  // 1% nulls or no null mask (<0)

  auto input_table = create_random_table(
    cycle_dtypes({cudf::type_to_id<Type>()}, n_cols), row_count{n_rows}, profile);
  auto input = cudf::table_view(*input_table);

  std::vector<double> q(n_quantiles);
  thrust::tabulate(
    thrust::seq, q.begin(), q.end(), [n_quantiles](auto i) { return i * (1.0f / n_quantiles); });

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());

    auto result = cudf::quantiles(input, q);
    // auto result = (stable) ? cudf::stable_sorted_order(input) : cudf::sorted_order(input);
  }
}

#define QUANTILES_BENCHMARK_DEFINE(name, nulls)          \
  BENCHMARK_DEFINE_F(Quantiles, name)                    \
  (::benchmark::State & st) { BM_quantiles(st, nulls); } \
  BENCHMARK_REGISTER_F(Quantiles, name)                  \
    ->RangeMultiplier(4)                                 \
    ->Ranges({{1 << 16, 1 << 26}, {1, 8}, {1, 12}})      \
    ->UseManualTime()                                    \
    ->Unit(benchmark::kMillisecond);

QUANTILES_BENCHMARK_DEFINE(no_nulls, false)
QUANTILES_BENCHMARK_DEFINE(nulls, true)
