/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cudf/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>

template <bool stable>
class Sort : public cudf::benchmark {};

template <bool stable>
static void BM_sort(benchmark::State& state, bool nulls)
{
  using Type       = int;
  auto const dtype = cudf::type_to_id<Type>();
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  cudf::size_type const n_cols{(cudf::size_type)state.range(1)};

  // Create table with values in the range [0,100)
  data_profile const profile = data_profile_builder()
                                 .cardinality(0)
                                 .null_probability(nulls ? std::optional{0.01} : std::nullopt)
                                 .distribution(dtype, distribution_id::UNIFORM, 0, 100);
  auto input_table = create_random_table(cycle_dtypes({dtype}, n_cols), row_count{n_rows}, profile);
  cudf::table_view input{*input_table};

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());

    auto result = (stable) ? cudf::stable_sorted_order(input) : cudf::sorted_order(input);
  }
}

#define SORT_BENCHMARK_DEFINE(name, stable, nulls)          \
  BENCHMARK_TEMPLATE_DEFINE_F(Sort, name, stable)           \
  (::benchmark::State & st) { BM_sort<stable>(st, nulls); } \
  BENCHMARK_REGISTER_F(Sort, name)                          \
    ->RangeMultiplier(8)                                    \
    ->Ranges({{1 << 10, 1 << 26}, {1, 8}})                  \
    ->UseManualTime()                                       \
    ->Unit(benchmark::kMillisecond);

SORT_BENCHMARK_DEFINE(unstable_no_nulls, false, false)
SORT_BENCHMARK_DEFINE(stable_no_nulls, true, false)
SORT_BENCHMARK_DEFINE(unstable, false, true)
SORT_BENCHMARK_DEFINE(stable, true, true)
