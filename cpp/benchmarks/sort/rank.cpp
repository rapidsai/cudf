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

#include <cudf/column/column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>

class Rank : public cudf::benchmark {};

static void BM_rank(benchmark::State& state, bool nulls)
{
  using Type = int;
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};

  // Create columns with values in the range [0,100)
  data_profile profile = data_profile_builder().cardinality(0).distribution(
    cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, 100);
  profile.set_null_probability(nulls ? std::optional{0.2} : std::nullopt);
  auto keys = create_random_column(cudf::type_to_id<Type>(), row_count{n_rows}, profile);

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());

    auto result = cudf::rank(keys->view(),
                             cudf::rank_method::FIRST,
                             cudf::order::ASCENDING,
                             nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE,
                             cudf::null_order::AFTER,
                             false);
  }
}

#define RANK_BENCHMARK_DEFINE(name, nulls)          \
  BENCHMARK_DEFINE_F(Rank, name)                    \
  (::benchmark::State & st) { BM_rank(st, nulls); } \
  BENCHMARK_REGISTER_F(Rank, name)                  \
    ->RangeMultiplier(8)                            \
    ->Ranges({{1 << 10, 1 << 26}})                  \
    ->UseManualTime()                               \
    ->Unit(benchmark::kMillisecond);

RANK_BENCHMARK_DEFINE(no_nulls, false)
RANK_BENCHMARK_DEFINE(nulls, true)
