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

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>

#include <algorithm>
#include <random>

class DurationsToString : public cudf::benchmark {};
template <class TypeParam>
void BM_convert_from_durations(benchmark::State& state)
{
  cudf::size_type const source_size = state.range(0);

  // Every element is valid
  auto data = cudf::detail::make_counting_transform_iterator(
    0, [source_size](auto i) { return TypeParam{i - source_size / 2}; });

  cudf::test::fixed_width_column_wrapper<TypeParam> source_durations(data, data + source_size);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf::strings::from_durations(source_durations, "%D days %H:%M:%S");
  }

  state.SetBytesProcessed(state.iterations() * source_size * sizeof(TypeParam));
}

class StringToDurations : public cudf::benchmark {};
template <class TypeParam>
void BM_convert_to_durations(benchmark::State& state)
{
  cudf::size_type const source_size = state.range(0);

  // Every element is valid
  auto data = cudf::detail::make_counting_transform_iterator(
    0, [source_size](auto i) { return TypeParam{i - source_size / 2}; });

  cudf::test::fixed_width_column_wrapper<TypeParam> source_durations(data, data + source_size);
  auto results = cudf::strings::from_durations(source_durations, "%D days %H:%M:%S");
  cudf::strings_column_view source_string(*results);
  auto output_type = cudf::data_type(cudf::type_to_id<TypeParam>());

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf::strings::to_durations(source_string, output_type, "%D days %H:%M:%S");
  }

  state.SetBytesProcessed(state.iterations() * source_size * sizeof(TypeParam));
}

#define DSBM_BENCHMARK_DEFINE(name, type)                                 \
  BENCHMARK_DEFINE_F(DurationsToString, name)(::benchmark::State & state) \
  {                                                                       \
    BM_convert_from_durations<type>(state);                               \
  }                                                                       \
  BENCHMARK_REGISTER_F(DurationsToString, name)                           \
    ->RangeMultiplier(1 << 5)                                             \
    ->Range(1 << 10, 1 << 25)                                             \
    ->UseManualTime()                                                     \
    ->Unit(benchmark::kMicrosecond);

#define SDBM_BENCHMARK_DEFINE(name, type)                                 \
  BENCHMARK_DEFINE_F(StringToDurations, name)(::benchmark::State & state) \
  {                                                                       \
    BM_convert_to_durations<type>(state);                                 \
  }                                                                       \
  BENCHMARK_REGISTER_F(StringToDurations, name)                           \
    ->RangeMultiplier(1 << 5)                                             \
    ->Range(1 << 10, 1 << 25)                                             \
    ->UseManualTime()                                                     \
    ->Unit(benchmark::kMicrosecond);

DSBM_BENCHMARK_DEFINE(from_durations_D, cudf::duration_D);
DSBM_BENCHMARK_DEFINE(from_durations_s, cudf::duration_s);
DSBM_BENCHMARK_DEFINE(from_durations_ms, cudf::duration_ms);
DSBM_BENCHMARK_DEFINE(from_durations_us, cudf::duration_us);
DSBM_BENCHMARK_DEFINE(from_durations_ns, cudf::duration_ns);

SDBM_BENCHMARK_DEFINE(to_durations_D, cudf::duration_D);
SDBM_BENCHMARK_DEFINE(to_durations_s, cudf::duration_s);
SDBM_BENCHMARK_DEFINE(to_durations_ms, cudf::duration_ms);
SDBM_BENCHMARK_DEFINE(to_durations_us, cudf::duration_us);
SDBM_BENCHMARK_DEFINE(to_durations_ns, cudf::duration_ns);
