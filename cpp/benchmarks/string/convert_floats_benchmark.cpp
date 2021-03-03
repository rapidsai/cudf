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

#include <fixture/benchmark_fixture.hpp>
#include <synchronization/synchronization.hpp>

#include <benchmark/benchmark.h>
#include <benchmarks/common/generate_benchmark_input.hpp>

#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

namespace {
template <class FloatType>
std::unique_ptr<cudf::column> get_floats_column(int64_t array_size)
{
  std::unique_ptr<cudf::table> tbl;
  if (sizeof(FloatType) == sizeof(float)) {
    tbl = create_random_table(
      {cudf::type_id::FLOAT32}, 1, row_count{static_cast<cudf::size_type>(array_size)});
  } else {
    tbl = create_random_table(
      {cudf::type_id::FLOAT64}, 1, row_count{static_cast<cudf::size_type>(array_size)});
  }
  return std::move(tbl->release().front());
}

std::unique_ptr<cudf::column> get_floats_string_column(int64_t array_size)
{
  const auto floats = get_floats_column<double>(array_size);
  return cudf::strings::from_floats(floats->view());
}
}  // anonymous namespace

class StringToFloatNumber : public cudf::benchmark {
};

template <cudf::type_id float_type>
void convert_to_float_number(benchmark::State& state)
{
  const auto array_size   = state.range(0);
  const auto strings_col  = get_floats_string_column(array_size);
  const auto strings_view = cudf::strings_column_view(strings_col->view());

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    volatile auto results = cudf::strings::to_floats(strings_view, cudf::data_type{float_type});
  }

  // bytes_processed = bytes_input + bytes_output
  state.SetBytesProcessed(
    state.iterations() *
    (strings_view.chars_size() + array_size * cudf::size_of(cudf::data_type{float_type})));
}

class StringFromFloatNumber : public cudf::benchmark {
};

template <class FloatType>
void convert_from_float_number(benchmark::State& state)
{
  const auto array_size                 = state.range(0);
  const auto floats                     = get_floats_column<FloatType>(array_size);
  const auto floats_view                = floats->view();
  std::unique_ptr<cudf::column> results = nullptr;

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    results = cudf::strings::from_floats(floats_view);
  }

  // bytes_processed = bytes_input + bytes_output
  state.SetBytesProcessed(
    state.iterations() *
    (cudf::strings_column_view(results->view()).chars_size() + array_size * sizeof(FloatType)));
}

#define CV_TO_FLOATS_BENCHMARK_DEFINE(name, float_type_id)                  \
  BENCHMARK_DEFINE_F(StringToFloatNumber, name)(::benchmark::State & state) \
  {                                                                         \
    convert_to_float_number<float_type_id>(state);                          \
  }                                                                         \
  BENCHMARK_REGISTER_F(StringToFloatNumber, name)                           \
    ->RangeMultiplier(4)                                                    \
    ->Range(1 << 10, 1 << 17)                                               \
    ->UseManualTime()                                                       \
    ->Unit(benchmark::kMicrosecond);

#define CV_FROM_FLOATS_BENCHMARK_DEFINE(name, float_type)                     \
  BENCHMARK_DEFINE_F(StringFromFloatNumber, name)(::benchmark::State & state) \
  {                                                                           \
    convert_from_float_number<float_type>(state);                             \
  }                                                                           \
  BENCHMARK_REGISTER_F(StringFromFloatNumber, name)                           \
    ->RangeMultiplier(4)                                                      \
    ->Range(1 << 10, 1 << 17)                                                 \
    ->UseManualTime()                                                         \
    ->Unit(benchmark::kMicrosecond);

CV_TO_FLOATS_BENCHMARK_DEFINE(string_to_float32, cudf::type_id::FLOAT32);
CV_TO_FLOATS_BENCHMARK_DEFINE(string_to_float64, cudf::type_id::FLOAT64);

CV_FROM_FLOATS_BENCHMARK_DEFINE(string_from_float32, float);
CV_FROM_FLOATS_BENCHMARK_DEFINE(string_from_float64, double);
