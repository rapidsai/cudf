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

#include "../fixture/benchmark_fixture.hpp"
#include "../synchronization/synchronization.hpp"

#include <benchmark/benchmark.h>

#include <cudf/column/column_view.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <numeric>
#include <random>
#include <unordered_map>

namespace {

// For each array_size, this function is called twice from both StringToFloatNumber and
// StringFromFloatNumber classes. Thus, the results are cached for reuse.
template <class FloatType>
static const std::vector<FloatType>& get_float_numbers(int64_t array_size)
{
  static std::unordered_map<int64_t, std::vector<FloatType>> number_arrays;
  auto& numbers = number_arrays[array_size];
  if (numbers.size() == 0) {
    numbers.reserve(array_size);
    cudf::test::UniformRandomGenerator<FloatType> rand_gen(std::numeric_limits<FloatType>::min(),
                                                           std::numeric_limits<FloatType>::max());
    std::generate_n(
      std::back_inserter(numbers), array_size, [&rand_gen]() { return rand_gen.generate(); });
  }
  return numbers;
}

template <class FloatType>
static std::vector<std::string> get_floats_numbers_as_string(int64_t array_size)
{
  std::vector<std::string> numbers_str(array_size);
  const auto& numbers = get_float_numbers<FloatType>(array_size);
  std::transform(
    numbers.begin(), numbers.end(), numbers_str.begin(), [](auto x) { return std::to_string(x); });
  return numbers_str;
}

}  // anonymous namespace

class StringToFloatNumber : public cudf::benchmark {
};
template <cudf::type_id float_type>
void convert_to_float_number(benchmark::State& state)
{
  const auto& h_strings   = get_floats_numbers_as_string<double>(state.range(0));
  const auto strings_size = std::accumulate(
    h_strings.begin(), h_strings.end(), std::size_t{0}, [](std::size_t size, const auto& str) {
      return size + str.length();
    });

  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  const auto strings_view = cudf::strings_column_view(strings);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    volatile auto results = cudf::strings::to_floats(strings_view, cudf::data_type{float_type});
  }

  state.SetBytesProcessed(state.iterations() * strings_size);
}

class StringFromFloatNumber : public cudf::benchmark {
};
template <class FloatType>
void convert_from_float_number(benchmark::State& state)
{
  const auto& h_floats   = get_float_numbers<FloatType>(state.range(0));
  const auto floats_size = h_floats.size() * sizeof(FloatType);

  cudf::test::fixed_width_column_wrapper<FloatType> floats(h_floats.begin(), h_floats.end());
  const auto floats_view = cudf::column_view(floats);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    volatile auto results = cudf::strings::from_floats(floats_view);
  }

  state.SetBytesProcessed(state.iterations() * floats_size);
}

#define CV_TO_FLOATS_BENCHMARK_DEFINE(name, float_type_id)                  \
  BENCHMARK_DEFINE_F(StringToFloatNumber, name)(::benchmark::State & state) \
  {                                                                         \
    convert_to_float_number<float_type_id>(state);                          \
  }                                                                         \
  BENCHMARK_REGISTER_F(StringToFloatNumber, name)                           \
    ->RangeMultiplier(1 << 2)                                               \
    ->Range(1 << 10, 1 << 17)                                               \
    ->UseManualTime()                                                       \
    ->Unit(benchmark::kMicrosecond);

#define CV_FROM_FLOATS_BENCHMARK_DEFINE(name, float_type)                     \
  BENCHMARK_DEFINE_F(StringFromFloatNumber, name)(::benchmark::State & state) \
  {                                                                           \
    convert_from_float_number<float_type>(state);                             \
  }                                                                           \
  BENCHMARK_REGISTER_F(StringFromFloatNumber, name)                           \
    ->RangeMultiplier(1 << 2)                                                 \
    ->Range(1 << 10, 1 << 17)                                                 \
    ->UseManualTime()                                                         \
    ->Unit(benchmark::kMicrosecond);

CV_TO_FLOATS_BENCHMARK_DEFINE(string_to_float32, cudf::type_id::FLOAT32);
CV_TO_FLOATS_BENCHMARK_DEFINE(string_to_float64, cudf::type_id::FLOAT64);

CV_FROM_FLOATS_BENCHMARK_DEFINE(string_from_float32, float);
CV_FROM_FLOATS_BENCHMARK_DEFINE(string_from_float64, double);
