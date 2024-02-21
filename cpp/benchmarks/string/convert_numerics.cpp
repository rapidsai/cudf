/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/types.hpp>

namespace {

template <typename NumericType>
std::unique_ptr<cudf::column> get_numerics_column(cudf::size_type rows)
{
  return create_random_column(cudf::type_to_id<NumericType>(), row_count{rows});
}

template <typename NumericType>
std::unique_ptr<cudf::column> get_strings_column(cudf::size_type rows)
{
  auto const numerics_col = get_numerics_column<NumericType>(rows);
  if constexpr (std::is_floating_point_v<NumericType>) {
    return cudf::strings::from_floats(numerics_col->view());
  } else {
    return cudf::strings::from_integers(numerics_col->view());
  }
}
}  // anonymous namespace

class StringsToNumeric : public cudf::benchmark {};

template <typename NumericType>
void convert_to_number(benchmark::State& state)
{
  auto const rows = static_cast<cudf::size_type>(state.range(0));

  auto const strings_col  = get_strings_column<NumericType>(rows);
  auto const strings_view = cudf::strings_column_view(strings_col->view());
  auto const col_type     = cudf::type_to_id<NumericType>();

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    if constexpr (std::is_floating_point_v<NumericType>) {
      cudf::strings::to_floats(strings_view, cudf::data_type{col_type});
    } else {
      cudf::strings::to_integers(strings_view, cudf::data_type{col_type});
    }
  }

  // bytes_processed = bytes_input + bytes_output
  state.SetBytesProcessed(
    state.iterations() *
    (strings_view.chars_size(cudf::get_default_stream()) + rows * sizeof(NumericType)));
}

class StringsFromNumeric : public cudf::benchmark {};

template <typename NumericType>
void convert_from_number(benchmark::State& state)
{
  auto const rows = static_cast<cudf::size_type>(state.range(0));

  auto const numerics_col  = get_numerics_column<NumericType>(rows);
  auto const numerics_view = numerics_col->view();

  std::unique_ptr<cudf::column> results = nullptr;

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    if constexpr (std::is_floating_point_v<NumericType>)
      results = cudf::strings::from_floats(numerics_view);
    else
      results = cudf::strings::from_integers(numerics_view);
  }

  // bytes_processed = bytes_input + bytes_output
  state.SetBytesProcessed(
    state.iterations() *
    (cudf::strings_column_view(results->view()).chars_size(cudf::get_default_stream()) +
     rows * sizeof(NumericType)));
}

#define CONVERT_TO_NUMERICS_BD(name, type)                               \
  BENCHMARK_DEFINE_F(StringsToNumeric, name)(::benchmark::State & state) \
  {                                                                      \
    convert_to_number<type>(state);                                      \
  }                                                                      \
  BENCHMARK_REGISTER_F(StringsToNumeric, name)                           \
    ->RangeMultiplier(4)                                                 \
    ->Range(1 << 10, 1 << 17)                                            \
    ->UseManualTime()                                                    \
    ->Unit(benchmark::kMicrosecond);

#define CONVERT_FROM_NUMERICS_BD(name, type)                               \
  BENCHMARK_DEFINE_F(StringsFromNumeric, name)(::benchmark::State & state) \
  {                                                                        \
    convert_from_number<type>(state);                                      \
  }                                                                        \
  BENCHMARK_REGISTER_F(StringsFromNumeric, name)                           \
    ->RangeMultiplier(4)                                                   \
    ->Range(1 << 10, 1 << 17)                                              \
    ->UseManualTime()                                                      \
    ->Unit(benchmark::kMicrosecond);

CONVERT_TO_NUMERICS_BD(strings_to_float32, float);
CONVERT_TO_NUMERICS_BD(strings_to_float64, double);
CONVERT_TO_NUMERICS_BD(strings_to_int32, int32_t);
CONVERT_TO_NUMERICS_BD(strings_to_int64, int64_t);
CONVERT_TO_NUMERICS_BD(strings_to_uint8, uint8_t);
CONVERT_TO_NUMERICS_BD(strings_to_uint16, uint16_t);

CONVERT_FROM_NUMERICS_BD(strings_from_float32, float);
CONVERT_FROM_NUMERICS_BD(strings_from_float64, double);
CONVERT_FROM_NUMERICS_BD(strings_from_int32, int32_t);
CONVERT_FROM_NUMERICS_BD(strings_from_int64, int64_t);
CONVERT_FROM_NUMERICS_BD(strings_from_uint8, uint8_t);
CONVERT_FROM_NUMERICS_BD(strings_from_uint16, uint16_t);
