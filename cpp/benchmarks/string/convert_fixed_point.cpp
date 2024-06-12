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

#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/types.hpp>

namespace {

std::unique_ptr<cudf::column> get_strings_column(cudf::size_type rows)
{
  auto result =
    create_random_column(cudf::type_id::FLOAT32, row_count{static_cast<cudf::size_type>(rows)});
  return cudf::strings::from_floats(result->view());
}

}  // anonymous namespace

class StringsToFixedPoint : public cudf::benchmark {};

template <typename fixed_point_type>
void convert_to_fixed_point(benchmark::State& state)
{
  auto const rows         = static_cast<cudf::size_type>(state.range(0));
  auto const strings_col  = get_strings_column(rows);
  auto const strings_view = cudf::strings_column_view(strings_col->view());
  auto const dtype = cudf::data_type{cudf::type_to_id<fixed_point_type>(), numeric::scale_type{-2}};

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto volatile results = cudf::strings::to_fixed_point(strings_view, dtype);
  }

  // bytes_processed = bytes_input + bytes_output
  state.SetBytesProcessed(
    state.iterations() *
    (strings_view.chars_size(cudf::get_default_stream()) + rows * cudf::size_of(dtype)));
}

class StringsFromFixedPoint : public cudf::benchmark {};

template <typename fixed_point_type>
void convert_from_fixed_point(benchmark::State& state)
{
  auto const rows        = static_cast<cudf::size_type>(state.range(0));
  auto const strings_col = get_strings_column(rows);
  auto const dtype = cudf::data_type{cudf::type_to_id<fixed_point_type>(), numeric::scale_type{-2}};
  auto const fp_col =
    cudf::strings::to_fixed_point(cudf::strings_column_view(strings_col->view()), dtype);

  std::unique_ptr<cudf::column> results = nullptr;

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    results = cudf::strings::from_fixed_point(fp_col->view());
  }

  // bytes_processed = bytes_input + bytes_output
  state.SetBytesProcessed(
    state.iterations() *
    (cudf::strings_column_view(results->view()).chars_size(cudf::get_default_stream()) +
     rows * cudf::size_of(dtype)));
}

#define CONVERT_TO_FIXED_POINT_BMD(name, fixed_point_type)                  \
  BENCHMARK_DEFINE_F(StringsToFixedPoint, name)(::benchmark::State & state) \
  {                                                                         \
    convert_to_fixed_point<fixed_point_type>(state);                        \
  }                                                                         \
  BENCHMARK_REGISTER_F(StringsToFixedPoint, name)                           \
    ->RangeMultiplier(4)                                                    \
    ->Range(1 << 12, 1 << 24)                                               \
    ->UseManualTime()                                                       \
    ->Unit(benchmark::kMicrosecond);

#define CONVERT_FROM_FIXED_POINT_BMD(name, fixed_point_type)                  \
  BENCHMARK_DEFINE_F(StringsFromFixedPoint, name)(::benchmark::State & state) \
  {                                                                           \
    convert_from_fixed_point<fixed_point_type>(state);                        \
  }                                                                           \
  BENCHMARK_REGISTER_F(StringsFromFixedPoint, name)                           \
    ->RangeMultiplier(4)                                                      \
    ->Range(1 << 12, 1 << 24)                                                 \
    ->UseManualTime()                                                         \
    ->Unit(benchmark::kMicrosecond);

CONVERT_TO_FIXED_POINT_BMD(strings_to_decimal32, numeric::decimal32);
CONVERT_TO_FIXED_POINT_BMD(strings_to_decimal64, numeric::decimal64);

CONVERT_FROM_FIXED_POINT_BMD(strings_from_decimal32, numeric::decimal32);
CONVERT_FROM_FIXED_POINT_BMD(strings_from_decimal64, numeric::decimal64);
