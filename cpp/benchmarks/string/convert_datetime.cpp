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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/wrappers/timestamps.hpp>

class StringDateTime : public cudf::benchmark {};

enum class direction { to, from };

template <class TypeParam>
void BM_convert_datetime(benchmark::State& state, direction dir)
{
  auto const n_rows    = static_cast<cudf::size_type>(state.range(0));
  auto const data_type = cudf::data_type(cudf::type_to_id<TypeParam>());

  auto const column = create_random_column(data_type.id(), row_count{n_rows});
  cudf::column_view input(column->view());

  auto source = dir == direction::to ? cudf::strings::from_timestamps(input, "%Y-%m-%d %H:%M:%S")
                                     : make_empty_column(cudf::data_type{cudf::type_id::STRING});
  cudf::strings_column_view source_string(source->view());

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    if (dir == direction::to)
      cudf::strings::to_timestamps(source_string, data_type, "%Y-%m-%d %H:%M:%S");
    else
      cudf::strings::from_timestamps(input, "%Y-%m-%d %H:%M:%S");
  }

  auto const bytes = dir == direction::to ? source_string.chars_size(cudf::get_default_stream())
                                          : n_rows * sizeof(TypeParam);
  state.SetBytesProcessed(state.iterations() * bytes);
}

#define STR_BENCHMARK_DEFINE(name, type, dir)                          \
  BENCHMARK_DEFINE_F(StringDateTime, name)(::benchmark::State & state) \
  {                                                                    \
    BM_convert_datetime<type>(state, dir);                             \
  }                                                                    \
  BENCHMARK_REGISTER_F(StringDateTime, name)                           \
    ->RangeMultiplier(1 << 5)                                          \
    ->Range(1 << 10, 1 << 25)                                          \
    ->UseManualTime()                                                  \
    ->Unit(benchmark::kMicrosecond);

STR_BENCHMARK_DEFINE(from_days, cudf::timestamp_D, direction::from);
STR_BENCHMARK_DEFINE(from_seconds, cudf::timestamp_s, direction::from);
STR_BENCHMARK_DEFINE(from_mseconds, cudf::timestamp_ms, direction::from);
STR_BENCHMARK_DEFINE(from_useconds, cudf::timestamp_us, direction::from);
STR_BENCHMARK_DEFINE(from_nseconds, cudf::timestamp_ns, direction::from);

STR_BENCHMARK_DEFINE(to_days, cudf::timestamp_D, direction::to);
STR_BENCHMARK_DEFINE(to_seconds, cudf::timestamp_s, direction::to);
STR_BENCHMARK_DEFINE(to_mseconds, cudf::timestamp_ms, direction::to);
STR_BENCHMARK_DEFINE(to_useconds, cudf::timestamp_us, direction::to);
STR_BENCHMARK_DEFINE(to_nseconds, cudf::timestamp_ns, direction::to);
