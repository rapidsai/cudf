/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf_test/column_wrapper.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <limits>

class StringSplit : public cudf::benchmark {
};

enum split_type { split, split_ws, record, record_ws };

static void BM_split(benchmark::State& state, split_type rt)
{
  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.range(0))};
  cudf::size_type const max_str_length{static_cast<cudf::size_type>(state.range(1))};
  data_profile const table_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const table = create_random_table({cudf::type_id::STRING}, row_count{n_rows}, table_profile);
  cudf::strings_column_view input(table->view().column(0));
  cudf::string_scalar target("+");

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::default_stream_value);
    switch (rt) {
      case split: cudf::strings::split(input, target); break;
      case split_ws: cudf::strings::split(input); break;
      case record: cudf::strings::split_record(input, target); break;
      case record_ws: cudf::strings::split_record(input); break;
    }
  }

  state.SetBytesProcessed(state.iterations() * input.chars_size());
}

static void generate_bench_args(benchmark::internal::Benchmark* b)
{
  int const min_rows   = 1 << 12;
  int const max_rows   = 1 << 24;
  int const row_mult   = 8;
  int const min_rowlen = 1 << 5;
  int const max_rowlen = 1 << 13;
  int const len_mult   = 4;
  for (int row_count = min_rows; row_count <= max_rows; row_count *= row_mult) {
    for (int rowlen = min_rowlen; rowlen <= max_rowlen; rowlen *= len_mult) {
      // avoid generating combinations that exceed the cudf column limit
      size_t total_chars = static_cast<size_t>(row_count) * rowlen;
      if (total_chars < static_cast<size_t>(std::numeric_limits<cudf::size_type>::max())) {
        b->Args({row_count, rowlen});
      }
    }
  }
}

#define STRINGS_BENCHMARK_DEFINE(name)                          \
  BENCHMARK_DEFINE_F(StringSplit, name)                         \
  (::benchmark::State & st) { BM_split(st, split_type::name); } \
  BENCHMARK_REGISTER_F(StringSplit, name)                       \
    ->Apply(generate_bench_args)                                \
    ->UseManualTime()                                           \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(split)
STRINGS_BENCHMARK_DEFINE(split_ws)
STRINGS_BENCHMARK_DEFINE(record)
STRINGS_BENCHMARK_DEFINE(record_ws)
