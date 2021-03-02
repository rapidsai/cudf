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

#include <benchmark/benchmark.h>
#include <benchmarks/common/generate_benchmark_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <algorithm>
#include <random>

#include "string_bench_args.hpp"

class StringCopy : public cudf::benchmark {
};

enum copy_type { gather, scatter };

static void BM_copy(benchmark::State& state, copy_type ct)
{
  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.range(0))};
  cudf::size_type const max_str_length{static_cast<cudf::size_type>(state.range(1))};
  data_profile table_profile;
  table_profile.set_distribution_params(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);

  auto const source =
    create_random_table({cudf::type_id::STRING}, 1, row_count{n_rows}, table_profile);
  auto const target =
    create_random_table({cudf::type_id::STRING}, 1, row_count{n_rows}, table_profile);

  // scatter indices
  std::vector<cudf::size_type> host_map_data(n_rows);
  std::iota(host_map_data.begin(), host_map_data.end(), 0);
  std::random_shuffle(host_map_data.begin(), host_map_data.end());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> index_map(host_map_data.begin(),
                                                                    host_map_data.end());

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);
    switch (ct) {
      case gather: cudf::gather(source->view(), index_map); break;
      case scatter: cudf::scatter(source->view(), index_map, target->view()); break;
    }
  }

  state.SetBytesProcessed(state.iterations() *
                          cudf::strings_column_view(source->view().column(0)).chars_size());
}

static void generate_bench_args(benchmark::internal::Benchmark* b)
{
  int const min_rows   = 1 << 12;
  int const max_rows   = 1 << 24;
  int const row_mult   = 8;
  int const min_rowlen = 1 << 5;
  int const max_rowlen = 1 << 13;
  int const len_mult   = 4;
  generate_string_bench_args(b, min_rows, max_rows, row_mult, min_rowlen, max_rowlen, len_mult);
}

#define COPY_BENCHMARK_DEFINE(name)                           \
  BENCHMARK_DEFINE_F(StringCopy, name)                        \
  (::benchmark::State & st) { BM_copy(st, copy_type::name); } \
  BENCHMARK_REGISTER_F(StringCopy, name)                      \
    ->Apply(generate_bench_args)                              \
    ->UseManualTime()                                         \
    ->Unit(benchmark::kMillisecond);

COPY_BENCHMARK_DEFINE(gather)
COPY_BENCHMARK_DEFINE(scatter)
