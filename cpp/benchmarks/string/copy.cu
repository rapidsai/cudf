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

#include "string_bench_args.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

class StringCopy : public cudf::benchmark {};

enum copy_type { gather, scatter };

static void BM_copy(benchmark::State& state, copy_type ct)
{
  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.range(0))};
  cudf::size_type const max_str_length{static_cast<cudf::size_type>(state.range(1))};
  data_profile const table_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);

  auto const source =
    create_random_table({cudf::type_id::STRING}, row_count{n_rows}, table_profile);
  auto const target =
    create_random_table({cudf::type_id::STRING}, row_count{n_rows}, table_profile);

  // scatter indices
  auto index_map_col = make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n_rows, cudf::mask_state::UNALLOCATED);
  auto index_map = index_map_col->mutable_view();
  thrust::shuffle_copy(thrust::device,
                       thrust::counting_iterator<cudf::size_type>(0),
                       thrust::counting_iterator<cudf::size_type>(n_rows),
                       index_map.begin<cudf::size_type>(),
                       thrust::default_random_engine());

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());
    switch (ct) {
      case gather: cudf::gather(source->view(), index_map); break;
      case scatter: cudf::scatter(source->view(), index_map, target->view()); break;
    }
  }

  state.SetBytesProcessed(
    state.iterations() *
    cudf::strings_column_view(source->view().column(0)).chars_size(cudf::get_default_stream()));
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

  // Benchmark for very small strings
  b->Args({67108864, 2});
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
