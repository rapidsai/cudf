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

#include "string_bench_args.hpp"

#include <benchmark/benchmark.h>
#include <benchmarks/common/generate_benchmark_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <limits>

namespace {
using string_pair = thrust::pair<char const*, cudf::size_type>;
struct string_view_to_pair {
  __device__ string_pair operator()(thrust::pair<cudf::string_view, bool> const& p)
  {
    return (p.second) ? string_pair{p.first.data(), p.first.size_bytes()} : string_pair{nullptr, 0};
  }
};
}  // namespace

class StringsFactory : public cudf::benchmark {
};

static void BM_factory(benchmark::State& state)
{
  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.range(0))};
  cudf::size_type const max_str_length{static_cast<cudf::size_type>(state.range(1))};
  data_profile table_profile;
  table_profile.set_distribution_params(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const table =
    create_random_table({cudf::type_id::STRING}, 1, row_count{n_rows}, table_profile);
  auto d_column = cudf::column_device_view::create(table->view().column(0));
  rmm::device_vector<string_pair> pairs(d_column->size());
  thrust::transform(thrust::device,
                    d_column->pair_begin<cudf::string_view, true>(),
                    d_column->pair_end<cudf::string_view, true>(),
                    pairs.data(),
                    string_view_to_pair{});

  for (auto _ : state) {
    cuda_event_timer raii(state, true, rmm::cuda_stream_default);
    cudf::make_strings_column(pairs);
  }

  cudf::strings_column_view input(table->view().column(0));
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
  generate_string_bench_args(b, min_rows, max_rows, row_mult, min_rowlen, max_rowlen, len_mult);
}

#define STRINGS_BENCHMARK_DEFINE(name)          \
  BENCHMARK_DEFINE_F(StringsFactory, name)      \
  (::benchmark::State & st) { BM_factory(st); } \
  BENCHMARK_REGISTER_F(StringsFactory, name)    \
    ->Apply(generate_bench_args)                \
    ->UseManualTime()                           \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(factory)
