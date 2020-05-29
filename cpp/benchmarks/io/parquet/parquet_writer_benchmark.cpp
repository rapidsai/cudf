/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include "../generate_input.hpp"

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/io/functions.hpp>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

constexpr int64_t data_size = 1 << 10;

namespace cudf_io = cudf::io;

template <typename T>
class ParquetWrite : public cudf::benchmark {
};

template <typename T>
void PQ_write(benchmark::State& state)
{
  int64_t const total_desired_bytes = state.range(0);
  cudf::size_type const num_cols    = state.range(1);

  constexpr auto el_size = sizeof(T);
  int64_t const num_rows = total_desired_bytes / (num_cols * el_size);

  auto const tbl  = create_random_table<T>(num_cols, num_rows, true);
  auto const view = tbl->view();

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::write_parquet_args args{cudf_io::sink_info(), view};
    cudf_io::write_parquet(args);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0));
}

#define PWBM_BENCHMARK_DEFINE(name, type)                 \
  BENCHMARK_TEMPLATE_DEFINE_F(ParquetWrite, name, type)   \
  (::benchmark::State & state) { PQ_write<type>(state); } \
  BENCHMARK_REGISTER_F(ParquetWrite, name)                \
    ->Args({data_size, 64})                               \
    ->Unit(benchmark::kMillisecond)                       \
    ->UseManualTime();

// TODO: cover all supported types here

PWBM_BENCHMARK_DEFINE(Short, short);
PWBM_BENCHMARK_DEFINE(Int, int);
PWBM_BENCHMARK_DEFINE(Long, long);

PWBM_BENCHMARK_DEFINE(Float, float);
PWBM_BENCHMARK_DEFINE(Double, double);

// TODO: add support for strings in create_random_table
// PWBM_BENCHMARK_DEFINE(String, std::string);