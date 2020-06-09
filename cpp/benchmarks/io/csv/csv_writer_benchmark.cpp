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

#include <benchmarks/common/generate_benchmark_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/io/cuio_benchmarks_common.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/io/functions.hpp>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

constexpr int64_t data_size = 512 << 19;  // 512 MB

namespace cudf_io = cudf::io;

template <typename T>
class CsvWrite : public cudf::benchmark {
};

template <typename T>
void CSV_write(benchmark::State& state)
{
  int64_t const total_bytes      = state.range(0);
  cudf::size_type const num_cols = state.range(1);

  int64_t const col_bytes = total_bytes / num_cols;

  auto const tbl  = create_random_table<T>(num_cols, col_bytes, true);
  auto const view = tbl->view();

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::write_csv_args args{cudf_io::sink_info(), view, "null", false, 1 << 30};
    cudf_io::write_csv(args);
  }

  state.SetBytesProcessed(total_bytes * state.iterations());
}

#define CSV_WR_BENCHMARK_DEFINE(name, datatype, compression)   \
  BENCHMARK_TEMPLATE_DEFINE_F(CsvWrite, name, datatype)        \
  (::benchmark::State & state) { CSV_write<datatype>(state); } \
  BENCHMARK_REGISTER_F(CsvWrite, name)                         \
    ->Args({data_size, 64})                                    \
    ->Unit(benchmark::kMillisecond)                            \
    ->UseManualTime();

// no compression support; compression parameter unused
CUIO_BENCH_ALL_TYPES(CSV_WR_BENCHMARK_DEFINE, UNCOMPRESSED)
