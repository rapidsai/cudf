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

constexpr int64_t data_size = 256 << 20;  // 256 MB

namespace cudf_io = cudf::io;

template <typename T>
class CsvRead : public cudf::benchmark {
};

template <typename T>
void CSV_read(benchmark::State& state)
{
  int64_t const total_bytes      = state.range(0);
  cudf::size_type const num_cols = state.range(1);

  int64_t const col_bytes = total_bytes / num_cols;
  std::vector<char> out_buffer;
  out_buffer.reserve(total_bytes);

  auto const tbl  = create_random_table<T>(num_cols, col_bytes, true);
  auto const view = tbl->view();

  cudf_io::write_csv_args args{cudf_io::sink_info(&out_buffer), view, "null", false, 1 << 30};
  cudf_io::write_csv(args);

  cudf_io::read_csv_args read_args(cudf_io::source_info(out_buffer.data(), out_buffer.size()));

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::read_csv(read_args);
  }

  state.SetBytesProcessed(total_bytes * state.iterations());
}

#define CSV_RD_BENCHMARK_DEFINE(name, datatype, compression)  \
  BENCHMARK_TEMPLATE_DEFINE_F(CsvRead, name, datatype)        \
  (::benchmark::State & state) { CSV_read<datatype>(state); } \
  BENCHMARK_REGISTER_F(CsvRead, name)                         \
    ->Args({data_size, 64})                                   \
    ->Unit(benchmark::kMillisecond)                           \
    ->UseManualTime();

// no compression support; compression parameter unused
CUIO_BENCH_ALL_TYPES(CSV_RD_BENCHMARK_DEFINE, UNCOMPRESSED)
