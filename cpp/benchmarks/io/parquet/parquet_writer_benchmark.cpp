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

#include <cudf/io/parquet.hpp>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

constexpr int64_t data_size = 512 << 20;  // 512 MB

namespace cudf_io = cudf::io;

template <typename T>
class ParquetWrite : public cudf::benchmark {
};

template <typename T>
void PQ_write(benchmark::State& state)
{
  int64_t const total_bytes      = state.range(0);
  cudf::size_type const num_cols = state.range(1);
  cudf_io::compression_type const compression =
    state.range(2) ? cudf_io::compression_type::SNAPPY : cudf_io::compression_type::NONE;

  int64_t const col_bytes = total_bytes / num_cols;

  auto const tbl  = create_random_table<T>(num_cols, col_bytes, true);
  auto const view = tbl->view();

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::parquet_writer_options opts =
      cudf_io::parquet_writer_options::builder(cudf_io::sink_info(), view).compression(compression);
    cudf_io::write_parquet(opts);
  }

  state.SetBytesProcessed(total_bytes * state.iterations());
}

#define PARQ_WR_BENCHMARK_DEFINE(name, datatype, compression) \
  BENCHMARK_TEMPLATE_DEFINE_F(ParquetWrite, name, datatype)   \
  (::benchmark::State & state) { PQ_write<datatype>(state); } \
  BENCHMARK_REGISTER_F(ParquetWrite, name)                    \
    ->Args({data_size, 64, compression})                      \
    ->Unit(benchmark::kMillisecond)                           \
    ->UseManualTime();

CUIO_BENCH_ALL_TYPES(PARQ_WR_BENCHMARK_DEFINE, UNCOMPRESSED)
CUIO_BENCH_ALL_TYPES(PARQ_WR_BENCHMARK_DEFINE, USE_SNAPPY)
