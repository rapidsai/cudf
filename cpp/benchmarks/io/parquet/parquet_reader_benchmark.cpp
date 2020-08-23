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

constexpr int64_t data_size = 256 << 20;

namespace cudf_io = cudf::io;

class ParquetRead : public cudf::benchmark {
};

void PQ_read(benchmark::State& state)
{
  cudf::type_id const data_type  = static_cast<cudf::type_id>(state.range(0));
  int64_t const total_bytes      = state.range(1);
  cudf::size_type const num_cols = state.range(2);
  cudf_io::compression_type const compression =
    state.range(3) ? cudf_io::compression_type::SNAPPY : cudf_io::compression_type::NONE;

  int64_t const col_bytes = total_bytes / num_cols;
  std::vector<char> out_buffer;
  out_buffer.reserve(total_bytes);

  auto const tbl  = create_random_table(data_type, num_cols, col_bytes, true);
  auto const view = tbl->view();

  cudf_io::write_parquet_args write_args{
    cudf_io::sink_info(&out_buffer), view, nullptr, compression};
  cudf_io::write_parquet(write_args);

  cudf_io::read_parquet_args read_args{cudf_io::source_info(out_buffer.data(), out_buffer.size())};

  for (auto _ : state) {
    cuda_event_timer const raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::read_parquet(read_args);
  }

  state.SetBytesProcessed(total_bytes * state.iterations());
}

#define PARQ_RD_BENCHMARK_DEFINE(name, datatype, compression)            \
  BENCHMARK_DEFINE_F(ParquetRead, name)                                  \
  (::benchmark::State & state) { PQ_read(state); }                       \
  BENCHMARK_REGISTER_F(ParquetRead, name)                                \
    ->Args({static_cast<int32_t>(datatype), data_size, 64, compression}) \
    ->Unit(benchmark::kMillisecond)                                      \
    ->UseManualTime();

PARQ_RD_BENCHMARK_DEFINE(int32, cudf::type_id::INT32, UNCOMPRESSED);
PARQ_RD_BENCHMARK_DEFINE(string, cudf::type_id::STRING, UNCOMPRESSED);
PARQ_RD_BENCHMARK_DEFINE(ddays, cudf::type_id::DURATION_DAYS, UNCOMPRESSED);
PARQ_RD_BENCHMARK_DEFINE(tdays, cudf::type_id::TIMESTAMP_DAYS, UNCOMPRESSED);