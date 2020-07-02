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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <benchmarks/common/generate_benchmark_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/io/cuio_benchmarks_common.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/io/functions.hpp>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

constexpr int64_t data_size = 1 << 30;  // 1 GB

namespace cudf_io = cudf::io;

class ParquetWrite : public cudf::benchmark {
};
class ParquetWriteChunked : public cudf::benchmark {
};

void PQ_write(benchmark::State& state)
{
  int64_t total_desired_bytes = state.range(0);
  cudf::size_type num_cols    = state.range(1);

  cudf::size_type el_size = 4;
  int64_t num_rows        = total_desired_bytes / (num_cols * el_size);

  auto tbl              = create_random_table<int32_t>(num_cols, num_rows, true);
  cudf::table_view view = tbl->view();

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::write_parquet_args args{cudf_io::sink_info(), view};
    cudf_io::write_parquet(args);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0));
}

void PQ_write_chunked(benchmark::State& state)
{
  int64_t total_desired_bytes = state.range(0);
  cudf::size_type num_cols    = state.range(1);
  cudf::size_type num_tables  = state.range(2);

  cudf::size_type el_size = 4;
  int64_t num_rows        = (total_desired_bytes / (num_cols * el_size)) / num_tables;

  std::vector<std::unique_ptr<cudf::table>> tables;
  for (cudf::size_type idx = 0; idx < num_tables; idx++) {
    tables.push_back(create_random_table<int32_t>(num_cols, num_rows, true));
  }

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::write_parquet_chunked_args args{cudf_io::sink_info()};

    auto writer_state = cudf_io::write_parquet_chunked_begin(args);
    std::for_each(
      tables.begin(), tables.end(), [&writer_state](std::unique_ptr<cudf::table> const& tbl) {
        cudf_io::write_parquet_chunked(*tbl, writer_state);
      });
    cudf_io::write_parquet_chunked_end(writer_state);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0));
}

#define PWBM_BENCHMARK_DEFINE(name, size, num_columns)                                    \
  BENCHMARK_DEFINE_F(ParquetWrite, name)(::benchmark::State & state) { PQ_write(state); } \
  BENCHMARK_REGISTER_F(ParquetWrite, name)                                                \
    ->Args({size, num_columns})                                                           \
    ->Unit(benchmark::kMillisecond)                                                       \
    ->UseManualTime()

PWBM_BENCHMARK_DEFINE(3Gb8Cols, data_size, 8);
PWBM_BENCHMARK_DEFINE(3Gb1024Cols, data_size, 1024);

#define PWCBM_BENCHMARK_DEFINE(name, size, num_columns, num_chunks)         \
  BENCHMARK_DEFINE_F(ParquetWriteChunked, name)(::benchmark::State & state) \
  {                                                                         \
    PQ_write_chunked(state);                                                \
  }                                                                         \
  BENCHMARK_REGISTER_F(ParquetWriteChunked, name)                           \
    ->Args({size, num_columns, num_chunks})                                 \
    ->Unit(benchmark::kMillisecond)                                         \
    ->UseManualTime()                                                       \
    ->Iterations(4)

PWCBM_BENCHMARK_DEFINE(3Gb8Cols64Chunks, data_size, 8, 8);
PWCBM_BENCHMARK_DEFINE(3Gb1024Cols64Chunks, data_size, 1024, 8);

PWCBM_BENCHMARK_DEFINE(3Gb8Cols128Chunks, data_size, 8, 64);
PWCBM_BENCHMARK_DEFINE(3Gb1024Cols128Chunks, data_size, 1024, 64);
