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

#include <chrono>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

constexpr size_t data_size         = 512 << 20;
constexpr cudf::size_type num_cols = 64;

namespace cudf_io = cudf::io;

class ParquetRead : public cudf::benchmark {
};

void PQ_read(benchmark::State& state)
{
  auto const data_types = get_type_or_group(state.range(0));
  cudf_io::compression_type const compression =
    state.range(1) ? cudf_io::compression_type::SNAPPY : cudf_io::compression_type::NONE;

  std::vector<char> out_buffer;
  out_buffer.reserve(data_size);
  data_profile table_data_profile;
  // table_data_profile.set_bool_probability(0.);
  auto t1        = std::chrono::high_resolution_clock::now();
  auto const tbl = create_random_table(data_types, num_cols, data_size, table_data_profile);
  auto t2        = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  std::cout << duration << std::endl;
  auto const view = tbl->view();

  cudf_io::write_parquet_args write_args{
    cudf_io::sink_info(&out_buffer), view, nullptr, compression};
  cudf_io::write_parquet(write_args);

  cudf_io::read_parquet_args read_args{cudf_io::source_info(out_buffer.data(), out_buffer.size())};

  for (auto _ : state) {
    cuda_event_timer const raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::read_parquet(read_args);
  }

  state.SetBytesProcessed(data_size * state.iterations());
}

#define PARQ_RD_BENCHMARK_DEFINE(name, type_or_group, compression) \
  BENCHMARK_DEFINE_F(ParquetRead, name##_##compression)            \
  (::benchmark::State & state) { PQ_read(state); }                 \
  BENCHMARK_REGISTER_F(ParquetRead, name##_##compression)          \
    ->Args({static_cast<int32_t>(type_or_group), compression})     \
    ->Unit(benchmark::kMillisecond)                                \
    ->UseManualTime();

PARQ_RD_BENCHMARK_DEFINE(integral, type_group_id::INTEGRAL, UNCOMPRESSED);
PARQ_RD_BENCHMARK_DEFINE(floats, type_group_id::FLOATING_POINT, UNCOMPRESSED);
PARQ_RD_BENCHMARK_DEFINE(timestamps, type_group_id::TIMESTAMP, UNCOMPRESSED);
PARQ_RD_BENCHMARK_DEFINE(string, cudf::type_id::STRING, UNCOMPRESSED);

// PARQ_RD_BENCHMARK_DEFINE(integral, type_group_id::INTEGRAL, USE_SNAPPY);
// PARQ_RD_BENCHMARK_DEFINE(floats, type_group_id::FLOATING_POINT, USE_SNAPPY);
// PARQ_RD_BENCHMARK_DEFINE(timestamps, type_group_id::TIMESTAMP, USE_SNAPPY);
// PARQ_RD_BENCHMARK_DEFINE(string, cudf::type_id::STRING, USE_SNAPPY);