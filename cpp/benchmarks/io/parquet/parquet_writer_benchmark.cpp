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
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/io/parquet.hpp>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

constexpr size_t data_size         = 512 << 20;
constexpr cudf::size_type num_cols = 64;

namespace cudf_io = cudf::io;

class ParquetWrite : public cudf::benchmark {
};

void PQ_write(benchmark::State& state)
{
  auto const data_types             = get_type_or_group(state.range(0));
  cudf::size_type const cardinality = state.range(1);
  cudf::size_type const run_length  = state.range(2);
  cudf_io::compression_type const compression =
    state.range(3) ? cudf_io::compression_type::SNAPPY : cudf_io::compression_type::NONE;

  data_profile table_data_profile;
  table_data_profile.set_cardinality(cardinality);
  table_data_profile.set_avg_run_length(run_length);
  auto const tbl =
    create_random_table(data_types, num_cols, table_size_bytes{data_size}, table_data_profile);
  auto const view = tbl->view();

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::parquet_writer_options opts =
      cudf_io::parquet_writer_options::builder(cudf_io::sink_info(), view).compression(compression);
    cudf_io::write_parquet(opts);
  }

  state.SetBytesProcessed(data_size * state.iterations());
}

// TODO: replace with ArgsProduct once available
#define PARQ_WR_BENCHMARK_DEFINE(name, type_or_group) \
  BENCHMARK_DEFINE_F(ParquetWrite, name)              \
  (::benchmark::State & state) { PQ_write(state); }   \
  BENCHMARK_REGISTER_F(ParquetWrite, name)            \
    ->Args({int32_t(type_or_group), 0, 1, false})     \
    ->Args({int32_t(type_or_group), 0, 1, true})      \
    ->Args({int32_t(type_or_group), 0, 32, false})    \
    ->Args({int32_t(type_or_group), 0, 32, true})     \
    ->Args({int32_t(type_or_group), 1000, 1, false})  \
    ->Args({int32_t(type_or_group), 1000, 1, true})   \
    ->Args({int32_t(type_or_group), 1000, 32, false}) \
    ->Args({int32_t(type_or_group), 1000, 32, true})  \
    ->Unit(benchmark::kMillisecond)                   \
    ->UseManualTime();

PARQ_WR_BENCHMARK_DEFINE(integral, type_group_id::INTEGRAL);
PARQ_WR_BENCHMARK_DEFINE(floats, type_group_id::FLOATING_POINT);
PARQ_WR_BENCHMARK_DEFINE(timestamps, type_group_id::TIMESTAMP);
PARQ_WR_BENCHMARK_DEFINE(string, cudf::type_id::STRING);
