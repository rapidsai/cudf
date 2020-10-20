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
#include <benchmarks/io/cuio_benchmark_common.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/io/parquet.hpp>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

constexpr size_t data_size         = 512 << 20;
constexpr cudf::size_type num_cols = 64;

namespace cudf_io = cudf::io;

class ParquetRead : public cudf::benchmark {
};

void PQ_read(benchmark::State& state)
{
  auto const data_types             = get_type_or_group(state.range(0));
  cudf::size_type const cardinality = state.range(1);
  cudf::size_type const run_length  = state.range(2);
  cudf_io::compression_type const compression =
    state.range(3) ? cudf_io::compression_type::SNAPPY : cudf_io::compression_type::NONE;
  io_type const source_type = static_cast<io_type>(state.range(4));

  data_profile table_data_profile;
  table_data_profile.set_cardinality(cardinality);
  table_data_profile.set_avg_run_length(run_length);
  auto const tbl =
    create_random_table(data_types, num_cols, table_size_bytes{data_size}, table_data_profile);
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(source_type);
  cudf_io::parquet_writer_options write_opts =
    cudf_io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(compression);
  cudf_io::write_parquet(write_opts);

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(source_sink.make_source_info());

  for (auto _ : state) {
    cuda_event_timer const raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::read_parquet(read_opts);
  }

  state.SetBytesProcessed(data_size * state.iterations());
}

#define PARQ_RD_BENCHMARK_DEFINE(name, type_or_group, src_type)                              \
  BENCHMARK_DEFINE_F(ParquetRead, name)                                                      \
  (::benchmark::State & state) { PQ_read(state); }                                           \
  BENCHMARK_REGISTER_F(ParquetRead, name)                                                    \
    ->ArgsProduct({{int32_t(type_or_group)}, {0, 1000}, {1, 32}, {true, false}, {src_type}}) \
    ->Unit(benchmark::kMillisecond)                                                          \
    ->UseManualTime();

RD_BENCHMARK_DEFINE_ALL_SOURCES(PARQ_RD_BENCHMARK_DEFINE, integral, type_group_id::INTEGRAL);
RD_BENCHMARK_DEFINE_ALL_SOURCES(PARQ_RD_BENCHMARK_DEFINE, floats, type_group_id::FLOATING_POINT);
RD_BENCHMARK_DEFINE_ALL_SOURCES(PARQ_RD_BENCHMARK_DEFINE, timestamps, type_group_id::TIMESTAMP);
RD_BENCHMARK_DEFINE_ALL_SOURCES(PARQ_RD_BENCHMARK_DEFINE, string, cudf::type_id::STRING);
RD_BENCHMARK_DEFINE_ALL_SOURCES(PARQ_RD_BENCHMARK_DEFINE, list, cudf::type_id::LIST);
