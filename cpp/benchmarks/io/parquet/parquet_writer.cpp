/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/io/parquet.hpp>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

constexpr size_t data_size         = 512 << 20;
constexpr cudf::size_type num_cols = 64;

namespace cudf_io = cudf::io;

class ParquetWrite : public cudf::benchmark {
};

void BM_parq_write_varying_inout(benchmark::State& state)
{
  auto const data_types             = get_type_or_group(state.range(0));
  cudf::size_type const cardinality = state.range(1);
  cudf::size_type const run_length  = state.range(2);
  cudf_io::compression_type const compression =
    state.range(3) ? cudf_io::compression_type::SNAPPY : cudf_io::compression_type::NONE;
  auto const sink_type = static_cast<io_type>(state.range(4));

  auto const tbl =
    create_random_table(cycle_dtypes(data_types, num_cols),
                        table_size_bytes{data_size},
                        data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(sink_type);
  auto mem_stats_logger = cudf::memory_stats_logger();
  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::parquet_writer_options opts =
      cudf_io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(compression);
    cudf_io::write_parquet(opts);
  }

  state.SetBytesProcessed(data_size * state.iterations());
  state.counters["peak_memory_usage"] = mem_stats_logger.peak_memory_usage();
  state.counters["encoded_file_size"] = source_sink.size();
}

void BM_parq_write_varying_options(benchmark::State& state)
{
  auto const compression  = static_cast<cudf::io::compression_type>(state.range(0));
  auto const enable_stats = static_cast<cudf::io::statistics_freq>(state.range(1));
  auto const file_path    = state.range(2) != 0 ? "unused_path.parquet" : "";

  auto const data_types = get_type_or_group({int32_t(type_group_id::INTEGRAL_SIGNED),
                                             int32_t(type_group_id::FLOATING_POINT),
                                             int32_t(type_group_id::FIXED_POINT),
                                             int32_t(type_group_id::TIMESTAMP),
                                             int32_t(type_group_id::DURATION),
                                             int32_t(cudf::type_id::STRING),
                                             int32_t(cudf::type_id::LIST)});

  auto const tbl  = create_random_table(data_types, table_size_bytes{data_size});
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(io_type::FILEPATH);
  auto mem_stats_logger = cudf::memory_stats_logger();
  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::parquet_writer_options const options =
      cudf_io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(compression)
        .stats_level(enable_stats)
        .column_chunks_file_paths({file_path});
    cudf_io::write_parquet(options);
  }

  state.SetBytesProcessed(data_size * state.iterations());
  state.counters["peak_memory_usage"] = mem_stats_logger.peak_memory_usage();
  state.counters["encoded_file_size"] = source_sink.size();
}

#define PARQ_WR_BM_INOUTS_DEFINE(name, type_or_group, sink_type)                              \
  BENCHMARK_DEFINE_F(ParquetWrite, name)                                                      \
  (::benchmark::State & state) { BM_parq_write_varying_inout(state); }                        \
  BENCHMARK_REGISTER_F(ParquetWrite, name)                                                    \
    ->ArgsProduct({{int32_t(type_or_group)}, {0, 1000}, {1, 32}, {true, false}, {sink_type}}) \
    ->Unit(benchmark::kMillisecond)                                                           \
    ->UseManualTime();

WR_BENCHMARK_DEFINE_ALL_SINKS(PARQ_WR_BM_INOUTS_DEFINE, integral, type_group_id::INTEGRAL);
WR_BENCHMARK_DEFINE_ALL_SINKS(PARQ_WR_BM_INOUTS_DEFINE, floats, type_group_id::FLOATING_POINT);
WR_BENCHMARK_DEFINE_ALL_SINKS(PARQ_WR_BM_INOUTS_DEFINE, decimal, type_group_id::FIXED_POINT);
WR_BENCHMARK_DEFINE_ALL_SINKS(PARQ_WR_BM_INOUTS_DEFINE, timestamps, type_group_id::TIMESTAMP);
WR_BENCHMARK_DEFINE_ALL_SINKS(PARQ_WR_BM_INOUTS_DEFINE, durations, type_group_id::DURATION);
WR_BENCHMARK_DEFINE_ALL_SINKS(PARQ_WR_BM_INOUTS_DEFINE, string, cudf::type_id::STRING);
WR_BENCHMARK_DEFINE_ALL_SINKS(PARQ_WR_BM_INOUTS_DEFINE, list, cudf::type_id::LIST);
WR_BENCHMARK_DEFINE_ALL_SINKS(PARQ_WR_BM_INOUTS_DEFINE, struct, cudf::type_id::STRUCT);

BENCHMARK_DEFINE_F(ParquetWrite, writer_options)
(::benchmark::State& state) { BM_parq_write_varying_options(state); }
BENCHMARK_REGISTER_F(ParquetWrite, writer_options)
  ->ArgsProduct({{int32_t(cudf::io::compression_type::NONE),
                  int32_t(cudf::io::compression_type::SNAPPY)},
                 {int32_t(cudf::io::statistics_freq::STATISTICS_NONE),
                  int32_t(cudf::io::statistics_freq::STATISTICS_ROWGROUP),
                  int32_t(cudf::io::statistics_freq::STATISTICS_PAGE)},
                 {false, true}})
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();
