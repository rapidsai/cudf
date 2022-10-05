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

#include <cudf/io/csv.hpp>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

constexpr size_t data_size         = 256 << 20;
constexpr cudf::size_type num_cols = 64;

class CsvWrite : public cudf::benchmark {
};

void BM_csv_write_varying_inout(benchmark::State& state)
{
  auto const data_types = get_type_or_group(state.range(0));
  auto const sink_type  = static_cast<io_type>(state.range(1));

  auto const tbl =
    create_random_table(cycle_dtypes(data_types, num_cols), table_size_bytes{data_size});
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(sink_type);
  auto mem_stats_logger = cudf::memory_stats_logger();
  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf::io::csv_writer_options options =
      cudf::io::csv_writer_options::builder(source_sink.make_sink_info(), view);
    cudf::io::write_csv(options);
  }

  state.SetBytesProcessed(data_size * state.iterations());
  state.counters["peak_memory_usage"] = mem_stats_logger.peak_memory_usage();
  state.counters["encoded_file_size"] = source_sink.size();
}

void BM_csv_write_varying_options(benchmark::State& state)
{
  auto const na_per_len     = state.range(0);
  auto const rows_per_chunk = 1 << state.range(1);

  auto const data_types = get_type_or_group({int32_t(type_group_id::INTEGRAL),
                                             int32_t(type_group_id::FLOATING_POINT),
                                             int32_t(type_group_id::FIXED_POINT),
                                             int32_t(type_group_id::TIMESTAMP),
                                             int32_t(type_group_id::DURATION),
                                             int32_t(cudf::type_id::STRING)});

  auto const tbl  = create_random_table(data_types, table_size_bytes{data_size});
  auto const view = tbl->view();

  std::string const na_per(na_per_len, '#');
  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  auto mem_stats_logger = cudf::memory_stats_logger();
  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf::io::csv_writer_options options =
      cudf::io::csv_writer_options::builder(source_sink.make_sink_info(), view)
        .na_rep(na_per)
        .rows_per_chunk(rows_per_chunk);
    cudf::io::write_csv(options);
  }

  state.SetBytesProcessed(data_size * state.iterations());
  state.counters["peak_memory_usage"] = mem_stats_logger.peak_memory_usage();
  state.counters["encoded_file_size"] = source_sink.size();
}

#define CSV_WR_BM_INOUTS_DEFINE(name, type_or_group, sink_type)       \
  BENCHMARK_DEFINE_F(CsvWrite, name)                                  \
  (::benchmark::State & state) { BM_csv_write_varying_inout(state); } \
  BENCHMARK_REGISTER_F(CsvWrite, name)                                \
    ->Args({int32_t(type_or_group), sink_type})                       \
    ->Unit(benchmark::kMillisecond)                                   \
    ->UseManualTime();

WR_BENCHMARK_DEFINE_ALL_SINKS(CSV_WR_BM_INOUTS_DEFINE, integral, type_group_id::INTEGRAL);
WR_BENCHMARK_DEFINE_ALL_SINKS(CSV_WR_BM_INOUTS_DEFINE, floats, type_group_id::FLOATING_POINT);
WR_BENCHMARK_DEFINE_ALL_SINKS(CSV_WR_BM_INOUTS_DEFINE, decimal, type_group_id::FIXED_POINT);
WR_BENCHMARK_DEFINE_ALL_SINKS(CSV_WR_BM_INOUTS_DEFINE, timestamps, type_group_id::TIMESTAMP);
WR_BENCHMARK_DEFINE_ALL_SINKS(CSV_WR_BM_INOUTS_DEFINE, durations, type_group_id::DURATION);
WR_BENCHMARK_DEFINE_ALL_SINKS(CSV_WR_BM_INOUTS_DEFINE, string, cudf::type_id::STRING);

BENCHMARK_DEFINE_F(CsvWrite, writer_options)
(::benchmark::State& state) { BM_csv_write_varying_options(state); }
BENCHMARK_REGISTER_F(CsvWrite, writer_options)
  ->ArgsProduct({{0, 16}, {8, 10, 12, 14, 16, 18, 20}})
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();
