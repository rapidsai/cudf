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

namespace cudf_io = cudf::io;

class CsvRead : public cudf::benchmark {
};

void BM_csv_read_varying_input(benchmark::State& state)
{
  auto const data_types  = get_type_or_group(state.range(0));
  auto const source_type = static_cast<io_type>(state.range(1));

  auto const tbl =
    create_random_table(cycle_dtypes(data_types, num_cols), table_size_bytes{data_size});
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(source_type);
  cudf_io::csv_writer_options options =
    cudf_io::csv_writer_options::builder(source_sink.make_sink_info(), view).include_header(true);
  cudf_io::write_csv(options);

  cudf_io::csv_reader_options const read_options =
    cudf_io::csv_reader_options::builder(source_sink.make_source_info());

  auto mem_stats_logger = cudf::memory_stats_logger();
  for (auto _ : state) {
    try_drop_l3_cache();
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::read_csv(read_options);
  }

  state.SetBytesProcessed(data_size * state.iterations());
  state.counters["peak_memory_usage"] = mem_stats_logger.peak_memory_usage();
  state.counters["encoded_file_size"] = source_sink.size();
}

void BM_csv_read_varying_options(benchmark::State& state)
{
  auto const col_sel    = static_cast<column_selection>(state.range(0));
  auto const row_sel    = static_cast<row_selection>(state.range(1));
  auto const num_chunks = state.range(2);

  auto const data_types =
    dtypes_for_column_selection(get_type_or_group({int32_t(type_group_id::INTEGRAL),
                                                   int32_t(type_group_id::FLOATING_POINT),
                                                   int32_t(type_group_id::FIXED_POINT),
                                                   int32_t(type_group_id::TIMESTAMP),
                                                   int32_t(type_group_id::DURATION),
                                                   int32_t(cudf::type_id::STRING)}),
                                col_sel);
  auto const cols_to_read = select_column_indexes(data_types.size(), col_sel);

  auto const tbl  = create_random_table(data_types, table_size_bytes{data_size});
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  cudf_io::csv_writer_options options =
    cudf_io::csv_writer_options::builder(source_sink.make_sink_info(), view)
      .include_header(true)
      .line_terminator("\r\n");
  cudf_io::write_csv(options);

  cudf_io::csv_reader_options read_options =
    cudf_io::csv_reader_options::builder(source_sink.make_source_info())
      .use_cols_indexes(cols_to_read)
      .thousands('\'')
      .windowslinetermination(true)
      .comment('#')
      .prefix("BM_");

  size_t const chunk_size             = source_sink.size() / num_chunks;
  cudf::size_type const chunk_row_cnt = view.num_rows() / num_chunks;
  auto mem_stats_logger               = cudf::memory_stats_logger();
  for (auto _ : state) {
    try_drop_l3_cache();
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    for (int32_t chunk = 0; chunk < num_chunks; ++chunk) {
      // only read the header in the first chunk
      read_options.set_header(chunk == 0 ? 0 : -1);

      auto const is_last_chunk = chunk == (num_chunks - 1);
      switch (row_sel) {
        case row_selection::ALL: break;
        case row_selection::BYTE_RANGE:
          read_options.set_byte_range_offset(chunk * chunk_size);
          read_options.set_byte_range_size(chunk_size);
          if (is_last_chunk) read_options.set_byte_range_size(0);
          break;
        case row_selection::NROWS:
          read_options.set_skiprows(chunk * chunk_row_cnt);
          read_options.set_nrows(chunk_row_cnt);
          if (is_last_chunk) read_options.set_nrows(-1);
          break;
        case row_selection::SKIPFOOTER:
          read_options.set_skiprows(chunk * chunk_row_cnt);
          read_options.set_skipfooter(view.num_rows() - (chunk + 1) * chunk_row_cnt);
          if (is_last_chunk) read_options.set_skipfooter(0);
          break;
        default: CUDF_FAIL("Unsupported row selection method");
      }

      cudf_io::read_csv(read_options);
    }
  }

  auto const data_processed = data_size * cols_to_read.size() / view.num_columns();
  state.SetBytesProcessed(data_processed * state.iterations());
  state.counters["peak_memory_usage"] = mem_stats_logger.peak_memory_usage();
  state.counters["encoded_file_size"] = source_sink.size();
}

#define CSV_RD_BM_INPUTS_DEFINE(name, type_or_group, src_type)       \
  BENCHMARK_DEFINE_F(CsvRead, name)                                  \
  (::benchmark::State & state) { BM_csv_read_varying_input(state); } \
  BENCHMARK_REGISTER_F(CsvRead, name)                                \
    ->Args({int32_t(type_or_group), src_type})                       \
    ->Unit(benchmark::kMillisecond)                                  \
    ->UseManualTime();

RD_BENCHMARK_DEFINE_ALL_SOURCES(CSV_RD_BM_INPUTS_DEFINE, integral, type_group_id::INTEGRAL);
RD_BENCHMARK_DEFINE_ALL_SOURCES(CSV_RD_BM_INPUTS_DEFINE, floats, type_group_id::FLOATING_POINT);
RD_BENCHMARK_DEFINE_ALL_SOURCES(CSV_RD_BM_INPUTS_DEFINE, decimal, type_group_id::FIXED_POINT);
RD_BENCHMARK_DEFINE_ALL_SOURCES(CSV_RD_BM_INPUTS_DEFINE, timestamps, type_group_id::TIMESTAMP);
RD_BENCHMARK_DEFINE_ALL_SOURCES(CSV_RD_BM_INPUTS_DEFINE, durations, type_group_id::DURATION);
RD_BENCHMARK_DEFINE_ALL_SOURCES(CSV_RD_BM_INPUTS_DEFINE, string, cudf::type_id::STRING);

BENCHMARK_DEFINE_F(CsvRead, column_selection)
(::benchmark::State& state) { BM_csv_read_varying_options(state); }
BENCHMARK_REGISTER_F(CsvRead, column_selection)
  ->ArgsProduct({{int32_t(column_selection::ALL),
                  int32_t(column_selection::ALTERNATE),
                  int32_t(column_selection::FIRST_HALF),
                  int32_t(column_selection::SECOND_HALF)},
                 {int32_t(row_selection::ALL)},
                 {1}})
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();

BENCHMARK_DEFINE_F(CsvRead, row_selection)
(::benchmark::State& state) { BM_csv_read_varying_options(state); }
BENCHMARK_REGISTER_F(CsvRead, row_selection)
  ->ArgsProduct({{int32_t(column_selection::ALL)},
                 {int32_t(row_selection::BYTE_RANGE),
                  int32_t(row_selection::NROWS),
                  int32_t(row_selection::SKIPFOOTER)},
                 {1, 8}})
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();
