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

#include <cudf/io/orc.hpp>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

constexpr int64_t data_size        = 512 << 20;
constexpr cudf::size_type num_cols = 64;

namespace cudf_io = cudf::io;

class OrcRead : public cudf::benchmark {
};

void BM_orc_read_varying_input(benchmark::State& state)
{
  auto const data_types             = get_type_or_group(state.range(0));
  cudf::size_type const cardinality = state.range(1);
  cudf::size_type const run_length  = state.range(2);
  cudf_io::compression_type const compression =
    state.range(3) ? cudf_io::compression_type::SNAPPY : cudf_io::compression_type::NONE;
  auto const source_type = static_cast<io_type>(state.range(4));

  auto const tbl =
    create_random_table(cycle_dtypes(data_types, num_cols),
                        table_size_bytes{data_size},
                        data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(source_type);
  cudf_io::orc_writer_options opts =
    cudf_io::orc_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(compression);
  cudf_io::write_orc(opts);

  cudf_io::orc_reader_options read_opts =
    cudf_io::orc_reader_options::builder(source_sink.make_source_info());

  auto mem_stats_logger = cudf::memory_stats_logger();
  for (auto _ : state) {
    try_drop_l3_cache();
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::read_orc(read_opts);
  }

  state.SetBytesProcessed(data_size * state.iterations());
  state.counters["peak_memory_usage"] = mem_stats_logger.peak_memory_usage();
  state.counters["encoded_file_size"] = source_sink.size();
}

std::vector<std::string> get_col_names(cudf_io::source_info const& source)
{
  cudf_io::orc_reader_options const read_options =
    cudf_io::orc_reader_options::builder(source).num_rows(1);
  return cudf_io::read_orc(read_options).metadata.column_names;
}

void BM_orc_read_varying_options(benchmark::State& state)
{
  auto state_idx        = 0;
  auto const col_sel    = static_cast<column_selection>(state.range(state_idx++));
  auto const row_sel    = static_cast<row_selection>(state.range(state_idx++));
  auto const num_chunks = state.range(state_idx++);

  auto const flags         = state.range(state_idx++);
  auto const use_index     = (flags & 1) != 0;
  auto const use_np_dtypes = (flags & 2) != 0;
  auto const ts_type       = cudf::data_type{static_cast<cudf::type_id>(state.range(state_idx++))};

  // skip_rows is not supported on nested types
  auto const data_types =
    dtypes_for_column_selection(get_type_or_group({int32_t(type_group_id::INTEGRAL_SIGNED),
                                                   int32_t(type_group_id::FLOATING_POINT),
                                                   int32_t(type_group_id::FIXED_POINT),
                                                   int32_t(type_group_id::TIMESTAMP),
                                                   int32_t(cudf::type_id::STRING)}),
                                col_sel);
  auto const tbl  = create_random_table(data_types, table_size_bytes{data_size});
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  cudf_io::orc_writer_options options =
    cudf_io::orc_writer_options::builder(source_sink.make_sink_info(), view);
  cudf_io::write_orc(options);

  auto const cols_to_read =
    select_column_names(get_col_names(source_sink.make_source_info()), col_sel);
  cudf_io::orc_reader_options read_options =
    cudf_io::orc_reader_options::builder(source_sink.make_source_info())
      .columns(cols_to_read)
      .use_index(use_index)
      .use_np_dtypes(use_np_dtypes)
      .timestamp_type(ts_type);

  auto const num_stripes              = data_size / (64 << 20);
  cudf::size_type const chunk_row_cnt = view.num_rows() / num_chunks;
  auto mem_stats_logger               = cudf::memory_stats_logger();
  for (auto _ : state) {
    try_drop_l3_cache();
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0

    cudf::size_type rows_read = 0;
    for (int32_t chunk = 0; chunk < num_chunks; ++chunk) {
      auto const is_last_chunk = chunk == (num_chunks - 1);
      switch (row_sel) {
        case row_selection::ALL: break;
        case row_selection::STRIPES: {
          auto stripes_to_read = segments_in_chunk(num_stripes, num_chunks, chunk);
          if (is_last_chunk) {
            // Need to assume that an additional "overflow" stripe is present
            stripes_to_read.push_back(num_stripes);
          }
          read_options.set_stripes({stripes_to_read});
        } break;
        case row_selection::NROWS:
          read_options.set_skip_rows(chunk * chunk_row_cnt);
          read_options.set_num_rows(chunk_row_cnt);
          if (is_last_chunk) read_options.set_num_rows(-1);
          break;
        default: CUDF_FAIL("Unsupported row selection method");
      }

      rows_read += cudf_io::read_orc(read_options).tbl->num_rows();
    }

    CUDF_EXPECTS(rows_read == view.num_rows(), "Benchmark did not read the entire table");
  }

  auto const data_processed = data_size * cols_to_read.size() / view.num_columns();
  state.SetBytesProcessed(data_processed * state.iterations());
  state.counters["peak_memory_usage"] = mem_stats_logger.peak_memory_usage();
  state.counters["encoded_file_size"] = source_sink.size();
}

#define ORC_RD_BM_INPUTS_DEFINE(name, type_or_group, src_type)                               \
  BENCHMARK_DEFINE_F(OrcRead, name)                                                          \
  (::benchmark::State & state) { BM_orc_read_varying_input(state); }                         \
  BENCHMARK_REGISTER_F(OrcRead, name)                                                        \
    ->ArgsProduct({{int32_t(type_or_group)}, {0, 1000}, {1, 32}, {true, false}, {src_type}}) \
    ->Unit(benchmark::kMillisecond)                                                          \
    ->UseManualTime();

RD_BENCHMARK_DEFINE_ALL_SOURCES(ORC_RD_BM_INPUTS_DEFINE, integral, type_group_id::INTEGRAL_SIGNED);
RD_BENCHMARK_DEFINE_ALL_SOURCES(ORC_RD_BM_INPUTS_DEFINE, floats, type_group_id::FLOATING_POINT);
RD_BENCHMARK_DEFINE_ALL_SOURCES(ORC_RD_BM_INPUTS_DEFINE, decimal, type_group_id::FIXED_POINT);
RD_BENCHMARK_DEFINE_ALL_SOURCES(ORC_RD_BM_INPUTS_DEFINE, timestamps, type_group_id::TIMESTAMP);
RD_BENCHMARK_DEFINE_ALL_SOURCES(ORC_RD_BM_INPUTS_DEFINE, string, cudf::type_id::STRING);
RD_BENCHMARK_DEFINE_ALL_SOURCES(ORC_RD_BM_INPUTS_DEFINE, list, cudf::type_id::LIST);
RD_BENCHMARK_DEFINE_ALL_SOURCES(ORC_RD_BM_INPUTS_DEFINE, struct, cudf::type_id::STRUCT);

BENCHMARK_DEFINE_F(OrcRead, column_selection)
(::benchmark::State& state) { BM_orc_read_varying_options(state); }
BENCHMARK_REGISTER_F(OrcRead, column_selection)
  ->ArgsProduct({{int32_t(column_selection::ALL),
                  int32_t(column_selection::ALTERNATE),
                  int32_t(column_selection::FIRST_HALF),
                  int32_t(column_selection::SECOND_HALF)},
                 {int32_t(row_selection::ALL)},
                 {1},
                 {0b11},  // defaults
                 {int32_t(cudf::type_id::EMPTY)}})
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();

// Need an API to get the number of stripes to enable row_selection::STRIPES here
BENCHMARK_DEFINE_F(OrcRead, row_selection)
(::benchmark::State& state) { BM_orc_read_varying_options(state); }
BENCHMARK_REGISTER_F(OrcRead, row_selection)
  ->ArgsProduct({{int32_t(column_selection::ALL)},
                 {int32_t(row_selection::NROWS)},
                 {1, 8},
                 {0b11},  // defaults
                 {int32_t(cudf::type_id::EMPTY)}})
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();

BENCHMARK_DEFINE_F(OrcRead, misc_options)
(::benchmark::State& state) { BM_orc_read_varying_options(state); }
BENCHMARK_REGISTER_F(OrcRead, misc_options)
  ->ArgsProduct({{int32_t(column_selection::ALL)},
                 {int32_t(row_selection::NROWS)},
                 {1},
                 {0b11, 0b10, 0b01},  // `true` is default for each boolean parameter here
                 {int32_t(cudf::type_id::EMPTY), int32_t(cudf::type_id::TIMESTAMP_NANOSECONDS)}})
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();
