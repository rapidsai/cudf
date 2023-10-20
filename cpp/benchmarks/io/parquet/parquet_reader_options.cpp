/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr std::size_t data_size = 512 << 20;
// The number of separate read calls to use when reading files in multiple chunks
// Each call reads roughly equal amounts of data
constexpr int32_t chunked_read_num_chunks = 4;

std::vector<std::string> get_top_level_col_names(cudf::io::source_info const& source)
{
  auto const top_lvl_cols = cudf::io::read_parquet_metadata(source).schema().root().children();
  std::vector<std::string> col_names;
  std::transform(top_lvl_cols.cbegin(),
                 top_lvl_cols.cend(),
                 std::back_inserter(col_names),
                 [](auto const& col_meta) { return col_meta.name(); });

  return col_names;
}

template <column_selection ColSelection,
          row_selection RowSelection,
          converts_strings ConvertsStrings,
          uses_pandas_metadata UsesPandasMetadata,
          cudf::type_id Timestamp>
void BM_parquet_read_options(nvbench::state& state,
                             nvbench::type_list<nvbench::enum_type<ColSelection>,
                                                nvbench::enum_type<RowSelection>,
                                                nvbench::enum_type<ConvertsStrings>,
                                                nvbench::enum_type<UsesPandasMetadata>,
                                                nvbench::enum_type<Timestamp>>)
{
  auto const num_chunks = RowSelection == row_selection::ALL ? 1 : chunked_read_num_chunks;

  auto constexpr str_to_categories = ConvertsStrings == converts_strings::YES;
  auto constexpr uses_pd_metadata  = UsesPandasMetadata == uses_pandas_metadata::YES;

  auto const ts_type = cudf::data_type{Timestamp};

  auto const data_types =
    dtypes_for_column_selection(get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                                   static_cast<int32_t>(data_type::FLOAT),
                                                   static_cast<int32_t>(data_type::DECIMAL),
                                                   static_cast<int32_t>(data_type::TIMESTAMP),
                                                   static_cast<int32_t>(data_type::DURATION),
                                                   static_cast<int32_t>(data_type::STRING),
                                                   static_cast<int32_t>(data_type::LIST),
                                                   static_cast<int32_t>(data_type::STRUCT)}),
                                ColSelection);
  auto const tbl  = create_random_table(data_types, table_size_bytes{data_size});
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  cudf::io::parquet_writer_options options =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view);
  cudf::io::write_parquet(options);

  auto const cols_to_read =
    select_column_names(get_top_level_col_names(source_sink.make_source_info()), ColSelection);
  cudf::io::parquet_reader_options read_options =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info())
      .columns(cols_to_read)
      .convert_strings_to_categories(str_to_categories)
      .use_pandas_metadata(uses_pd_metadata)
      .timestamp_type(ts_type);

  auto const num_row_groups = read_parquet_metadata(source_sink.make_source_info()).num_rowgroups();
  auto const chunk_row_cnt  = cudf::util::div_rounding_up_unsafe(view.num_rows(), num_chunks);

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      try_drop_l3_cache();

      timer.start();
      cudf::size_type rows_read = 0;
      for (int32_t chunk = 0; chunk < num_chunks; ++chunk) {
        switch (RowSelection) {
          case row_selection::ALL: break;
          case row_selection::ROW_GROUPS: {
            read_options.set_row_groups({segments_in_chunk(num_row_groups, num_chunks, chunk)});
          } break;
          case row_selection::NROWS:
            read_options.set_skip_rows(chunk * chunk_row_cnt);
            read_options.set_num_rows(chunk_row_cnt);
            break;
          default: CUDF_FAIL("Unsupported row selection method");
        }

        rows_read += cudf::io::read_parquet(read_options).tbl->num_rows();
      }

      CUDF_EXPECTS(rows_read == view.num_rows(), "Benchmark did not read the entire table");
      timer.stop();
    });

  auto const elapsed_time   = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  auto const data_processed = data_size * cols_to_read.size() / view.num_columns();
  state.add_element_count(static_cast<double>(data_processed) / elapsed_time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

using row_selections =
  nvbench::enum_type_list<row_selection::ALL, row_selection::NROWS, row_selection::ROW_GROUPS>;
NVBENCH_BENCH_TYPES(BM_parquet_read_options,
                    NVBENCH_TYPE_AXES(nvbench::enum_type_list<column_selection::ALL>,
                                      row_selections,
                                      nvbench::enum_type_list<converts_strings::YES>,
                                      nvbench::enum_type_list<uses_pandas_metadata::YES>,
                                      nvbench::enum_type_list<cudf::type_id::EMPTY>))
  .set_name("parquet_read_row_selection")
  .set_type_axes_names({"column_selection",
                        "row_selection",
                        "str_to_categories",
                        "uses_pandas_metadata",
                        "timestamp_type"})
  .set_min_samples(4);

using col_selections = nvbench::enum_type_list<column_selection::ALL,
                                               column_selection::ALTERNATE,
                                               column_selection::FIRST_HALF,
                                               column_selection::SECOND_HALF>;
NVBENCH_BENCH_TYPES(BM_parquet_read_options,
                    NVBENCH_TYPE_AXES(col_selections,
                                      nvbench::enum_type_list<row_selection::ALL>,
                                      nvbench::enum_type_list<converts_strings::YES>,
                                      nvbench::enum_type_list<uses_pandas_metadata::YES>,
                                      nvbench::enum_type_list<cudf::type_id::EMPTY>))
  .set_name("parquet_read_column_selection")
  .set_type_axes_names({"column_selection",
                        "row_selection",
                        "str_to_categories",
                        "uses_pandas_metadata",
                        "timestamp_type"})
  .set_min_samples(4);

NVBENCH_BENCH_TYPES(
  BM_parquet_read_options,
  NVBENCH_TYPE_AXES(nvbench::enum_type_list<column_selection::ALL>,
                    nvbench::enum_type_list<row_selection::ALL>,
                    nvbench::enum_type_list<converts_strings::YES, converts_strings::NO>,
                    nvbench::enum_type_list<uses_pandas_metadata::YES, uses_pandas_metadata::NO>,
                    nvbench::enum_type_list<cudf::type_id::EMPTY>))
  .set_name("parquet_read_misc_options")
  .set_type_axes_names({"column_selection",
                        "row_selection",
                        "str_to_categories",
                        "uses_pandas_metadata",
                        "timestamp_type"})
  .set_min_samples(4);
