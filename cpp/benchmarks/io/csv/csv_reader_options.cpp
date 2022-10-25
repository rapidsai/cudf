/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <benchmarks/fixture/rmm_pool_raii.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/csv.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

constexpr size_t data_size = 256 << 20;

template <column_selection ColSelection, row_selection RowSelection>
void BM_csv_read_varying_options(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<ColSelection>, nvbench::enum_type<RowSelection>>)
{
  cudf::rmm_pool_raii rmm_pool;

  auto const data_types =
    dtypes_for_column_selection(get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                                   static_cast<int32_t>(data_type::FLOAT),
                                                   static_cast<int32_t>(data_type::DECIMAL),
                                                   static_cast<int32_t>(data_type::TIMESTAMP),
                                                   static_cast<int32_t>(data_type::DURATION),
                                                   static_cast<int32_t>(data_type::STRING)}),
                                ColSelection);
  auto const cols_to_read = select_column_indexes(data_types.size(), ColSelection);
  auto const num_chunks   = state.get_int64("num_chunks");

  auto const tbl  = create_random_table(data_types, table_size_bytes{data_size});
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  cudf::io::csv_writer_options options =
    cudf::io::csv_writer_options::builder(source_sink.make_sink_info(), view)
      .include_header(true)
      .line_terminator("\r\n");
  cudf::io::write_csv(options);

  cudf::io::csv_reader_options read_options =
    cudf::io::csv_reader_options::builder(source_sink.make_source_info())
      .use_cols_indexes(cols_to_read)
      .thousands('\'')
      .windowslinetermination(true)
      .comment('#')
      .prefix("BM_");

  size_t const chunk_size             = source_sink.size() / num_chunks;
  cudf::size_type const chunk_row_cnt = view.num_rows() / num_chunks;
  auto const mem_stats_logger         = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               try_drop_l3_cache();  // Drop L3 cache for accurate measurement

               timer.start();
               for (int32_t chunk = 0; chunk < num_chunks; ++chunk) {
                 // only read the header in the first chunk
                 read_options.set_header(chunk == 0 ? 0 : -1);

                 auto const is_last_chunk = chunk == (num_chunks - 1);
                 switch (RowSelection) {
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

                 cudf::io::read_csv(read_options);
               }
               timer.stop();
             });

  auto const elapsed_time   = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  auto const data_processed = data_size * cols_to_read.size() / view.num_columns();
  state.add_element_count(static_cast<double>(data_processed) / elapsed_time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

using col_selections = nvbench::enum_type_list<column_selection::ALL,
                                               column_selection::ALTERNATE,
                                               column_selection::FIRST_HALF,
                                               column_selection::SECOND_HALF>;

using row_selections = nvbench::
  enum_type_list<row_selection::BYTE_RANGE, row_selection::NROWS, row_selection::SKIPFOOTER>;

NVBENCH_BENCH_TYPES(BM_csv_read_varying_options,
                    NVBENCH_TYPE_AXES(col_selections, nvbench::enum_type_list<row_selection::ALL>))
  .set_name("csv_read_column_selection")
  .set_type_axes_names({"column_selection", "row_selection"})
  .set_min_samples(4)
  .add_int64_axis("num_chunks", {1});

NVBENCH_BENCH_TYPES(BM_csv_read_varying_options,
                    NVBENCH_TYPE_AXES(nvbench::enum_type_list<column_selection::ALL>,
                                      row_selections))
  .set_name("csv_read_row_selection")
  .set_type_axes_names({"column_selection", "row_selection"})
  .set_min_samples(4)
  .add_int64_axis("num_chunks", {1, 8});
