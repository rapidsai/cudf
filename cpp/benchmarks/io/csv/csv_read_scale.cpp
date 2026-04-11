/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/csv.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

constexpr cudf::size_type num_cols = 16;

// Use alphanumeric character range to avoid CSV special characters (comma, quote, hash)
// that can trigger quoting issues.
data_profile const scale_profile = data_profile_builder().string_char_range('0', 'z');  // ASCII 48-122

void BM_csv_read_scale(nvbench::state& state)
{
  auto const data_size_mb = state.get_int64("data_size_mb");
  size_t const data_size  = static_cast<size_t>(data_size_mb) << 20;

  // Mixed-type column set: INTEGRAL, FLOAT, TIMESTAMP, STRING cycled across 16 columns
  auto const data_types = cycle_dtypes(
    get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                       static_cast<int32_t>(data_type::FLOAT),
                       static_cast<int32_t>(data_type::TIMESTAMP),
                       static_cast<int32_t>(data_type::STRING)}),
    num_cols);

  auto const tbl =
    create_random_table(data_types, table_size_bytes{data_size}, scale_profile);
  auto const view = tbl->view();

  // Write to HOST_BUFFER to measure realistic host->device transfer path at large sizes
  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  cudf::io::csv_writer_options write_options =
    cudf::io::csv_writer_options::builder(source_sink.make_sink_info(), view).include_header(true);
  cudf::io::write_csv(write_options);

  // Extract column types from the source table to avoid type inference overhead
  std::vector<cudf::data_type> column_types;
  column_types.reserve(view.num_columns());
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    column_types.push_back(view.column(i).type());
  }

  cudf::io::csv_reader_options const read_options =
    cudf::io::csv_reader_options::builder(source_sink.make_source_info())
      .compression(cudf::io::compression_type::NONE)
      .dtypes(column_types);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_options.get_source().filepaths());

      timer.start();
      auto const result = cudf::io::read_csv(read_options);
      timer.stop();

      CUDF_EXPECTS(result.tbl->num_columns() == view.num_columns(), "Unexpected number of columns");
      CUDF_EXPECTS(result.tbl->num_rows() == view.num_rows(), "Unexpected number of rows");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

NVBENCH_BENCH(BM_csv_read_scale)
  .set_name("csv_read_scale")
  .set_min_samples(2)
  .add_int64_axis("data_size_mb", {256, 512, 1024, 2048, 4096});
