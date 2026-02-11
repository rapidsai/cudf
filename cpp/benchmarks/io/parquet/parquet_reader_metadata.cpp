/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// Common mixed dtypes used by all benchmarks in this file
auto const mixed_dtypes = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                             static_cast<int32_t>(data_type::FLOAT),
                                             static_cast<int32_t>(data_type::STRING),
                                             static_cast<int32_t>(data_type::LIST),
                                             static_cast<int32_t>(data_type::STRUCT)});

constexpr cudf::size_type rows_per_row_group = 2000;

namespace {

// Helper to generate and write parquet data
auto write_file_data(cudf::size_type num_cols,
                     cudf::size_type num_row_groups,
                     io_type source_type,
                     bool write_page_index)
{
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows = rows_per_row_group * num_row_groups;

  auto const tbl  = create_random_table(cycle_dtypes(mixed_dtypes, num_cols),
                                       row_count{num_rows},
                                       data_profile_builder().cardinality(0).avg_run_length(1));
  auto const view = tbl->view();

  auto const stats_level = write_page_index ? cudf::io::statistics_freq::STATISTICS_COLUMN
                                            : cudf::io::statistics_freq::STATISTICS_ROWGROUP;
  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(cudf::io::compression_type::NONE)
      .row_group_size_rows(rows_per_row_group)
      .stats_level(stats_level);

  cudf::io::write_parquet(write_opts, cudf::get_default_stream());

  return source_sink;
}

}  // namespace

// Benchmark to measure parquet footer read time
void BM_parquet_read_footer(nvbench::state& state)
{
  auto const num_cols         = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const num_row_groups   = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const source_type      = retrieve_io_type_enum(state.get_string("io_type"));
  auto const write_page_index = state.get_int64("page_index") != 0;

  auto source_sink = write_file_data(num_cols, num_row_groups, source_type, write_page_index);

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      try_drop_l3_cache();
      auto sources = cudf::io::make_datasources(source_sink.make_source_info());

      timer.start();
      auto const metadatas = cudf::io::read_parquet_footers(sources);
      timer.stop();

      // Validate metadata
      CUDF_EXPECTS(std::cmp_equal(metadatas.size(), 1), "Expected one metadata object");
      CUDF_EXPECTS(std::cmp_equal(metadatas.front().row_groups.size(), num_row_groups),
                   "Unexpected number of row groups in metadata");
      // Using >= here as we have struct columns in the input
      CUDF_EXPECTS(
        std::cmp_greater_equal(metadatas.front().row_groups.front().columns.size(), num_cols),
        "Unexpected number of columns in metadata");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_cols * num_row_groups) / time,
                          "metadata_read_per_secon");
}

// Benchmark to measure chunked parquet reader construction time
void BM_parquet_reader_construction(nvbench::state& state)
{
  auto const num_cols         = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const num_row_groups   = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const source_type      = retrieve_io_type_enum(state.get_string("io_type"));
  auto const write_page_index = state.get_int64("page_index") != 0;

  auto source_sink = write_file_data(num_cols, num_row_groups, source_type, write_page_index);

  auto constexpr chunk_read_limit = 0;
  auto constexpr pass_read_limit  = 0;

  auto const read_opts = cudf::io::parquet_reader_options::builder(source_sink.make_source_info())
                           .use_arrow_schema(false)
                           .build();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      try_drop_l3_cache();
      timer.start();
      auto reader = cudf::io::chunked_parquet_reader(chunk_read_limit, pass_read_limit, read_opts);
      timer.stop();

      // Validate
      CUDF_EXPECTS(reader.has_next(), "Expected reader to have data");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_cols) / time, "columns_per_second");
}

// Benchmark to measure parquet column selection time
void BM_parquet_column_selection(nvbench::state& state)
{
  auto const num_cols    = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));

  cuio_source_sink_pair source_sink(source_type);

  // Create a table with minimal rows (1 row is enough to create valid parquet)
  constexpr cudf::size_type num_rows = 1;
  auto const tbl                     = create_random_table(cycle_dtypes(mixed_dtypes, num_cols),
                                       row_count{num_rows},
                                       data_profile_builder().cardinality(0).avg_run_length(1));
  auto const view                    = tbl->view();

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(write_opts);

  auto constexpr chunk_read_limit = 0;
  auto constexpr pass_read_limit  = 0;

  auto const read_opts = cudf::io::parquet_reader_options::builder(source_sink.make_source_info())
                           .use_arrow_schema(false)
                           .build();

  try_drop_l3_cache();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      auto sources   = cudf::io::make_datasources(source_sink.make_source_info());
      auto metadatas = cudf::io::read_parquet_footers(sources);

      // Constructing chunked parquet reader with existing datasource and metadata spends almost
      // entire time in column selection
      timer.start();
      auto reader = cudf::io::chunked_parquet_reader(
        chunk_read_limit, pass_read_limit, std::move(sources), std::move(metadatas), read_opts);
      timer.stop();

      // Validate
      CUDF_EXPECTS(reader.has_next(), "Expected reader to have data");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_cols) / time, "columns_per_second");
}

NVBENCH_BENCH(BM_parquet_read_footer)
  .set_name("parquet_read_footer")
  .set_min_samples(4)
  .add_string_axis("io_type", {"FILEPATH"})
  .add_int64_axis("page_index", {true, false})
  .add_int64_axis("num_cols", {64, 256, 512})
  .add_int64_axis("num_row_groups", {10, 20, 50});

NVBENCH_BENCH(BM_parquet_reader_construction)
  .set_name("parquet_reader_construction")
  .set_min_samples(4)
  .add_string_axis("io_type", {"FILEPATH"})
  .add_int64_axis("page_index", {true, false})
  .add_int64_axis("num_cols", {64, 256, 512})
  .add_int64_axis("num_row_groups", {10, 20, 50});

NVBENCH_BENCH(BM_parquet_column_selection)
  .set_name("parquet_column_selection")
  .set_min_samples(4)
  .add_string_axis("io_type", {"FILEPATH"})
  .add_int64_axis("num_cols", {64, 512, 2048});