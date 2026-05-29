/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>

#include <cstdint>

// Common mixed dtypes used by all benchmarks in this file
auto const mixed_dtypes = get_type_or_group({static_cast<int32_t>(data_type::STRING),
                                             static_cast<int32_t>(data_type::INTEGRAL),
                                             static_cast<int32_t>(data_type::FLOAT),
                                             static_cast<int32_t>(data_type::DECIMAL),
                                             static_cast<int32_t>(data_type::LIST)});

// Pure-integral dtypes for the stats availability sniff benchmark. Each column is just
// `int32` so the resulting file stays small (~ num_cols * num_row_groups * 4 bytes), and
// the metadata layout for the sniff is uniform across all column chunks.
auto const int32_dtypes = get_type_or_group({static_cast<int32_t>(cudf::type_id::INT32)});

namespace {

// Helper to generate and write parquet data using chunked writer
auto write_file_data(cudf::size_type num_cols,
                     cudf::size_type num_row_groups,
                     io_type source_type,
                     bool write_page_index)
{
  cuio_source_sink_pair source_sink(source_type);

  // Minimum row group size that cudf will almost always follow
  constexpr auto rows_per_row_group = 5000;
  constexpr auto min_row_groups     = 10;
  constexpr auto table_rows         = min_row_groups * rows_per_row_group;

  CUDF_EXPECTS(num_row_groups > 0 and num_row_groups % min_row_groups == 0,
               "Number of requested row groups must be non-zero and a multiple of " +
                 std::to_string(min_row_groups));

  // Create a table with the enough rows to cover min_row_groups
  auto const tbl =
    create_random_table(cycle_dtypes(mixed_dtypes, num_cols),
                        row_count{table_rows},
                        data_profile_builder().cardinality(0).avg_run_length(1).distribution(
                          cudf::type_id::LIST, distribution_id::GEOMETRIC, 0, 4));
  auto const view = tbl->view();

  auto const stats_level = write_page_index ? cudf::io::statistics_freq::STATISTICS_COLUMN
                                            : cudf::io::statistics_freq::STATISTICS_ROWGROUP;
  auto const options =
    cudf::io::chunked_parquet_writer_options::builder(source_sink.make_sink_info())
      .row_group_size_rows(rows_per_row_group)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(stats_level)
      .build();
  auto writer = cudf::io::chunked_parquet_writer(options, cudf::get_default_stream());

  // Compute the number of times the table needs to be written to cover the requested number of row
  // groups
  auto num_writes = cudf::util::div_rounding_up_unsafe(num_row_groups, min_row_groups);
  std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                cuda::counting_iterator{num_writes},
                [&](cudf::size_type) { writer.write(view); });

  std::ignore = writer.close();

  return source_sink;
}

// Writes a synthetic parquet file with `num_cols` int32 columns split into `num_row_groups`
// row groups, controlling the stats level. Used by the stats-availability sniff benchmark
// to compare the cost of walking metadata for files with stats vs. files written with
// STATISTICS_NONE.
auto write_sniff_test_file(cudf::size_type num_cols,
                           cudf::size_type num_row_groups,
                           io_type source_type,
                           cudf::io::statistics_freq stats_level)
{
  cuio_source_sink_pair source_sink(source_type);

  // Use the smallest row-group size that cuDF's writer will accept so the total bytes
  // written stays bounded for wide × tall configurations.
  constexpr auto rows_per_row_group = 5000;
  constexpr auto min_row_groups     = 10;
  constexpr auto table_rows         = min_row_groups * rows_per_row_group;

  CUDF_EXPECTS(num_row_groups > 0 and num_row_groups % min_row_groups == 0,
               "Number of requested row groups must be non-zero and a multiple of " +
                 std::to_string(min_row_groups));

  auto const tbl  = create_random_table(cycle_dtypes(int32_dtypes, num_cols),
                                       row_count{table_rows},
                                       data_profile_builder().cardinality(0).avg_run_length(1));
  auto const view = tbl->view();

  auto const options =
    cudf::io::chunked_parquet_writer_options::builder(source_sink.make_sink_info())
      .row_group_size_rows(rows_per_row_group)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(stats_level)
      .build();
  auto writer = cudf::io::chunked_parquet_writer(options, cudf::get_default_stream());

  auto num_writes = cudf::util::div_rounding_up_unsafe(num_row_groups, min_row_groups);
  std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                cuda::counting_iterator{num_writes},
                [&](cudf::size_type) { writer.write(view); });

  std::ignore = writer.close();

  return source_sink;
}

// First-row-group sniff: walks only the first selected row group of each source, but
// still scans every column chunk within that row group. Mirrors an earlier shape of
// `any_row_group_stats_available` in predicate_pushdown.cpp (before per-col scoping).
[[nodiscard]] bool sniff_first_rg_only(
  std::vector<cudf::io::parquet::FileMetaData> const& per_file_metadata,
  std::vector<std::vector<cudf::size_type>> const& input_row_group_indices)
{
  for (size_t src_idx = 0; src_idx < input_row_group_indices.size(); ++src_idx) {
    if (input_row_group_indices[src_idx].empty()) { continue; }
    auto const first_rg_idx = input_row_group_indices[src_idx].front();
    auto const& row_group   = per_file_metadata[src_idx].row_groups[first_rg_idx];
    for (auto const& colchunk : row_group.columns) {
      auto const& stats = colchunk.meta_data.statistics;
      if (stats.min_value.has_value() or stats.max_value.has_value() or stats.min.has_value() or
          stats.max.has_value() or stats.null_count.has_value()) {
        return true;
      }
    }
  }
  return false;
}

// Filter-cols-only sniff: walks only the first selected row group of each source, and
// within that row group only the column chunks whose schema indices appear in
// `filter_column_schemas`. Mirrors the current shape of `any_row_group_stats_available`
// in predicate_pushdown.cpp.
[[nodiscard]] bool sniff_first_rg_filter_cols_only(
  std::vector<cudf::io::parquet::FileMetaData> const& per_file_metadata,
  std::vector<std::vector<cudf::size_type>> const& input_row_group_indices,
  std::vector<int> const& filter_column_schemas)
{
  for (size_t src_idx = 0; src_idx < input_row_group_indices.size(); ++src_idx) {
    if (input_row_group_indices[src_idx].empty()) { continue; }
    auto const first_rg_idx = input_row_group_indices[src_idx].front();
    auto const& row_group   = per_file_metadata[src_idx].row_groups[first_rg_idx];
    for (auto const schema_idx : filter_column_schemas) {
      auto const col = std::find_if(row_group.columns.begin(),
                                    row_group.columns.end(),
                                    [schema_idx](cudf::io::parquet::ColumnChunk const& c) {
                                      return c.schema_idx == schema_idx;
                                    });
      if (col == row_group.columns.end()) { continue; }
      auto const& stats = col->meta_data.statistics;
      if (stats.min_value.has_value() or stats.max_value.has_value() or stats.min.has_value() or
          stats.max.has_value() or stats.null_count.has_value()) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

// Microbenchmark for the row-group-stats availability sniff used by
// `apply_stats_filters` in predicate_pushdown.cpp. Compares two variants:
//   - `first_rg_only`           : walks every column chunk of the first row group of each source
//                                 (scales O(num_sources × num_cols))
//   - `first_rg_filter_cols_only`: current, walks only the filter columns' chunks in that row
//                                 group (scales O(num_sources × num_filter_cols × num_cols) because
//                                 find_if linearly scans the chunk list unless schema_idx is dense)
// Both variants only read the first selected row group, so the timed cost is independent of
// num_row_groups; that axis is pinned to a single small value to keep the file-writing setup cheap.
// Run with --csv <file> to get plottable output.
void BM_parquet_stats_availability_sniff(nvbench::state& state)
{
  auto const num_cols        = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const num_row_groups  = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const has_stats       = state.get_int64("has_stats") != 0;
  auto const num_filter_cols = static_cast<cudf::size_type>(state.get_int64("num_filter_cols"));
  auto const variant         = state.get_string("variant");

  auto const stats_level = has_stats ? cudf::io::statistics_freq::STATISTICS_ROWGROUP
                                     : cudf::io::statistics_freq::STATISTICS_NONE;

  auto source_sink =
    write_sniff_test_file(num_cols, num_row_groups, io_type::FILEPATH, stats_level);

  auto sources = cudf::io::make_datasources(source_sink.make_source_info());
  auto footers = cudf::io::read_parquet_footers(sources);

  // Build "all row groups" indices once outside the timed loop.
  std::vector<std::vector<cudf::size_type>> input_row_group_indices(footers.size());
  for (size_t i = 0; i < footers.size(); ++i) {
    input_row_group_indices[i].resize(footers[i].row_groups.size());
    std::iota(input_row_group_indices[i].begin(), input_row_group_indices[i].end(), 0);
  }

  // Sanity check that the file we just wrote matches the requested config.
  CUDF_EXPECTS(std::cmp_equal(footers.front().row_groups.size(), num_row_groups),
               "Unexpected number of row groups in metadata");
  CUDF_EXPECTS(std::cmp_equal(footers.front().row_groups.front().columns.size(), num_cols),
               "Unexpected number of columns in metadata");
  CUDF_EXPECTS(num_filter_cols <= num_cols, "num_filter_cols must be <= num_cols");

  // Filter-cols schema indices: take the first num_filter_cols column chunks' schema_idx.
  std::vector<int> filter_column_schemas;
  filter_column_schemas.reserve(num_filter_cols);
  for (cudf::size_type i = 0; i < num_filter_cols; ++i) {
    filter_column_schemas.push_back(footers.front().row_groups.front().columns[i].schema_idx);
  }

  // Volatile sink so the compiler does not constant-fold the sniff away.
  volatile bool sink = false;

  // Host-only sniff: no GPU work, so use the manual timer without the sync tag. Adding
  // exec_tag::sync would force a GPU stream sync per iteration and dominate the wall time
  // (the sniff itself is microseconds), making each measurement hit the wall-clock timeout.
  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    bool result =
      (variant == "first_rg_only")
        ? sniff_first_rg_only(footers, input_row_group_indices)
        : sniff_first_rg_filter_cols_only(footers, input_row_group_indices, filter_column_schemas);
    timer.stop();
    sink = result;
  });
}

NVBENCH_BENCH(BM_parquet_stats_availability_sniff)
  .set_name("parquet_stats_availability_sniff")
  .set_min_samples(16)
  .add_int64_axis("num_cols", {64, 256, 1024, 4096})
  .add_int64_axis("num_row_groups", {10})
  .add_int64_axis("has_stats", {0, 1})
  .add_int64_axis("num_filter_cols", {1, 4, 16, 32})
  .add_string_axis("variant", {"first_rg_only", "first_rg_filter_cols_only"});

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
      auto const source_info = source_sink.make_source_info();
      drop_page_cache_if_enabled(source_info.filepaths());
      auto sources = cudf::io::make_datasources(source_info);

      timer.start();
      auto const metadatas = cudf::io::read_parquet_footers(sources);
      timer.stop();

      // Validate metadata
      CUDF_EXPECTS(std::cmp_equal(metadatas.size(), 1), "Expected one metadata object");
      CUDF_EXPECTS(std::cmp_equal(metadatas.front().row_groups.size(), num_row_groups),
                   "Unexpected number of row groups in metadata. Got: " +
                     std::to_string(metadatas.front().row_groups.size()) +
                     " Expected: " + std::to_string(num_row_groups));
      CUDF_EXPECTS(std::cmp_equal(metadatas.front().row_groups.front().columns.size(), num_cols),
                   "Unexpected number of columns in metadata");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_cols * num_row_groups) / time,
                          "colchunks_per_sec");
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
                           .allow_mismatched_pq_schemas(false)
                           .convert_strings_to_categories(false)
                           .build();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_opts.get_source().filepaths());
      timer.start();
      auto reader = cudf::io::chunked_parquet_reader(chunk_read_limit, pass_read_limit, read_opts);
      timer.stop();

      // Validate
      CUDF_EXPECTS(reader.has_next(), "Expected reader to have data");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_cols * num_row_groups) / time,
                          "colchunks_per_sec");
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
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      auto const source_info = source_sink.make_source_info();
      drop_page_cache_if_enabled(source_info.filepaths());
      auto sources   = cudf::io::make_datasources(source_info);
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
  state.add_element_count(static_cast<double>(num_cols) / time, "cols_per_sec");
}

NVBENCH_BENCH(BM_parquet_read_footer)
  .set_name("parquet_read_footer")
  .set_min_samples(4)
  .add_string_axis("io_type", {"FILEPATH"})
  .add_int64_axis("page_index", {true, false})
  .add_int64_axis("num_cols", {64, 256, 512})
  .add_int64_axis("num_row_groups", {10, 50});

NVBENCH_BENCH(BM_parquet_reader_construction)
  .set_name("parquet_reader_construction")
  .set_min_samples(4)
  .add_string_axis("io_type", {"FILEPATH"})
  .add_int64_axis("page_index", {true, false})
  .add_int64_axis("num_cols", {64, 256, 512})
  .add_int64_axis("num_row_groups", {10, 50});

NVBENCH_BENCH(BM_parquet_column_selection)
  .set_name("parquet_column_selection")
  .set_min_samples(4)
  .add_string_axis("io_type", {"FILEPATH"})
  .add_int64_axis("num_cols", {64, 512, 2048});
