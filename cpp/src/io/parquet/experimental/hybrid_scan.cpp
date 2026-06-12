/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_impl.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/aligned.hpp>
#include <rmm/mr/aligned_resource_adaptor.hpp>

#include <thrust/host_vector.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>

namespace cudf::io::parquet::experimental {

namespace {

/**
 * @brief Collects the top-level column names of a Parquet file in schema order
 *
 * Walks the flattened Parquet schema tree depth-first and records the names of the immediate
 * children of the root, skipping each child's descendants. This yields the column order that
 * `cudf::io::read_parquet` produces when no column selection is set.
 *
 * @param metadata Parquet file footer metadata
 * @return Top-level column names in file schema order
 */
[[nodiscard]] std::vector<std::string> top_level_column_names(FileMetaData const& metadata)
{
  auto const& schema = metadata.schema;
  if (schema.empty()) { return {}; }

  std::vector<std::string> names;
  names.reserve(schema.front().num_children);

  // Depth-first walk tracking unvisited children remaining at each open level. The root (index 0)
  // is skipped; a column is top-level when the only open level is the root's.
  std::vector<int32_t> remaining_children{schema.front().num_children};
  for (std::size_t i = 1; i < schema.size() and not remaining_children.empty(); ++i) {
    auto const& element = schema[i];
    if (remaining_children.size() == 1) { names.push_back(element.name); }

    --remaining_children.back();
    if (element.num_children > 0) {
      remaining_children.push_back(element.num_children);
    } else {
      while (not remaining_children.empty() and remaining_children.back() == 0) {
        remaining_children.pop_back();
      }
    }
  }

  return names;
}

/**
 * @brief Reassembles the filter and payload tables into a single table in `output_order`
 *
 * The two passes of a hybrid scan materialize disjoint sets of columns (filter columns and
 * payload columns). This combines them into one table whose columns follow `output_order`,
 * matching the projection order of `cudf::io::read_parquet`.
 *
 * @param filter Materialized filter columns and metadata
 * @param payload Materialized payload columns and metadata
 * @param output_order Desired output column names in order
 * @return Combined table and metadata in `output_order`
 */
[[nodiscard]] table_with_metadata assemble_output(table_with_metadata filter,
                                                  table_with_metadata payload,
                                                  std::vector<std::string> const& output_order)
{
  enum class source : bool { filter, payload };
  std::unordered_map<std::string, std::pair<source, std::size_t>> location;
  location.reserve(filter.metadata.schema_info.size() + payload.metadata.schema_info.size());
  for (std::size_t i = 0; i < filter.metadata.schema_info.size(); ++i) {
    location.emplace(filter.metadata.schema_info[i].name, std::pair{source::filter, i});
  }
  for (std::size_t i = 0; i < payload.metadata.schema_info.size(); ++i) {
    location.emplace(payload.metadata.schema_info[i].name, std::pair{source::payload, i});
  }

  auto filter_columns  = filter.tbl->release();
  auto payload_columns = payload.tbl->release();

  std::vector<std::unique_ptr<cudf::column>> output_columns;
  output_columns.reserve(output_order.size());
  table_metadata out_metadata;
  out_metadata.schema_info.reserve(output_order.size());
  out_metadata.num_rows_per_source = filter.metadata.num_rows_per_source;

  for (auto const& name : output_order) {
    auto const it = location.find(name);
    CUDF_EXPECTS(it != location.end(),
                 "Projected column not found in materialized hybrid scan output: " + name);
    auto const [tbl, pos] = it->second;
    if (tbl == source::filter) {
      output_columns.push_back(std::move(filter_columns[pos]));
      out_metadata.schema_info.push_back(std::move(filter.metadata.schema_info[pos]));
    } else {
      output_columns.push_back(std::move(payload_columns[pos]));
      out_metadata.schema_info.push_back(std::move(payload.metadata.schema_info[pos]));
    }
  }

  return table_with_metadata{std::make_unique<cudf::table>(std::move(output_columns)),
                             std::move(out_metadata)};
}

}  // namespace

hybrid_scan_reader::hybrid_scan_reader(cudf::host_span<uint8_t const> footer_bytes,
                                       parquet_reader_options const& options)
  : _impl{std::make_unique<detail::hybrid_scan_reader_impl>(
      std::vector<cudf::host_span<uint8_t const>>{footer_bytes}, options)}
{
}

hybrid_scan_reader::hybrid_scan_reader(FileMetaData const& parquet_metadata,
                                       parquet_reader_options const& options)
  : _impl{std::make_unique<detail::hybrid_scan_reader_impl>(
      std::vector<FileMetaData>{parquet_metadata}, options)}
{
}

hybrid_scan_reader::~hybrid_scan_reader() = default;

table_with_metadata hybrid_scan_reader::read(cudf::io::datasource& source,
                                             cudf::host_span<size_type const> row_group_indices,
                                             parquet_reader_options const& options,
                                             hybrid_scan_read_options const& read_options,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  // Transient byte-range fetches use the current device resource; output tables use `mr`.
  auto temp_mr = cudf::get_current_device_resource_ref();

  auto const output_order = options.get_column_names().has_value()
                              ? options.get_column_names().value()
                              : top_level_column_names(_impl->parquet_metadatas().front());

  std::vector<size_type> current_row_groups(row_group_indices.begin(), row_group_indices.end());

  // No filter: read all selected columns in a single pass.
  if (not options.get_filter().has_value()) {
    auto const byte_ranges = all_column_chunks_byte_ranges(current_row_groups, options);
    auto [buffers, data, tasks] =
      parquet::fetch_byte_ranges_to_device_async(source, byte_ranges, stream, temp_mr);
    tasks.get();
    return materialize_all_columns(current_row_groups, data, options, stream, mr);
  }

  // Use caller-provided page index bytes when available, otherwise fetch them from the source.
  auto const page_index_range = page_index_byte_range();
  auto const has_page_index   = not page_index_range.is_empty();
  if (has_page_index) {
    if (not read_options.page_index_bytes.empty()) {
      setup_page_index(read_options.page_index_bytes);
    } else {
      auto const page_index_buffer = parquet::fetch_page_index_to_host(source, page_index_range);
      setup_page_index(
        cudf::host_span<uint8_t const>{page_index_buffer->data(), page_index_buffer->size()});
    }
  }

  if (read_options.use_stats_filter and not current_row_groups.empty()) {
    current_row_groups = filter_row_groups_with_stats(current_row_groups, options, stream);
  }

  if ((read_options.use_dictionary_filter or read_options.use_bloom_filter) and
      not current_row_groups.empty()) {
    auto const [bloom_filter_ranges, dictionary_page_ranges] =
      secondary_filters_byte_ranges(current_row_groups, options);

    if (read_options.use_dictionary_filter and not dictionary_page_ranges.empty()) {
      auto [buffers, data, tasks] =
        parquet::fetch_byte_ranges_to_device_async(source, dictionary_page_ranges, stream, temp_mr);
      tasks.get();
      current_row_groups =
        filter_row_groups_with_dictionary_pages(data, current_row_groups, options, stream);
    }

    if (read_options.use_bloom_filter and not bloom_filter_ranges.empty() and
        not current_row_groups.empty()) {
      // Bloom filter data buffers must be allocated on 32-byte aligned addresses.
      auto aligned_mr = rmm::mr::aligned_resource_adaptor{temp_mr, rmm::CUDA_ALLOCATION_ALIGNMENT};
      auto [buffers, data, tasks] =
        parquet::fetch_byte_ranges_to_device_async(source, bloom_filter_ranges, stream, aligned_mr);
      tasks.get();
      current_row_groups =
        filter_row_groups_with_bloom_filters(data, current_row_groups, options, stream);
    }
  }

  // All row groups pruned: return a correctly typed, zero-row table in projection order. The
  // materialization path requires a non-empty row mask, so defer to the main reader here.
  if (current_row_groups.empty()) {
    auto empty_options = options;
    empty_options.set_num_rows(0);
    return cudf::io::read_parquet(empty_options, stream, mr);
  }

  // Use page-level statistics for the row mask when a page index is available; otherwise all-true.
  auto row_mask = has_page_index
                    ? build_row_mask_with_page_index_stats(current_row_groups, options, stream, mr)
                    : build_all_true_row_mask(current_row_groups, stream, mr);

  // Filter pass: materialize filter columns and narrow the row mask to surviving rows.
  auto row_mask_view = row_mask->mutable_view();
  auto filter_table  = [&] {
    auto const byte_ranges = filter_column_chunks_byte_ranges(current_row_groups, options);
    auto [buffers, data, tasks] =
      parquet::fetch_byte_ranges_to_device_async(source, byte_ranges, stream, temp_mr);
    tasks.get();
    return materialize_filter_columns(current_row_groups,
                                      data,
                                      row_mask_view,
                                      read_options.prune_filter_column_pages,
                                      options,
                                      stream,
                                      mr);
  }();

  // Payload pass: materialize payload columns under the surviving row mask.
  auto payload_table = [&] {
    auto const byte_ranges = payload_column_chunks_byte_ranges(current_row_groups, options);
    auto [buffers, data, tasks] =
      parquet::fetch_byte_ranges_to_device_async(source, byte_ranges, stream, temp_mr);
    tasks.get();
    return materialize_payload_columns(current_row_groups,
                                       data,
                                       row_mask->view(),
                                       read_options.prune_payload_column_pages,
                                       options,
                                       stream,
                                       mr);
  }();

  return assemble_output(std::move(filter_table), std::move(payload_table), output_order);
}

[[nodiscard]] text::byte_range_info hybrid_scan_reader::page_index_byte_range() const
{
  return _impl->page_index_byte_ranges().front();
}

[[nodiscard]] FileMetaData hybrid_scan_reader::parquet_metadata() const
{
  return _impl->parquet_metadatas().front();
}

void hybrid_scan_reader::setup_page_index(cudf::host_span<uint8_t const> page_index_bytes) const
{
  CUDF_FUNC_RANGE();
  return _impl->setup_page_indexes(std::vector<cudf::host_span<uint8_t const>>{page_index_bytes});
}

std::vector<cudf::size_type> hybrid_scan_reader::all_row_groups(
  parquet_reader_options const& options) const
{
  CUDF_EXPECTS(options.get_row_groups().size() <= 1,
               "Encountered invalid size of row group indices in parquet reader options");

  // If row groups are specified in parquet reader options, return them as is
  if (options.get_row_groups().size() == 1) { return options.get_row_groups().front(); }

  return _impl->all_row_groups(options).front();
}

std::size_t hybrid_scan_reader::total_rows_in_row_groups(
  cudf::host_span<size_type const> row_group_indices) const
{
  if (row_group_indices.empty()) { return 0; }

  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return _impl->total_rows_in_row_groups(input_row_group_indices);
}

void hybrid_scan_reader::reset_column_selection() const { _impl->reset_column_selection(); }

std::vector<size_type> hybrid_scan_reader::filter_row_groups_with_byte_range(
  cudf::host_span<size_type const> row_group_indices, parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->filter_row_groups_with_byte_range(input_row_group_indices, options).front();
}

std::vector<cudf::size_type> hybrid_scan_reader::filter_row_groups_with_stats(
  cudf::host_span<size_type const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->filter_row_groups_with_stats(input_row_group_indices, options, stream).front();
}

std::pair<std::vector<text::byte_range_info>, std::vector<text::byte_range_info>>
hybrid_scan_reader::secondary_filters_byte_ranges(
  cudf::host_span<size_type const> row_group_indices, parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->secondary_filters_byte_ranges(input_row_group_indices, options);
}

std::vector<cudf::size_type> hybrid_scan_reader::filter_row_groups_with_dictionary_pages(
  cudf::host_span<cudf::device_span<uint8_t const> const> dictionary_page_data,
  cudf::host_span<size_type const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl
    ->filter_row_groups_with_dictionary_pages(
      dictionary_page_data, input_row_group_indices, options, stream)
    .front();
}

std::vector<cudf::size_type> hybrid_scan_reader::filter_row_groups_with_bloom_filters(
  cudf::host_span<cudf::device_span<uint8_t const> const> bloom_filter_data,
  cudf::host_span<size_type const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl
    ->filter_row_groups_with_bloom_filters(
      bloom_filter_data, input_row_group_indices, options, stream)
    .front();
}

std::unique_ptr<cudf::column> hybrid_scan_reader::build_all_true_row_mask(
  cudf::host_span<size_type const> row_group_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->build_all_true_row_mask(input_row_group_indices, stream, mr);
}

std::unique_ptr<cudf::column> hybrid_scan_reader::build_row_mask_with_page_index_stats(
  cudf::host_span<size_type const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->build_row_mask_with_page_index_stats(input_row_group_indices, options, stream, mr);
}

[[nodiscard]] std::vector<text::byte_range_info>
hybrid_scan_reader::filter_column_chunks_byte_ranges(
  cudf::host_span<size_type const> row_group_indices, parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->filter_column_chunks_byte_ranges(input_row_group_indices, options).first;
}

table_with_metadata hybrid_scan_reader::materialize_filter_columns(
  cudf::host_span<size_type const> row_group_indices,
  cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::mutable_column_view& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->materialize_filter_columns(
    input_row_group_indices, column_chunk_data, row_mask, mask_data_pages, options, stream, mr);
}

[[nodiscard]] std::vector<text::byte_range_info>
hybrid_scan_reader::payload_column_chunks_byte_ranges(
  cudf::host_span<size_type const> row_group_indices, parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();

  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->payload_column_chunks_byte_ranges(input_row_group_indices, options).first;
}

table_with_metadata hybrid_scan_reader::materialize_payload_columns(
  cudf::host_span<size_type const> row_group_indices,
  cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->materialize_payload_columns(
    input_row_group_indices, column_chunk_data, row_mask, mask_data_pages, options, stream, mr);
}

std::vector<byte_range_info> hybrid_scan_reader::all_column_chunks_byte_ranges(
  cudf::host_span<size_type const> row_group_indices, parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();

  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->all_column_chunks_byte_ranges(input_row_group_indices, options).first;
}

table_with_metadata hybrid_scan_reader::materialize_all_columns(
  cudf::host_span<size_type const> row_group_indices,
  cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->materialize_all_columns(
    input_row_group_indices, column_chunk_data, options, stream, mr);
}

void hybrid_scan_reader::setup_chunking_for_filter_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  cudf::host_span<size_type const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->setup_chunking_for_filter_columns(chunk_read_limit,
                                                  pass_read_limit,
                                                  input_row_group_indices,
                                                  row_mask,
                                                  mask_data_pages,
                                                  column_chunk_data,
                                                  options,
                                                  stream,
                                                  mr);
}

table_with_metadata hybrid_scan_reader::materialize_filter_columns_chunk(
  cudf::mutable_column_view& row_mask) const
{
  CUDF_FUNC_RANGE();

  return _impl->materialize_filter_columns_chunk(row_mask);
}

void hybrid_scan_reader::setup_chunking_for_payload_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  cudf::host_span<size_type const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->setup_chunking_for_payload_columns(chunk_read_limit,
                                                   pass_read_limit,
                                                   input_row_group_indices,
                                                   row_mask,
                                                   mask_data_pages,
                                                   column_chunk_data,
                                                   options,
                                                   stream,
                                                   mr);
}

table_with_metadata hybrid_scan_reader::materialize_payload_columns_chunk(
  cudf::column_view const& row_mask) const
{
  CUDF_FUNC_RANGE();

  return _impl->materialize_payload_columns_chunk(row_mask);
}

void hybrid_scan_reader::setup_chunking_for_all_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  cudf::host_span<size_type const> row_group_indices,
  cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->setup_chunking_for_all_columns(chunk_read_limit,
                                               pass_read_limit,
                                               input_row_group_indices,
                                               column_chunk_data,
                                               options,
                                               stream,
                                               mr);
}

table_with_metadata hybrid_scan_reader::materialize_all_columns_chunk() const
{
  CUDF_FUNC_RANGE();

  return _impl->materialize_all_columns_chunk();
}

std::vector<std::vector<cudf::size_type>> hybrid_scan_reader::construct_row_group_passes(
  cudf::host_span<cudf::size_type const> row_group_indices, std::size_t pass_read_limit) const
{
  return _impl->construct_row_group_passes(row_group_indices, pass_read_limit);
}

bool hybrid_scan_reader::has_next_table_chunk() const { return _impl->has_next_table_chunk(); }

}  // namespace cudf::io::parquet::experimental
