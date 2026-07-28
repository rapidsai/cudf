/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_impl.hpp"

#include "cudf/io/text/byte_range_info.hpp"
#include "hybrid_scan_helpers.hpp"
#include "io/parquet/reader_impl_chunking_utils.cuh"

#include <cudf/copying.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <thrust/host_vector.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

namespace cudf::io::parquet::experimental::detail {

using io::detail::inline_column_buffer;
using parquet::detail::ColumnChunkDesc;
using parquet::detail::decode_kernel_mask;
using parquet::detail::file_intermediate_data;
using parquet::detail::PageInfo;
using parquet::detail::PageNestingDecodeInfo;
using text::byte_range_info;

namespace {

/**
 * @brief Tests the logical type for a fixed length byte array column to see if it should be
 * treated as a string.
 *
 * Currently the only logical type that has special handling is DECIMAL. Other valid types in the
 * future would be UUID (still treated as string) and FLOAT16 (which for now would also be treated
 * as a string).
 *
 * @param logical_type The logical type to test
 * @return Boolean indicating if the logical type should be treated as a string
 */
[[maybe_unused]] inline bool is_treat_fixed_length_as_string(
  cuda::std::optional<LogicalType> const& logical_type)
{
  if (!logical_type.has_value()) { return true; }
  return logical_type->type != LogicalType::DECIMAL;
}

/**
 * @brief Get the output types from the output buffer template
 *
 * @param output_buffer_template Output buffer template
 * @return Output types
 */
[[nodiscard]] std::vector<cudf::data_type> get_output_types(
  std::span<inline_column_buffer const> output_buffer_template)
{
  std::vector<cudf::data_type> output_dtypes;
  output_dtypes.reserve(output_buffer_template.size());
  std::transform(output_buffer_template.begin(),
                 output_buffer_template.end(),
                 std::back_inserter(output_dtypes),
                 [](auto const& col) { return col.type; });
  return output_dtypes;
}

/**
 * @brief Count the number of row groups in the input
 *
 * @param row_group_indices Row group indices
 * @return Number of row groups
 */
[[nodiscard]] inline size_type count_row_groups(
  std::span<std::vector<size_type> const> row_group_indices)
{
  return std::accumulate(
    row_group_indices.begin(),
    row_group_indices.end(),
    size_type{0},
    [](auto sum, auto const& rgs) { return sum + static_cast<size_type>(rgs.size()); });
}

}  // namespace

hybrid_scan_reader_impl::hybrid_scan_reader_impl(
  cudf::host_span<cudf::host_span<uint8_t const> const> footer_bytes,
  parquet_reader_options const& options)
{
  _metadata = std::make_unique<aggregate_reader_metadata>(
    footer_bytes, options.is_enabled_use_arrow_schema(), has_cols_from_mismatched_sources(options));

  _extended_metadata = static_cast<aggregate_reader_metadata*>(_metadata.get());
}

hybrid_scan_reader_impl::hybrid_scan_reader_impl(
  cudf::host_span<FileMetaData const> parquet_metadatas, parquet_reader_options const& options)
{
  _metadata =
    std::make_unique<aggregate_reader_metadata>(parquet_metadatas,
                                                options.is_enabled_use_arrow_schema(),
                                                has_cols_from_mismatched_sources(options));
  _extended_metadata = static_cast<aggregate_reader_metadata*>(_metadata.get());
}

std::vector<FileMetaData> hybrid_scan_reader_impl::parquet_metadatas() const
{
  return _extended_metadata->parquet_metadatas();
}

std::vector<byte_range_info> hybrid_scan_reader_impl::page_index_byte_ranges() const
{
  return _extended_metadata->page_index_byte_ranges();
}

void hybrid_scan_reader_impl::setup_page_indexes(
  cudf::host_span<cudf::host_span<uint8_t const> const> page_index_bytes) const
{
  _extended_metadata->setup_page_indexes(page_index_bytes);
}

void hybrid_scan_reader_impl::select_columns(read_columns_mode read_columns_mode,
                                             parquet_reader_options const& options)
{
  // Initialize reader configuration.
  initialize_reader_config(options);

  // Build column selection options directly from the user options.
  auto selection_options = make_column_selection_options(options);

  if (read_columns_mode == read_columns_mode::ALL_COLUMNS) {
    if (_is_all_columns_selected) { return; }

    // Select only columns required by the options and filter
    auto const select_column_names = get_column_projection(options);

    // Select only columns required by the options and filter.
    // Using as is from:
    // https://github.com/rapidsai/cudf/blob/a8b25cd205dc5d04b9918dcb0b3abd6b8c4e4a74/cpp/src/io/parquet/reader_impl.cpp#L556-L569
    std::optional<std::vector<std::string>> filter_only_columns_names;
    if (options.get_filter().has_value() and select_column_names.has_value()) {
      filter_only_columns_names = parquet::detail::get_column_names_in_expression(
        options.get_filter(), *select_column_names, options, _extended_metadata->get_schema_tree());
      _num_filter_only_columns = filter_only_columns_names->size();
    }
    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _metadata->select_columns(select_column_names, filter_only_columns_names, selection_options);

    _is_all_columns_selected     = true;
    _is_filter_columns_selected  = false;
    _is_payload_columns_selected = false;
  } else if (read_columns_mode == read_columns_mode::FILTER_COLUMNS) {
    if (_is_filter_columns_selected) { return; }
    // Must not ignore missing filter columns
    selection_options.ignore_missing_columns = false;

    _filter_columns_names = cudf::io::parquet::detail::get_column_names_in_expression(
      options.get_filter(), {}, options, _extended_metadata->get_schema_tree());
    // Select only filter columns using the base `select_columns` method
    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _extended_metadata->select_columns(_filter_columns_names, {}, selection_options);

    _is_filter_columns_selected  = true;
    _is_payload_columns_selected = false;
    _is_all_columns_selected     = false;
  } else {
    if (_is_payload_columns_selected) { return; }

    auto select_column_names = get_column_projection(options);
    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _extended_metadata->select_payload_columns(
        select_column_names, _filter_columns_names, selection_options);

    _is_payload_columns_selected = true;
    _is_filter_columns_selected  = false;
    _is_all_columns_selected     = false;
  }

  // Reset the materialization step flag
  _output_chunk_produced = false;

  CUDF_EXPECTS(_input_columns.size() > 0 and _output_buffers.size() > 0, "No columns selected");

  // Clear the output buffers templates
  _output_buffers_template.clear();

  // Save the states of the output buffers for reuse.
  std::transform(_output_buffers.begin(),
                 _output_buffers.end(),
                 std::back_inserter(_output_buffers_template),
                 [](auto const& buff) { return inline_column_buffer::empty_like(buff); });
}

std::vector<std::vector<size_type>> hybrid_scan_reader_impl::all_row_groups(
  parquet_reader_options const& options) const
{
  return _extended_metadata->all_row_groups(options);
}

std::size_t hybrid_scan_reader_impl::total_rows_in_row_groups(
  std::span<std::vector<size_type> const> row_group_indices) const
{
  return _extended_metadata->total_rows_in_row_groups(row_group_indices);
}

void hybrid_scan_reader_impl::reset_column_selection()
{
  CUDF_EXPECTS(not _pending_payload_page_io_plan.has_value(),
               "Cannot reset column selection while a payload page I/O plan is pending");
  _is_all_columns_selected     = false;
  _is_filter_columns_selected  = false;
  _is_payload_columns_selected = false;
}

std::pair<named_to_reference_converter, std::vector<cudf::data_type>>
hybrid_scan_reader_impl::prepare_filter_and_output_types(parquet_reader_options const& options)
{
  CUDF_EXPECTS(options.get_filter().has_value(), "Empty input filter expression encountered");

  select_columns(read_columns_mode::FILTER_COLUMNS, options);

  // Convert the input expression (must be done after column selection)
  auto expr_conv     = build_converted_expression(options);
  auto output_dtypes = get_output_types(_output_buffers_template);

  return {std::move(expr_conv), std::move(output_dtypes)};
}

void hybrid_scan_reader_impl::prepare_materialization(read_columns_mode read_columns_mode,
                                                      std::size_t num_sources,
                                                      parquet_reader_options const& options,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not _pending_payload_page_io_plan.has_value(),
               "Pending payload page I/O plan must be consumed by its setup overload");
  reset_internal_state();
  initialize_options(options, num_sources, stream, mr);
  select_columns(read_columns_mode, options);
}

std::vector<std::vector<cudf::size_type>>
hybrid_scan_reader_impl::filter_row_groups_with_byte_range(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  if (options.get_skip_bytes() == 0 and not options.get_num_bytes().has_value()) {
    return std::vector<std::vector<cudf::size_type>>{row_group_indices.begin(),
                                                     row_group_indices.end()};
  }

  return _extended_metadata->filter_row_groups_with_byte_range(
    row_group_indices, options.get_skip_bytes(), options.get_num_bytes());
}

std::vector<std::vector<size_type>> hybrid_scan_reader_impl::filter_row_groups_with_stats(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  return _extended_metadata->filter_row_groups_with_stats(row_group_indices,
                                                          output_dtypes,
                                                          _output_column_schemas,
                                                          expr_conv.get_converted_expr().value(),
                                                          stream);
}

std::pair<std::vector<byte_range_info>, std::vector<byte_range_info>>
hybrid_scan_reader_impl::secondary_filters_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  auto const bloom_filter_bytes =
    _extended_metadata->get_bloom_filter_bytes(row_group_indices,
                                               output_dtypes,
                                               _output_column_schemas,
                                               expr_conv.get_converted_expr().value());
  auto const dictionary_page_bytes =
    _extended_metadata
      ->dictionary_pages_byte_ranges(row_group_indices,
                                     output_dtypes,
                                     _output_column_schemas,
                                     expr_conv.get_converted_expr().value())
      .first;

  return {bloom_filter_bytes, dictionary_page_bytes};
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::dictionary_pages_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  return _extended_metadata->dictionary_pages_byte_ranges(row_group_indices,
                                                          output_dtypes,
                                                          _output_column_schemas,
                                                          expr_conv.get_converted_expr().value());
}

std::vector<std::vector<size_type>>
hybrid_scan_reader_impl::filter_row_groups_with_dictionary_pages(
  std::span<cudf::device_span<uint8_t const> const> dictionary_page_data,
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  // Collect literal and operator pairs for each input column with an (in)equality predicate
  auto const [literals, operators] =
    dictionary_literals_collector{expr_conv.get_converted_expr().value().get(), output_dtypes}
      .get_literals_and_operators();

  // Return all row groups if no dictionary page filtering is needed
  if (literals.empty() or std::all_of(literals.begin(), literals.end(), [](auto& col_literals) {
        return col_literals.empty();
      })) {
    return std::vector<std::vector<size_type>>(row_group_indices.begin(), row_group_indices.end());
  }

  // Collect schema indices of input columns with a non-empty (in)equality literal/operator vector
  std::vector<cudf::size_type> dictionary_col_schemas;
  thrust::copy_if(thrust::host,
                  _output_column_schemas.begin(),
                  _output_column_schemas.end(),
                  literals.begin(),
                  std::back_inserter(dictionary_col_schemas),
                  [](auto& dict_literals) { return not dict_literals.empty(); });

  // Prepare dictionary column chunks and decode page headers
  auto [has_compressed_data, chunks, pages] = prepare_dictionaries(
    row_group_indices, dictionary_page_data, dictionary_col_schemas, options, stream);

  // Decompress dictionary pages if needed and store uncompressed buffers here
  auto const mr                          = cudf::get_current_device_resource_ref();
  auto decompressed_dictionary_page_data = std::optional<rmm::device_buffer>{};
  if (has_compressed_data) {
    // Use the `decompress_page_data` utility to decompress dictionary pages (passed as pass_pages)
    decompressed_dictionary_page_data =
      std::get<0>(parquet::detail::decompress_page_data(chunks, pages, {}, {}, stream, mr));
    pages.host_to_device_async(stream);
  }

  // Filter row groups using dictionary pages
  return _extended_metadata->filter_row_groups_with_dictionary_pages(
    chunks,
    pages,
    row_group_indices,
    literals,
    operators,
    output_dtypes,
    dictionary_col_schemas,
    expr_conv.get_converted_expr().value(),
    stream);
}

std::vector<std::vector<size_type>> hybrid_scan_reader_impl::filter_row_groups_with_bloom_filters(
  std::span<cudf::device_span<uint8_t const> const> bloom_filter_data,
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  return _extended_metadata->filter_row_groups_with_bloom_filters(
    bloom_filter_data,
    row_group_indices,
    output_dtypes,
    _output_column_schemas,
    expr_conv.get_converted_expr().value(),
    stream);
}

std::unique_ptr<cudf::column> hybrid_scan_reader_impl::build_all_true_row_mask(
  std::span<std::vector<size_type> const> row_group_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  auto const num_rows = total_rows_in_row_groups(row_group_indices);
  CUDF_EXPECTS(num_rows < std::numeric_limits<cudf::size_type>::max(),
               "Total rows in row groups exceed the cudf's column size limit. Retry with a smaller "
               "set of row groups",
               std::invalid_argument);
  auto true_scalar =
    cudf::numeric_scalar<bool>(true, true, stream, cudf::get_current_device_resource_ref());
  return cudf::make_column_from_scalar(true_scalar, num_rows, stream, mr);
}

std::unique_ptr<cudf::column> hybrid_scan_reader_impl::build_row_mask_with_page_index_stats(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  return _extended_metadata->build_row_mask_with_page_index_stats(
    row_group_indices,
    output_dtypes,
    _output_column_schemas,
    expr_conv.get_converted_expr().value(),
    stream,
    mr);
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::get_input_column_chunk_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices) const
{
  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_row_groups    = std::accumulate(
    row_group_indices.begin(),
    row_group_indices.end(),
    std::size_t{0},
    [](std::size_t sum, auto const& row_groups) { return sum + row_groups.size(); });
  auto const num_chunks = num_row_groups * num_input_columns;

  // Association between each column chunk and its source
  auto chunk_source_map = std::vector<size_type>{};
  chunk_source_map.reserve(num_chunks);

  // Keep track of column chunk byte ranges
  auto column_chunk_byte_ranges = std::vector<byte_range_info>{};
  column_chunk_byte_ranges.reserve(num_chunks);

  std::for_each(cuda::counting_iterator<std::size_t>{0},
                cuda::counting_iterator{row_group_indices.size()},
                [&](auto const source_idx) {
                  auto const& row_groups = row_group_indices[source_idx];
                  for (auto const row_group_index : row_groups) {
                    // generate ColumnChunkDesc objects for everything to be decoded (all input
                    // columns)
                    for (auto const& col : _input_columns) {
                      // look up metadata
                      auto const& col_meta = _extended_metadata->get_column_metadata(
                        row_group_index, source_idx, col.schema_idx);

                      auto const chunk_offset =
                        (col_meta.dictionary_page_offset != 0)
                          ? std::min(col_meta.data_page_offset, col_meta.dictionary_page_offset)
                          : col_meta.data_page_offset;

                      auto const chunk_size = col_meta.total_compressed_size;
                      column_chunk_byte_ranges.emplace_back(chunk_offset, chunk_size);

                      // Map each column chunk to its column index and its source index
                      chunk_source_map.emplace_back(static_cast<size_type>(source_idx));
                    }
                  }
                });

  return {std::move(column_chunk_byte_ranges), std::move(chunk_source_map)};
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::filter_column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  select_columns(read_columns_mode::FILTER_COLUMNS, options);
  return get_input_column_chunk_byte_ranges(row_group_indices);
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::payload_column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  select_columns(read_columns_mode::PAYLOAD_COLUMNS, options);
  return get_input_column_chunk_byte_ranges(row_group_indices);
}

std::vector<std::vector<byte_range_info>>
hybrid_scan_reader_impl::payload_column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(row_group_indices.size() == _extended_metadata->get_num_sources(),
               "Row group source count must match the number of input sources");
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");
  CUDF_EXPECTS(row_mask.null_count() == 0,
               "Row mask must not have any nulls when planning payload pages");
  CUDF_EXPECTS(not _pending_payload_page_io_plan.has_value(),
               "The previous payload page I/O plan has not been consumed");

  select_columns(read_columns_mode::PAYLOAD_COLUMNS, options);

  auto column_schemas = std::vector<int>{};
  column_schemas.reserve(_input_columns.size());
  std::transform(_input_columns.begin(),
                 _input_columns.end(),
                 std::back_inserter(column_schemas),
                 [](auto const& col) { return col.schema_idx; });

  auto make_full_chunk_plan = [&]() {
    auto [flat_ranges, source_map] = get_input_column_chunk_byte_ranges(row_group_indices);
    auto source_ranges = std::vector<std::vector<byte_range_info>>(row_group_indices.size());
    CUDF_EXPECTS(flat_ranges.size() == source_map.size(),
                 "Column chunk range source map is invalid");
    for (std::size_t i = 0; i < flat_ranges.size(); ++i) {
      CUDF_EXPECTS(std::cmp_less(source_map[i], source_ranges.size()),
                   "Column chunk range has an invalid source index");
      source_ranges[source_map[i]].push_back(flat_ranges[i]);
    }

    _pending_payload_page_io_plan = payload_page_io_plan{
      .sparse                       = false,
      .mask_data_pages              = mask_data_pages,
      .row_group_indices            = {row_group_indices.begin(), row_group_indices.end()},
      .column_schema_indices        = column_schemas,
      .source_ranges                = source_ranges,
      .page_mappings                = {},
      .resident_bytes_per_chunk     = {},
      .dictionary_present_per_chunk = {},
      .data_page_mask               = {}};
    return source_ranges;
  };

  if (mask_data_pages == use_data_page_mask::NO or row_mask.is_empty()) {
    return make_full_chunk_plan();
  }

  // Sparse page planning only requires offset-index topology. Value counts and variable-width
  // sizes can be derived from each retained page after it is fetched.
  auto indexes_complete = true;
  for (std::size_t source_idx = 0; source_idx < row_group_indices.size(); ++source_idx) {
    for (auto const row_group_idx : row_group_indices[source_idx]) {
      auto const& row_group = _extended_metadata->get_row_group(row_group_idx, source_idx);
      for (auto const schema_idx : column_schemas) {
        auto const candidate_it = std::find_if(
          row_group.columns.begin(), row_group.columns.end(), [schema_idx](auto const& candidate) {
            return candidate.schema_idx == schema_idx;
          });
        if (candidate_it == row_group.columns.end() or
            not candidate_it->offset_index.has_value()) {
          indexes_complete = false;
          break;
        }
        auto const& candidate = *candidate_it;
        auto const& oi = candidate.offset_index.value();
        auto const num_pages = oi.page_locations.size();
        auto const index_vector_sizes_valid =
          not oi.unencoded_byte_array_data_bytes.has_value() or
          oi.unencoded_byte_array_data_bytes->size() == num_pages;
        auto const dictionary_offsets_valid =
          candidate.meta_data.dictionary_page_offset <= 0 or
          candidate.meta_data.data_page_offset > candidate.meta_data.dictionary_page_offset;
        auto const page_rows_valid =
          num_pages > 0 and oi.page_locations.front().first_row_index == 0 and
          std::is_sorted(oi.page_locations.begin(),
                         oi.page_locations.end(),
                         [](auto const& lhs, auto const& rhs) {
                           return lhs.first_row_index < rhs.first_row_index;
                         }) and
          std::all_of(
            oi.page_locations.begin(), oi.page_locations.end(), [&](auto const& location) {
              return location.first_row_index >= 0 and
                     std::cmp_less_equal(location.first_row_index, row_group.num_rows);
            });
        if (num_pages == 0 or not index_vector_sizes_valid or not page_rows_valid or
            candidate.meta_data.data_page_offset <= 0 or not dictionary_offsets_valid or
            std::any_of(
              oi.page_locations.begin(), oi.page_locations.end(), [](auto const& location) {
                return location.offset < 0 or location.compressed_page_size <= 0;
              })) {
          indexes_complete = false;
          break;
        }
      }
      if (not indexes_complete) { break; }
    }
    if (not indexes_complete) { break; }
  }
  if (not indexes_complete) { return make_full_chunk_plan(); }

  auto data_page_mask = _extended_metadata->compute_data_page_mask(
    row_mask, row_group_indices, _input_columns, 0, stream);
  // An empty mask is the established representation for "all pages retained".
  if (data_page_mask.empty()) { return make_full_chunk_plan(); }

  auto const num_columns = _input_columns.size();
  auto const num_row_groups =
    std::accumulate(row_group_indices.begin(),
                    row_group_indices.end(),
                    std::size_t{0},
                    [](auto sum, auto const& groups) { return sum + groups.size(); });
  auto const num_chunks = num_row_groups * num_columns;
  auto chunk_masks      = std::vector<std::vector<uint8_t>>(num_chunks);

  // Translate the column-major mask into source-major/row-group-major chunk slots once.
  std::size_t mask_idx = 0;
  for (std::size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
    std::size_t row_group_ordinal = 0;
    for (std::size_t source_idx = 0; source_idx < row_group_indices.size(); ++source_idx) {
      for (auto const row_group_idx : row_group_indices[source_idx]) {
        auto const& row_group = _extended_metadata->get_row_group(row_group_idx, source_idx);
        auto const schema_idx = column_schemas[col_idx];
        auto const col        = std::find_if(
          row_group.columns.begin(), row_group.columns.end(), [schema_idx](auto const& candidate) {
            return candidate.schema_idx == schema_idx;
          });
        CUDF_EXPECTS(col != row_group.columns.end(), "Selected payload column is missing");
        auto const page_count = col->offset_index->page_locations.size();
        CUDF_EXPECTS(mask_idx + page_count <= data_page_mask.size(),
                     "Computed data page mask is incomplete");
        auto& mask = chunk_masks[row_group_ordinal * num_columns + col_idx];
        mask.reserve(page_count);
        std::transform(data_page_mask.begin() + mask_idx,
                       data_page_mask.begin() + mask_idx + page_count,
                       std::back_inserter(mask),
                       [](bool retained) { return static_cast<uint8_t>(retained); });
        mask_idx += page_count;
        ++row_group_ordinal;
      }
    }
  }
  // compute_data_page_mask currently leaves unused trailing entries after the logical
  // column-major page mask. Preserve the established consumer behavior by discarding them here.
  data_page_mask.resize(mask_idx);

  struct exact_request {
    int64_t offset;
    int64_t size;
    std::size_t mapping_idx;
  };
  auto exact_requests     = std::vector<std::vector<exact_request>>(row_group_indices.size());
  auto page_mappings      = std::vector<page_range_mapping>{};
  auto resident_bytes     = std::vector<std::size_t>(num_chunks, 0);
  auto dictionary_present = std::vector<uint8_t>(num_chunks, 0);

  std::size_t row_group_ordinal = 0;
  for (std::size_t source_idx = 0; source_idx < row_group_indices.size(); ++source_idx) {
    for (auto const row_group_idx : row_group_indices[source_idx]) {
      auto const& row_group = _extended_metadata->get_row_group(row_group_idx, source_idx);
      for (std::size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
        auto const chunk_idx  = row_group_ordinal * num_columns + col_idx;
        auto const schema_idx = column_schemas[col_idx];
        auto const col        = std::find_if(
          row_group.columns.begin(), row_group.columns.end(), [schema_idx](auto const& candidate) {
            return candidate.schema_idx == schema_idx;
          });
        CUDF_EXPECTS(col != row_group.columns.end(), "Selected payload column is missing");
        auto const& page_locations = col->offset_index->page_locations;
        auto const& retained       = chunk_masks[chunk_idx];
        CUDF_EXPECTS(retained.size() == page_locations.size(),
                     "Data page mask does not match the offset index");
        auto const any_retained =
          std::any_of(retained.begin(), retained.end(), [](auto value) { return value != 0; });

        std::optional<std::pair<int64_t, int64_t>> dictionary_range;
        if (col->meta_data.dictionary_page_offset > 0) {
          auto const offset = col->meta_data.dictionary_page_offset;
          auto const size   = col->meta_data.data_page_offset - offset;
          if (size > 0) { dictionary_range = std::pair{offset, size}; }
        } else if (col->meta_data.data_page_offset < page_locations.front().offset) {
          auto const offset = col->meta_data.data_page_offset;
          dictionary_range =
            std::pair{offset, page_locations.front().offset - col->meta_data.data_page_offset};
        }

        auto add_mapping = [&](bool fetched, int64_t offset, int64_t size) {
          CUDF_EXPECTS(
            offset >= 0 and size > 0 and offset <= std::numeric_limits<int64_t>::max() - size,
            "Indexed page byte range is invalid");
          auto const mapping_idx = page_mappings.size();
          page_mappings.push_back(
            page_range_mapping{.source_idx   = static_cast<size_type>(source_idx),
                               .range_idx    = 0,
                               .range_offset = 0,
                               .size         = fetched ? static_cast<std::size_t>(size) : 0,
                               .fetched      = fetched});
          if (fetched) {
            exact_requests[source_idx].push_back(exact_request{offset, size, mapping_idx});
            resident_bytes[chunk_idx] += static_cast<std::size_t>(size);
          }
        };

        if (dictionary_range.has_value() and any_retained) {
          add_mapping(true, dictionary_range->first, dictionary_range->second);
          dictionary_present[chunk_idx] = 1;
        }
        for (std::size_t page_idx = 0; page_idx < page_locations.size(); ++page_idx) {
          auto const& location = page_locations[page_idx];
          add_mapping(retained[page_idx] != 0,
                      location.offset,
                      static_cast<int64_t>(location.compressed_page_size));
        }
      }
      ++row_group_ordinal;
    }
  }

  auto source_ranges = std::vector<std::vector<byte_range_info>>(row_group_indices.size());
  for (std::size_t source_idx = 0; source_idx < exact_requests.size(); ++source_idx) {
    auto& requests = exact_requests[source_idx];
    std::stable_sort(requests.begin(), requests.end(), [](auto const& lhs, auto const& rhs) {
      return std::tie(lhs.offset, lhs.size) < std::tie(rhs.offset, rhs.size);
    });
    for (auto const& request : requests) {
      auto& ranges = source_ranges[source_idx];
      if (ranges.empty() or request.offset > ranges.back().offset() + ranges.back().size()) {
        ranges.emplace_back(request.offset, request.size);
      } else {
        auto const end =
          std::max(ranges.back().offset() + ranges.back().size(), request.offset + request.size);
        ranges.back() = byte_range_info{ranges.back().offset(), end - ranges.back().offset()};
      }
      auto& mapping        = page_mappings[request.mapping_idx];
      mapping.range_idx    = ranges.size() - 1;
      mapping.range_offset = static_cast<std::size_t>(request.offset - ranges.back().offset());
    }
  }

  _pending_payload_page_io_plan =
    payload_page_io_plan{.sparse            = true,
                         .mask_data_pages   = mask_data_pages,
                         .row_group_indices = {row_group_indices.begin(), row_group_indices.end()},
                         .column_schema_indices        = std::move(column_schemas),
                         .source_ranges                = source_ranges,
                         .page_mappings                = std::move(page_mappings),
                         .resident_bytes_per_chunk     = std::move(resident_bytes),
                         .dictionary_present_per_chunk = std::move(dictionary_present),
                         .data_page_mask               = std::move(data_page_mask)};
  return source_ranges;
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::all_column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  select_columns(read_columns_mode::ALL_COLUMNS, options);
  return get_input_column_chunk_byte_ranges(row_group_indices);
}

table_with_metadata hybrid_scan_reader_impl::materialize_filter_columns(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::mutable_column_view& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");
  CUDF_EXPECTS(options.get_filter().has_value(), "Empty input filter expression encountered");

  prepare_materialization(
    read_columns_mode::FILTER_COLUMNS, row_group_indices.size(), options, stream, mr);

  // Convert the input expression (must be done after prepare_materialization)
  _expr_conv = build_converted_expression(options);

  // Return early if all rows are pruned
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::READ_ALL, empty_row_groups, {}, {});
    // Set correct number of input row groups to the output metadata
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::FILTER_COLUMNS, row_mask);
  }

  auto data_page_mask = thrust::host_vector<bool>{};
  if (mask_data_pages == use_data_page_mask::YES) {
    data_page_mask = _extended_metadata->compute_data_page_mask(
      row_mask, row_group_indices, _input_columns, _row_mask_offset, stream);
  }

  prepare_data(read_mode::READ_ALL, row_group_indices, column_chunk_data, data_page_mask);

  return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::FILTER_COLUMNS, row_mask);
}

table_with_metadata hybrid_scan_reader_impl::materialize_payload_columns(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");
  CUDF_EXPECTS(row_mask.null_count() == 0,
               "Row mask must not have any nulls when materializing payload column");

  prepare_materialization(
    read_columns_mode::PAYLOAD_COLUMNS, row_group_indices.size(), options, stream, mr);

  // Return early if all rows are pruned
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::READ_ALL, empty_row_groups, {}, {});
    // Set correct number of input row groups to the output metadata
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::PAYLOAD_COLUMNS, row_mask);
  }

  auto data_page_mask = thrust::host_vector<bool>{};
  if (not row_mask.is_empty() and mask_data_pages == use_data_page_mask::YES) {
    data_page_mask = _extended_metadata->compute_data_page_mask(
      row_mask, row_group_indices, _input_columns, _row_mask_offset, stream);
  }

  prepare_data(read_mode::READ_ALL, row_group_indices, column_chunk_data, data_page_mask);

  return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::PAYLOAD_COLUMNS, row_mask);
}

table_with_metadata hybrid_scan_reader_impl::materialize_all_columns(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  prepare_materialization(
    read_columns_mode::ALL_COLUMNS, row_group_indices.size(), options, stream, mr);

  // Convert the input expression (must be done after prepare_materialization)
  _expr_conv = build_converted_expression(options);

  prepare_data(read_mode::READ_ALL, row_group_indices, column_chunk_data, {});

  // Use the main reader's function
  auto result = reader_impl::read_chunk_internal(read_mode::READ_ALL);

  // base read_chunk_internal() does not update the output chunk produced flag
  _output_chunk_produced = true;

  return result;
}

void hybrid_scan_reader_impl::setup_chunking_for_filter_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(options.get_filter().has_value(), "Empty input filter expression encountered");
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");

  prepare_materialization(
    read_columns_mode::FILTER_COLUMNS, row_group_indices.size(), options, stream, mr);

  _input_pass_read_limit   = pass_read_limit;
  _output_chunk_read_limit = chunk_read_limit;

  // Convert the input expression (must be done after prepare_materialization)
  _expr_conv = build_converted_expression(options);

  // Return early if all rows are pruned
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::CHUNKED_READ, empty_row_groups, {}, {});
    // Set correct number of input row groups to the output metadata
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return;
  }

  auto data_page_mask = thrust::host_vector<bool>{};
  if (mask_data_pages == use_data_page_mask::YES) {
    data_page_mask = _extended_metadata->compute_data_page_mask(
      row_mask, row_group_indices, _input_columns, _row_mask_offset, stream);
  }

  prepare_data(read_mode::CHUNKED_READ, row_group_indices, column_chunk_data, data_page_mask);
}

table_with_metadata hybrid_scan_reader_impl::materialize_filter_columns_chunk(
  cudf::mutable_column_view& row_mask)
{
  CUDF_EXPECTS(_file_preprocessed, "Chunking for filter columns not yet setup");

  // Reset the output buffers to their original states (right after reader construction).
  // Don't need to do it if we read the file all at once.
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes() and
      not is_first_output_chunk()) {
    _output_buffers.resize(0);
    for (auto const& buff : _output_buffers_template) {
      _output_buffers.emplace_back(cudf::io::detail::inline_column_buffer::empty_like(buff));
    }
  }

  prepare_data(read_mode::CHUNKED_READ, {}, {}, {});
  return read_chunk_internal(read_mode::CHUNKED_READ, read_columns_mode::FILTER_COLUMNS, row_mask);
}

void hybrid_scan_reader_impl::setup_chunking_for_payload_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");
  CUDF_EXPECTS(row_mask.null_count() == 0,
               "Row mask must not have any nulls when materializing payload column");

  prepare_materialization(
    read_columns_mode::PAYLOAD_COLUMNS, row_group_indices.size(), options, stream, mr);

  _input_pass_read_limit   = pass_read_limit;
  _output_chunk_read_limit = chunk_read_limit;

  // Return early if all rows are pruned
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::CHUNKED_READ, empty_row_groups, {}, {});
    // Set correct number of input row groups to the output metadata
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return;
  }

  auto data_page_mask = thrust::host_vector<bool>{};
  if (not row_mask.is_empty() and mask_data_pages == use_data_page_mask::YES) {
    data_page_mask = _extended_metadata->compute_data_page_mask(
      row_mask, row_group_indices, _input_columns, _row_mask_offset, stream);
  }

  prepare_data(read_mode::CHUNKED_READ, row_group_indices, column_chunk_data, data_page_mask);
}

void hybrid_scan_reader_impl::setup_chunking_for_payload_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  std::span<std::vector<cudf::device_span<uint8_t const>> const> page_data_per_source,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(_pending_payload_page_io_plan.has_value(),
               "No pending payload page I/O plan to consume");
  // Consume first so a failed setup cannot accidentally reuse stale pointer/range mappings.
  auto plan = std::move(_pending_payload_page_io_plan.value());
  _pending_payload_page_io_plan.reset();

  CUDF_EXPECTS(plan.mask_data_pages == mask_data_pages,
               "Payload setup page-mask option does not match its pending plan");
  auto const setup_row_groups =
    std::vector<std::vector<size_type>>{row_group_indices.begin(), row_group_indices.end()};
  CUDF_EXPECTS(plan.row_group_indices == setup_row_groups,
               "Payload setup row groups do not match the pending page I/O plan");

  reset_column_selection();
  select_columns(read_columns_mode::PAYLOAD_COLUMNS, options);
  auto selected_schemas = std::vector<int>{};
  selected_schemas.reserve(_input_columns.size());
  std::transform(_input_columns.begin(),
                 _input_columns.end(),
                 std::back_inserter(selected_schemas),
                 [](auto const& col) { return col.schema_idx; });
  CUDF_EXPECTS(selected_schemas == plan.column_schema_indices,
               "Payload column selection does not match the pending page I/O plan");

  CUDF_EXPECTS(page_data_per_source.size() == plan.source_ranges.size(),
               "Fetched payload source count does not match the pending plan");
  for (std::size_t source_idx = 0; source_idx < page_data_per_source.size(); ++source_idx) {
    CUDF_EXPECTS(page_data_per_source[source_idx].size() == plan.source_ranges[source_idx].size(),
                 "Fetched payload range count does not match the pending plan");
    for (std::size_t range_idx = 0; range_idx < page_data_per_source[source_idx].size();
         ++range_idx) {
      auto const& data  = page_data_per_source[source_idx][range_idx];
      auto const& range = plan.source_ranges[source_idx][range_idx];
      CUDF_EXPECTS(std::cmp_equal(data.size(), range.size()),
                   "Fetched payload span size does not match its planned byte range");
      CUDF_EXPECTS(data.size() == 0 or data.data() != nullptr,
                   "Fetched payload span has a null data pointer");
    }
  }

  if (not plan.sparse) {
    auto flat_chunk_data = std::vector<cudf::device_span<uint8_t const>>{};
    auto const span_count =
      std::accumulate(page_data_per_source.begin(),
                      page_data_per_source.end(),
                      std::size_t{0},
                      [](auto sum, auto const& spans) { return sum + spans.size(); });
    flat_chunk_data.reserve(span_count);
    for (auto const& source_data : page_data_per_source) {
      flat_chunk_data.insert(flat_chunk_data.end(), source_data.begin(), source_data.end());
    }
    setup_chunking_for_payload_columns(chunk_read_limit,
                                       pass_read_limit,
                                       row_group_indices,
                                       row_mask,
                                       mask_data_pages,
                                       flat_chunk_data,
                                       options,
                                       stream,
                                       mr);
    return;
  }

  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");
  CUDF_EXPECTS(row_mask.null_count() == 0,
               "Row mask must not have any nulls when materializing payload column");

  prepare_materialization(
    read_columns_mode::PAYLOAD_COLUMNS, row_group_indices.size(), options, stream, mr);

  _input_pass_read_limit   = pass_read_limit;
  _output_chunk_read_limit = chunk_read_limit;

  // Preserve the existing all-rows-pruned setup path. An all-false page plan has no byte ranges.
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::CHUNKED_READ, empty_row_groups, {}, {});
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return;
  }

  _sparse_page_spans.clear();
  _sparse_page_spans.reserve(plan.page_mappings.size());
  for (auto const& mapping : plan.page_mappings) {
    if (not mapping.fetched) {
      _sparse_page_spans.emplace_back();
      continue;
    }
    CUDF_EXPECTS(std::cmp_less(mapping.source_idx, page_data_per_source.size()),
                 "Sparse page mapping has an invalid source index");
    auto const& source_data = page_data_per_source[mapping.source_idx];
    CUDF_EXPECTS(mapping.range_idx < source_data.size(),
                 "Sparse page mapping has an invalid range index");
    auto const& range_data = source_data[mapping.range_idx];
    CUDF_EXPECTS(mapping.range_offset <= range_data.size() and
                   mapping.size <= range_data.size() - mapping.range_offset,
                 "Sparse page mapping exceeds its fetched range");
    _sparse_page_spans.emplace_back(range_data.data() + mapping.range_offset, mapping.size);
  }
  _sparse_resident_bytes_per_chunk     = std::move(plan.resident_bytes_per_chunk);
  _sparse_dictionary_present_per_chunk = std::move(plan.dictionary_present_per_chunk);
  _sparse_page_io                      = true;

  prepare_data(read_mode::CHUNKED_READ, row_group_indices, {}, plan.data_page_mask);
}

table_with_metadata hybrid_scan_reader_impl::materialize_payload_columns_chunk(
  cudf::column_view const& row_mask)
{
  CUDF_EXPECTS(_file_preprocessed, "Chunking for payload columns not yet setup");

  // Reset the output buffers to their original states (right after reader construction).
  // Don't need to do it if we read the file all at once.
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes() and
      not is_first_output_chunk()) {
    _output_buffers.resize(0);
    for (auto const& buff : _output_buffers_template) {
      _output_buffers.emplace_back(cudf::io::detail::inline_column_buffer::empty_like(buff));
    }
  }
  prepare_data(read_mode::CHUNKED_READ, {}, {}, {});
  return read_chunk_internal(read_mode::CHUNKED_READ, read_columns_mode::PAYLOAD_COLUMNS, row_mask);
}

void hybrid_scan_reader_impl::setup_chunking_for_all_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  prepare_materialization(
    read_columns_mode::ALL_COLUMNS, row_group_indices.size(), options, stream, mr);

  _input_pass_read_limit   = pass_read_limit;
  _output_chunk_read_limit = chunk_read_limit;

  // Convert the input expression (must be done after column selection)
  _expr_conv = build_converted_expression(options);

  prepare_data(read_mode::CHUNKED_READ, row_group_indices, column_chunk_data, {});
}

table_with_metadata hybrid_scan_reader_impl::materialize_all_columns_chunk()
{
  CUDF_EXPECTS(_file_preprocessed, "Chunking for all columns not yet setup");

  // Reset the output buffers to their original states (right after reader construction).
  // Don't need to do it if we read the file all at once.
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes() and
      not is_first_output_chunk()) {
    _output_buffers.resize(0);
    for (auto const& buff : _output_buffers_template) {
      _output_buffers.emplace_back(cudf::io::detail::inline_column_buffer::empty_like(buff));
    }
  }
  prepare_data(read_mode::CHUNKED_READ, {}, {}, {});

  // Use the main reader's function for reading all columns
  auto result = reader_impl::read_chunk_internal(read_mode::CHUNKED_READ);

  // base read_chunk_internal() does not update the output chunk produced flag
  _output_chunk_produced = true;

  return result;
}

std::pair<std::vector<std::vector<cudf::size_type>>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::construct_row_group_passes(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::size_t total_row_groups,
  std::size_t pass_read_limit) const
{
  CUDF_EXPECTS(
    total_row_groups > 0, "Empty input row group indices encountered", std::invalid_argument);

  CUDF_EXPECTS(row_group_indices.size() == _extended_metadata->get_num_sources(),
               "Mismatch in the number of row group indices vectors and the number of input "
               "datasources",
               std::invalid_argument);

  if (pass_read_limit == 0) {
    return {
      std::vector<std::vector<cudf::size_type>>{row_group_indices.begin(), row_group_indices.end()},
      std::vector<cudf::size_type>{}};
  }

  CUDF_EXPECTS(
    pass_read_limit > 0, "Pass read limit must be greater than 0", std::invalid_argument);

  auto row_groups_info = std::vector<row_group_info>{};
  row_groups_info.reserve(total_row_groups);
  size_t start_row = 0;
  std::for_each(cuda::counting_iterator<cudf::size_type>(0),
                cuda::counting_iterator<cudf::size_type>(row_group_indices.size()),
                [&](auto const source_index) {
                  auto const& src_row_groups = row_group_indices[source_index];
                  std::transform(
                    src_row_groups.begin(),
                    src_row_groups.end(),
                    std::back_inserter(row_groups_info),
                    [&](auto const rg_index) {
                      auto const& row_group =
                        _extended_metadata->get_row_group(rg_index, source_index);
                      auto const [compressed_size, total_size, num_rows, max_leaf_values] =
                        _extended_metadata->get_row_group_properties(row_group);
                      auto rg_info = row_group_info{.index               = rg_index,
                                                    .start_row           = start_row,
                                                    .unadjusted_num_rows = num_rows,
                                                    .source_index        = source_index,
                                                    .compressed_size     = compressed_size,
                                                    .max_leaf_values     = max_leaf_values};
                      start_row += num_rows;
                      return rg_info;
                    });
                });

  auto const comp_read_limit = static_cast<std::size_t>(
    pass_read_limit * cudf::io::parquet::detail::input_limit_compression_reserve);

  auto const pass_data =
    cudf::io::parquet::detail::compute_row_group_passes(row_groups_info, comp_read_limit, 0);

  // Convert offset-based pass boundaries back to vectors of row group indices
  auto const& offsets = pass_data.pass_row_group_offsets;
  auto passes         = std::vector<std::vector<cudf::size_type>>{};
  passes.reserve(offsets.size() - 1);
  auto row_group_source_map       = std::vector<cudf::size_type>{};
  auto const has_multiple_sources = row_group_indices.size() > 1;
  if (has_multiple_sources) { row_group_source_map.reserve(row_groups_info.size()); }
  std::transform(offsets.begin(),
                 offsets.end() - 1,
                 offsets.begin() + 1,
                 std::back_inserter(passes),
                 [&](auto const start, auto const end) {
                   auto pass = std::vector<cudf::size_type>{};
                   pass.reserve(end - start);
                   std::for_each(row_groups_info.begin() + start,
                                 row_groups_info.begin() + end,
                                 [&](auto const& rg_info) {
                                   pass.emplace_back(rg_info.index);
                                   if (has_multiple_sources) {
                                     row_group_source_map.emplace_back(rg_info.source_index);
                                   }
                                 });
                   return pass;
                 });
  return {std::move(passes), std::move(row_group_source_map)};
}

bool hybrid_scan_reader_impl::has_next_table_chunk()
{
  CUDF_EXPECTS(_file_preprocessed, "Chunking not yet setup");
  prepare_data(read_mode::CHUNKED_READ, {}, {}, {});

  // current_input_pass will only be incremented to be == num_passes after
  // the last chunk in the last subpass in the last pass has been returned
  // if not has_more_work then check if this is the first pass in an empty
  // table and return true so it could be read once.
  return has_more_work() or is_first_output_chunk();
}

void hybrid_scan_reader_impl::reset_internal_state()
{
  _row_mask_offset   = 0;
  _file_itm_data     = file_intermediate_data{};
  _file_preprocessed = false;
  _has_offset_index  = false;
  _pass_itm_data.reset();
  _pass_page_mask.clear();
  _subpass_page_mask.reset();
  _output_metadata.reset();
  _sparse_page_spans.clear();
  _sparse_resident_bytes_per_chunk.clear();
  _sparse_dictionary_present_per_chunk.clear();
  _sparse_page_io = false;

  _options.timestamp_type = cudf::data_type{};
  _options.decimal_width  = type_id::EMPTY;
  _options.num_rows       = std::nullopt;
  _options.row_group_indices.clear();
  _options.use_jit_filter              = false;
  _options.case_sensitive_names        = true;
  _options.prepend_source_index_column = false;
  _options.prepend_row_index_column    = false;

  _num_sources             = 0;
  _input_pass_read_limit   = 0;
  _output_chunk_read_limit = 0;
  _strings_to_categorical  = false;
  _reader_column_schema.reset();
  _expr_conv = named_to_reference_converter{};
  _mr        = cudf::get_current_device_resource_ref();
}

void hybrid_scan_reader_impl::initialize_reader_config(parquet_reader_options const& options)
{
  // Strings may be returned as either string or categorical columns
  _strings_to_categorical = options.is_enabled_convert_strings_to_categories();

  _options.timestamp_type              = cudf::data_type{options.get_timestamp_type().id()};
  _options.decimal_width               = options.get_decimal_width();
  _options.use_jit_filter              = options.is_enabled_use_jit_filter();
  _options.case_sensitive_names        = options.is_enabled_case_sensitive_names();
  _options.prepend_source_index_column = options.is_enabled_prepend_source_index_column();
  _options.prepend_row_index_column    = options.is_enabled_prepend_row_index_column();
}

void hybrid_scan_reader_impl::initialize_options(parquet_reader_options const& options,
                                                 std::size_t num_sources,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  // Binary columns can be read as binary or strings
  _reader_column_schema = options.get_column_schema();

  _num_sources = num_sources;

  // CUDA stream to use for internal operations
  _stream = stream;

  // Device memory resource to use for allocations
  _mr = mr;
}

named_to_reference_converter hybrid_scan_reader_impl::build_converted_expression(
  parquet_reader_options const& options)
{
  if (not options.get_filter().has_value()) { return named_to_reference_converter{}; }

  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(),
                                                metadata,
                                                _extended_metadata->get_schema_tree(),
                                                options,
                                                options.is_enabled_case_sensitive_names());
  CUDF_EXPECTS(expr_conv.get_converted_expr().has_value(),
               "Columns names in filter expression must be convertible to index references");
  return expr_conv;
}

void hybrid_scan_reader_impl::prepare_data(
  read_mode mode,
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  host_span<bool const> data_page_mask)
{
  // if we have not preprocessed at the whole-file level, do that now
  if (not _file_preprocessed) {
    // setup file level information
    // - read row group information
    // - setup information on (parquet) chunks
    // - compute schedule of input passes
    prepare_row_groups(read_mode::READ_ALL, row_group_indices);
  }

  // handle any chunking work (ratcheting through the subpasses and chunks within
  // our current pass) if in bounds
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes()) {
    handle_chunking(mode, column_chunk_data, data_page_mask);
  }
}

template <typename RowMaskView>
table_with_metadata hybrid_scan_reader_impl::read_chunk_internal(
  read_mode mode, read_columns_mode read_columns_mode, RowMaskView row_mask)
{
  // If `_output_metadata` has been constructed, just copy it over.
  auto out_metadata = _output_metadata ? table_metadata{*_output_metadata} : table_metadata{};
  out_metadata.schema_info.resize(_output_buffers.size());

  // output cudf columns as determined by the top level schema
  auto out_columns = std::vector<std::unique_ptr<column>>{};
  out_columns.reserve(_output_buffers.size());

  // Copy number of total input row groups and number of surviving row groups from predicate
  // pushdown.
  out_metadata.num_input_row_groups = _file_itm_data.num_input_row_groups;
  // Copy the number surviving row groups from each predicate pushdown only if the filter has value
  if (_expr_conv.get_converted_expr().has_value()) {
    out_metadata.num_row_groups_after_stats_filter =
      _file_itm_data.surviving_row_groups.after_stats_filter;
    out_metadata.num_row_groups_after_bloom_filter =
      _file_itm_data.surviving_row_groups.after_bloom_filter;
  }

  // no work to do (this can happen on the first pass if we have no rows to read)
  if (!has_more_work()) {
    // Empty dataframe case: Simply initialize to a list of zeros
    out_metadata.num_rows_per_source =
      std::vector<std::size_t>(_file_itm_data.num_rows_per_source.size(), 0);

    // Finalize output
    return finalize_output(read_columns_mode, out_metadata, out_columns, row_mask);
  }

  auto& pass            = *_pass_itm_data;
  auto& subpass         = *pass.subpass;
  auto const& read_info = subpass.output_chunk_read_info[subpass.current_output_chunk];

  // computes:
  // PageNestingInfo::batch_size for each level of nesting, for each page, taking row bounds into
  // account. PageInfo::skipped_values, which tells us where to start decoding in the input to
  // respect the user bounds. It is only necessary to do this second pass if uses_custom_row_bounds
  // is set (if the user has specified artificial bounds).
  if (uses_custom_row_bounds(mode)) {
    compute_page_sizes(subpass.pages,
                       pass.chunks,
                       subpass_page_mask_span(),
                       read_info.skip_rows,
                       read_info.num_rows,
                       false,  // num_rows is already computed
                       pass.level_type_size,
                       _stream);
  }

  // preprocess strings
  preprocess_chunk_strings(mode, read_info);

  // Allocate memory buffers for the output columns.
  allocate_columns(mode, read_info.skip_rows, read_info.num_rows);

  // Parse data into the output buffers.
  decode_page_data(mode, read_info.skip_rows, read_info.num_rows);

  // Create the final output cudf columns.
  for (std::size_t i = 0; i < _output_buffers.size(); ++i) {
    auto metadata           = _reader_column_schema.has_value()
                                ? std::make_optional<reader_column_schema>((*_reader_column_schema)[i])
                                : std::nullopt;
    auto const& schema      = _extended_metadata->get_schema(_output_column_schemas[i]);
    auto const logical_type = schema.logical_type.value_or(LogicalType{});
    // FIXED_LEN_BYTE_ARRAY never read as string.
    // TODO: if we ever decide that the default reader behavior is to treat unannotated BINARY
    // as binary and not strings, this test needs to change.
    if (schema.type == Type::FIXED_LEN_BYTE_ARRAY and logical_type.type != LogicalType::DECIMAL) {
      metadata = std::make_optional<reader_column_schema>();
      metadata->set_convert_binary_to_strings(false);
      metadata->set_type_length(schema.type_length);
    }
    // Only construct `out_metadata` if `_output_metadata` has not been cached.
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info[i];
      out_columns.emplace_back(make_column(_output_buffers[i], &col_name, metadata, _stream));
    } else {
      out_columns.emplace_back(make_column(_output_buffers[i], nullptr, metadata, _stream));
    }
  }

  out_columns =
    cudf::structs::detail::enforce_null_consistency(std::move(out_columns), _stream, _mr);

  // Compute the output number of rows per source
  if (mode == read_mode::CHUNKED_READ) {
    out_metadata.num_rows_per_source =
      calculate_output_num_rows_per_source(read_info.skip_rows, read_info.num_rows);
  } else {
    // Move is okay here as we are reading in one go.
    out_metadata.num_rows_per_source = std::move(_file_itm_data.num_rows_per_source);
  }

  // Add empty columns if needed. Filter output columns based on filter.
  return finalize_output(read_columns_mode, out_metadata, out_columns, row_mask);
}

template <typename RowMaskView>
table_with_metadata hybrid_scan_reader_impl::finalize_output(
  read_columns_mode read_columns_mode,
  table_metadata& out_metadata,
  std::vector<std::unique_ptr<column>>& out_columns,
  RowMaskView row_mask)
{
  // Create empty columns as needed (this can happen if we've ended up with no actual data to
  // read)
  for (std::size_t i = out_columns.size(); i < _output_buffers.size(); ++i) {
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info[i];
      out_columns.emplace_back(io::detail::empty_like(_output_buffers[i], &col_name, _stream, _mr));
    } else {
      out_columns.emplace_back(io::detail::empty_like(_output_buffers[i], nullptr, _stream, _mr));
    }
  }

  if (!_output_metadata) {
    populate_metadata(out_metadata);
    // Finally, save the output table metadata into `_output_metadata` for reuse next time.
    _output_metadata = std::make_unique<table_metadata>(out_metadata);
  }

  // Row-range of the current output chunk relative to the current row group selection.
  auto const read_info =
    (_file_itm_data._current_input_pass < _file_itm_data.num_passes())
      ? _pass_itm_data->subpass
          ->output_chunk_read_info[_pass_itm_data->subpass->current_output_chunk]
      : cudf::io::parquet::detail::row_range{0, 0};

  // advance output chunk/subpass/pass info for non-empty tables if and only if we are in bounds
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes()) {
    auto& pass    = *_pass_itm_data;
    auto& subpass = *pass.subpass;
    subpass.current_output_chunk++;
  }

  // increment the output chunk count
  _file_itm_data._output_chunk_count++;

  apply_decimal_width_cast(out_columns);

  // Prepend the source and row index columns to filter columns only
  if (read_columns_mode == read_columns_mode::FILTER_COLUMNS) {
    if (_options.prepend_row_index_column) {
      out_columns.emplace(out_columns.begin(),
                          synthesize_row_index_column(read_info, _stream, _mr));
      out_metadata.schema_info.emplace(out_metadata.schema_info.begin(),
                                       column_name_info{.name = "row_index", .is_nullable = false});
    }
    if (_options.prepend_source_index_column) {
      out_columns.emplace(
        out_columns.begin(),
        synthesize_source_index_column(out_metadata.num_rows_per_source, _stream, _mr));
      out_metadata.schema_info.emplace(
        out_metadata.schema_info.begin(),
        column_name_info{.name = "source_index", .is_nullable = false});
    }
  }

  // Create a table from the output columns.
  auto read_table = std::make_unique<table>(std::move(out_columns));

  // If the input row mask is empty, all rows are pruned anyway.
  if (row_mask.is_empty()) {
    _output_chunk_produced = true;
    return {std::move(read_table), std::move(out_metadata)};
  }

  CUDF_EXPECTS(row_mask.type().id() == type_id::BOOL8, "Input row mask must be a boolean column");

  // Get the current row mask offset
  auto const mask_offset = _row_mask_offset;
  // Update the row mask offset and the output chunk produced flag
  _row_mask_offset += read_table->num_rows();
  _output_chunk_produced = true;

  // Clear the number of rows per source as it is not valid after filtering
  out_metadata.num_rows_per_source.clear();

  // For filter columns, apply the filter expression and update the input row mask
  if constexpr (std::is_same_v<RowMaskView, cudf::mutable_column_view>) {
    CUDF_EXPECTS(read_columns_mode == read_columns_mode::FILTER_COLUMNS, "Invalid read mode");

    // Compute the final filter expression incorporating any column reference offsets in _expr_conv
    auto const final_filter      = compute_offset_filter();
    auto const final_filter_expr = final_filter.get_converted_expr();

    auto final_row_mask = cudf::detail::compute_column(*read_table,
                                                       final_filter_expr.value().get(),
                                                       _stream,
                                                       cudf::get_current_device_resource_ref());
    CUDF_EXPECTS(final_row_mask->view().type().id() == type_id::BOOL8,
                 "Predicate filter should return a boolean");

    // Apply the final row mask to get the final output table
    auto output_table = cudf::detail::apply_mask(
      read_table->view(), *final_row_mask, cudf::detail::mask_type::RETENTION, _stream, _mr);

    // Update the input row mask to reflect the final row mask.
    update_row_mask(final_row_mask->view(), row_mask, mask_offset, _stream);

    // Return the final output table and metadata
    return {std::move(output_table), std::move(out_metadata)};
  }
  // For payload columns, simply apply the input row mask to the table.
  else {
    CUDF_EXPECTS(read_columns_mode == read_columns_mode::PAYLOAD_COLUMNS, "Invalid read mode");

    CUDF_EXPECTS(mask_offset + read_table->num_rows() <= row_mask.size(),
                 "Encountered invalid sized row mask to apply");
    auto effective_row_mask =
      (read_table->num_rows() == row_mask.size())
        ? row_mask
        : cudf::split(row_mask, {mask_offset, mask_offset + read_table->num_rows()}, _stream)[1];
    auto output_table = cudf::detail::apply_mask(
      read_table->view(), effective_row_mask, cudf::detail::mask_type::RETENTION, _stream, _mr);
    return {std::move(output_table), std::move(out_metadata)};
  }
}

void hybrid_scan_reader_impl::set_pass_page_mask(std::span<bool const> data_page_mask)
{
  auto const& pass   = _pass_itm_data;
  auto const& chunks = pass->chunks;

  _pass_page_mask        = cudf::detail::make_empty_host_vector<bool>(pass->pages.size(), _stream);
  auto const num_columns = _input_columns.size();

  // Handle the empty page mask case
  if (data_page_mask.empty()) {
    std::fill(_pass_page_mask.begin(), _pass_page_mask.end(), true);
    return;
  }

  std::size_t num_inserted_data_pages = 0;
  std::for_each(
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator{_input_columns.size()},
    [&](auto col_idx) {
      for (std::size_t chunk_idx = col_idx; chunk_idx < chunks.size(); chunk_idx += num_columns) {
        // Number of data pages in this column chunk
        auto const num_data_pages_this_col_chunk = chunks[chunk_idx].num_data_pages;

        // Make sure we have enough page mask for this column chunk
        CUDF_EXPECTS(
          data_page_mask.size() >= num_inserted_data_pages + num_data_pages_this_col_chunk,
          "Encountered invalid data page mask size");

        // Sparse chunks omit dictionaries when every data page is pruned. The contiguous path
        // retains its existing conservative dictionary behavior.
        if (chunks[chunk_idx].num_dict_pages > 0) {
          auto const chunk_has_retained_page = std::any_of(
            data_page_mask.begin() + num_inserted_data_pages,
            data_page_mask.begin() + num_inserted_data_pages + num_data_pages_this_col_chunk,
            [](bool retained) { return retained; });
          _pass_page_mask.push_back(_sparse_page_io ? chunk_has_retained_page : true);
        }

        // Insert page mask for this column chunk
        _pass_page_mask.insert(
          _pass_page_mask.end(),
          data_page_mask.begin() + num_inserted_data_pages,
          data_page_mask.begin() + num_inserted_data_pages + num_data_pages_this_col_chunk);
        // Update the number of inserted data pages
        num_inserted_data_pages += num_data_pages_this_col_chunk;
      }
    });

  // Make sure we inserted exactly the number of pages for this pass
  CUDF_EXPECTS(_pass_page_mask.size() == pass->pages.size(),
               "Encountered mismatch in number of pass pages and page mask size");
}

}  // namespace cudf::io::parquet::experimental::detail
