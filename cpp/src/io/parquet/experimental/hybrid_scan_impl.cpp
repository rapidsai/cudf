/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "hybrid_scan_impl.hpp"

#include "cudf/io/text/byte_range_info.hpp"
#include "hybrid_scan_helpers.hpp"
#include "io/parquet/reader_impl_chunking_utils.cuh"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <bitset>
#include <iterator>
#include <limits>
#include <numeric>

namespace cudf::io::parquet::experimental::detail {

using io::detail::inline_column_buffer;
using parquet::detail::ColumnChunkDesc;
using parquet::detail::decode_kernel_mask;
using parquet::detail::file_intermediate_data;
using parquet::detail::named_to_reference_converter;
using parquet::detail::PageInfo;
using parquet::detail::PageNestingDecodeInfo;
using text::byte_range_info;

namespace {
// Tests the passed in logical type for a FIXED_LENGTH_BYTE_ARRAY column to see if it should
// be treated as a string. Currently the only logical type that has special handling is DECIMAL.
// Other valid types in the future would be UUID (still treated as string) and FLOAT16 (which
// for now would also be treated as a string).
[[maybe_unused]] inline bool is_treat_fixed_length_as_string(
  std::optional<LogicalType> const& logical_type)
{
  if (!logical_type.has_value()) { return true; }
  return logical_type->type != LogicalType::DECIMAL;
}

[[nodiscard]] std::vector<cudf::data_type> get_output_types(
  cudf::host_span<inline_column_buffer const> output_buffer_template)
{
  std::vector<cudf::data_type> output_dtypes;
  output_dtypes.reserve(output_buffer_template.size());
  std::transform(output_buffer_template.begin(),
                 output_buffer_template.end(),
                 std::back_inserter(output_dtypes),
                 [](auto const& col) { return col.type; });
  return output_dtypes;
}

}  // namespace

hybrid_scan_reader_impl::hybrid_scan_reader_impl(cudf::host_span<uint8_t const> footer_bytes,
                                                 parquet_reader_options const& options)
{
  // Open and parse the source dataset metadata
  _metadata = std::make_unique<aggregate_reader_metadata>(
    footer_bytes,
    options.is_enabled_use_arrow_schema(),
    options.get_columns().has_value() and options.is_enabled_allow_mismatched_pq_schemas());

  _extended_metadata = static_cast<aggregate_reader_metadata*>(_metadata.get());
}

FileMetaData hybrid_scan_reader_impl::parquet_metadata() const
{
  return _extended_metadata->parquet_metadata();
}

byte_range_info hybrid_scan_reader_impl::page_index_byte_range() const
{
  return _extended_metadata->page_index_byte_range();
}

void hybrid_scan_reader_impl::setup_page_index(
  cudf::host_span<uint8_t const> page_index_bytes) const
{
  _extended_metadata->setup_page_index(page_index_bytes);
}

void hybrid_scan_reader_impl::select_columns(read_columns_mode read_columns_mode,
                                             parquet_reader_options const& options)
{
  // Select only columns required by the filter
  if (read_columns_mode == read_columns_mode::FILTER_COLUMNS) {
    if (_is_filter_columns_selected) { return; }
    // list, struct, dictionary are not supported by AST filter yet.
    _filter_columns_names =
      cudf::io::parquet::detail::get_column_names_in_expression(options.get_filter(), {});
    // Select only filter columns using the base `select_columns` method
    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _extended_metadata->select_columns(_filter_columns_names,
                                         {},
                                         _use_pandas_metadata,
                                         _strings_to_categorical,
                                         _options.timestamp_type.id());

    _is_filter_columns_selected  = true;
    _is_payload_columns_selected = false;
  } else {
    if (_is_payload_columns_selected) { return; }

    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _extended_metadata->select_payload_columns(options.get_columns(),
                                                 _filter_columns_names,
                                                 _use_pandas_metadata,
                                                 _strings_to_categorical,
                                                 _options.timestamp_type.id());

    _is_payload_columns_selected = true;
    _is_filter_columns_selected  = false;
  }

  CUDF_EXPECTS(_input_columns.size() > 0 and _output_buffers.size() > 0, "No columns selected");

  // Clear the output buffers templates
  _output_buffers_template.clear();

  // Save the states of the output buffers for reuse.
  std::transform(_output_buffers.begin(),
                 _output_buffers.end(),
                 std::back_inserter(_output_buffers_template),
                 [](auto const& buff) { return inline_column_buffer::empty_like(buff); });
}

std::vector<size_type> hybrid_scan_reader_impl::all_row_groups(
  parquet_reader_options const& options) const
{
  auto const num_row_groups = _extended_metadata->get_num_row_groups();
  auto row_groups_indices   = std::vector<size_type>(num_row_groups);
  std::iota(row_groups_indices.begin(), row_groups_indices.end(), size_type{0});
  return row_groups_indices;
}

size_type hybrid_scan_reader_impl::total_rows_in_row_groups(
  cudf::host_span<std::vector<size_type> const> row_group_indices) const
{
  return _extended_metadata->total_rows_in_row_groups(row_group_indices);
}

std::vector<std::vector<size_type>> hybrid_scan_reader_impl::filter_row_groups_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  CUDF_EXPECTS(options.get_filter().has_value(), "Encountered empty converted filter expression");

  select_columns(read_columns_mode::FILTER_COLUMNS, options);

  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);
  CUDF_EXPECTS(expr_conv.get_converted_expr().has_value(),
               "Columns names in filter expression must be convertible to index references");
  auto output_dtypes = get_output_types(_output_buffers_template);

  return _extended_metadata->filter_row_groups_with_stats(row_group_indices,
                                                          output_dtypes,
                                                          _output_column_schemas,
                                                          expr_conv.get_converted_expr().value(),
                                                          stream);
}

std::pair<std::vector<byte_range_info>, std::vector<byte_range_info>>
hybrid_scan_reader_impl::secondary_filters_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  CUDF_EXPECTS(options.get_filter().has_value(), "Filter expression must not be empty");

  select_columns(read_columns_mode::FILTER_COLUMNS, options);

  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);
  CUDF_EXPECTS(expr_conv.get_converted_expr().has_value(),
               "Columns names in filter expression must be convertible to index references");
  auto output_dtypes = get_output_types(_output_buffers_template);

  auto const bloom_filter_bytes =
    _extended_metadata->get_bloom_filter_bytes(row_group_indices,
                                               output_dtypes,
                                               _output_column_schemas,
                                               expr_conv.get_converted_expr().value());
  auto const dictionary_page_bytes =
    _extended_metadata->get_dictionary_page_bytes(row_group_indices,
                                                  output_dtypes,
                                                  _output_column_schemas,
                                                  expr_conv.get_converted_expr().value());

  return {bloom_filter_bytes, dictionary_page_bytes};
}

std::vector<std::vector<size_type>>
hybrid_scan_reader_impl::filter_row_groups_with_dictionary_pages(
  cudf::host_span<rmm::device_buffer> dictionary_page_data,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  CUDF_EXPECTS(options.get_filter().has_value(), "Encountered empty converted filter expression");

  select_columns(read_columns_mode::FILTER_COLUMNS, options);

  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);
  CUDF_EXPECTS(expr_conv.get_converted_expr().has_value(),
               "Columns names in filter expression must be convertible to index references");
  auto output_dtypes = get_output_types(_output_buffers_template);

  // Collect literal and operator pairs for each input column with an (in)equality predicate
  auto const [literals, operators] =
    dictionary_literals_collector{expr_conv.get_converted_expr().value().get(),
                                  static_cast<cudf::size_type>(output_dtypes.size())}
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
  cudf::host_span<rmm::device_buffer> bloom_filter_data,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  CUDF_EXPECTS(options.get_filter().has_value(), "Encountered empty converted filter expression");

  CUDF_FUNC_RANGE();

  select_columns(read_columns_mode::FILTER_COLUMNS, options);

  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);
  CUDF_EXPECTS(expr_conv.get_converted_expr().has_value(),
               "Columns names in filter expression must be convertible to index references");
  auto output_dtypes = get_output_types(_output_buffers_template);

  return _extended_metadata->filter_row_groups_with_bloom_filters(
    bloom_filter_data,
    row_group_indices,
    output_dtypes,
    _output_column_schemas,
    expr_conv.get_converted_expr().value(),
    stream);
}

std::unique_ptr<cudf::column> hybrid_scan_reader_impl::build_row_mask_with_page_index_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  CUDF_EXPECTS(options.get_filter().has_value(), "Encountered empty converted filter expression");

  CUDF_FUNC_RANGE();

  select_columns(read_columns_mode::FILTER_COLUMNS, options);

  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);
  CUDF_EXPECTS(expr_conv.get_converted_expr().has_value(),
               "Columns names in filter expression must be convertible to index references");
  auto output_dtypes = get_output_types(_output_buffers_template);

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
  cudf::host_span<std::vector<size_type> const> row_group_indices) const
{
  CUDF_FUNC_RANGE();

  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_row_groups =
    std::accumulate(row_group_indices.begin(),
                    row_group_indices.end(),
                    size_t{0},
                    [](size_t sum, auto const& row_groups) { return sum + row_groups.size(); });
  auto const num_chunks = num_row_groups * num_input_columns;

  // Association between each column chunk and its source
  auto chunk_source_map = std::vector<size_type>{};
  chunk_source_map.reserve(num_chunks);

  // Keep track of column chunk byte ranges
  auto column_chunk_byte_ranges = std::vector<byte_range_info>{};
  column_chunk_byte_ranges.reserve(num_chunks);

  std::for_each(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator(row_group_indices.size()),
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
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  CUDF_EXPECTS(options.get_filter().has_value(), "Encountered empty converted filter expression");

  select_columns(read_columns_mode::FILTER_COLUMNS, options);
  return get_input_column_chunk_byte_ranges(row_group_indices);
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::payload_column_chunks_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  CUDF_EXPECTS(options.get_filter().has_value(), "Encountered empty converted filter expression");

  select_columns(read_columns_mode::PAYLOAD_COLUMNS, options);
  return get_input_column_chunk_byte_ranges(row_group_indices);
}

table_with_metadata hybrid_scan_reader_impl::materialize_filter_columns(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::mutable_column_view row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  CUDF_EXPECTS(options.get_filter().has_value(), "Encountered empty converted filter expression");

  CUDF_FUNC_RANGE();

  reset_internal_state();

  table_metadata metadata;
  populate_metadata(metadata);
  _expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  CUDF_EXPECTS(_expr_conv.get_converted_expr().has_value(), "Filter expression must not be empty");

  initialize_options(row_group_indices, options, stream);

  select_columns(read_columns_mode::FILTER_COLUMNS, options);

  auto output_dtypes = get_output_types(_output_buffers_template);

  auto data_page_mask =
    (mask_data_pages == use_data_page_mask::YES)
      ? _extended_metadata->compute_data_page_mask(
          row_mask, row_group_indices, output_dtypes, _output_column_schemas, stream)
      : std::vector<std::vector<bool>>{};

  prepare_data(row_group_indices, std::move(column_chunk_buffers), data_page_mask, options);

  return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::FILTER_COLUMNS, row_mask);
}

table_with_metadata hybrid_scan_reader_impl::materialize_payload_columns(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::column_view row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  CUDF_EXPECTS(row_mask.null_count() == 0,
               "Row mask must not have any nulls when materializing payload column");

  CUDF_FUNC_RANGE();

  reset_internal_state();

  initialize_options(row_group_indices, options, stream);

  select_columns(read_columns_mode::PAYLOAD_COLUMNS, options);

  auto output_dtypes = get_output_types(_output_buffers_template);

  auto data_page_mask =
    (mask_data_pages == use_data_page_mask::YES)
      ? _extended_metadata->compute_data_page_mask(
          row_mask, row_group_indices, output_dtypes, _output_column_schemas, stream)
      : std::vector<std::vector<bool>>{};

  prepare_data(row_group_indices, std::move(column_chunk_buffers), data_page_mask, options);

  return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::PAYLOAD_COLUMNS, row_mask);
}

void hybrid_scan_reader_impl::reset_internal_state()
{
  _file_itm_data     = file_intermediate_data{};
  _file_preprocessed = false;
  _has_page_index    = false;
  _pass_itm_data.reset();
  _pass_page_mask.clear();
  _subpass_page_mask.clear();
  _output_metadata.reset();
  _options.timestamp_type = cudf::data_type{};
  _options.num_rows       = std::nullopt;
  _options.row_group_indices.clear();
  _num_sources             = 0;
  _input_pass_read_limit   = 0;
  _output_chunk_read_limit = 0;
  _strings_to_categorical  = false;
  _reader_column_schema.reset();
  _expr_conv = named_to_reference_converter(std::nullopt, table_metadata{});
}

void hybrid_scan_reader_impl::initialize_options(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  // Strings may be returned as either string or categorical columns
  _strings_to_categorical = options.is_enabled_convert_strings_to_categories();

  _options.timestamp_type = cudf::data_type{options.get_timestamp_type().id()};

  _use_pandas_metadata = options.is_enabled_use_pandas_metadata();

  // Binary columns can be read as binary or strings
  _reader_column_schema = options.get_column_schema();

  _num_sources = row_group_indices.size();

  // CUDA stream to use for internal operations
  _stream = stream;
}

void hybrid_scan_reader_impl::prepare_data(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::host_span<std::vector<bool> const> data_page_mask,
  parquet_reader_options const& options)
{
  // if we have not preprocessed at the whole-file level, do that now
  if (not _file_preprocessed) {
    // setup file level information
    // - read row group information
    // - setup information on (parquet) chunks
    // - compute schedule of input passes
    prepare_row_groups(row_group_indices, options);
  }

  // handle any chunking work (ratcheting through the subpasses and chunks within
  // our current pass) if in bounds
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes()) {
    handle_chunking(std::move(column_chunk_buffers), data_page_mask, options);
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
    // Check if number of rows per source should be included in output metadata.
    if (include_output_num_rows_per_source()) {
      // Empty dataframe case: Simply initialize to a list of zeros
      out_metadata.num_rows_per_source =
        std::vector<size_t>(_file_itm_data.num_rows_per_source.size(), 0);
    }

    // Finalize output
    return finalize_output(read_columns_mode, out_metadata, out_columns, row_mask);
  }

  auto& pass            = *_pass_itm_data;
  auto& subpass         = *pass.subpass;
  auto const& read_info = subpass.output_chunk_read_info[subpass.current_output_chunk];

  // Allocate memory buffers for the output columns.
  allocate_columns(mode, read_info.skip_rows, read_info.num_rows);

  // Parse data into the output buffers.
  decode_page_data(mode, read_info.skip_rows, read_info.num_rows);

  // Create the final output cudf columns.
  for (size_t i = 0; i < _output_buffers.size(); ++i) {
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

  // Check if number of rows per source should be included in output metadata.
  if (include_output_num_rows_per_source()) {
    // For chunked reading, compute the output number of rows per source
    if (mode == read_mode::CHUNKED_READ) {
      out_metadata.num_rows_per_source =
        calculate_output_num_rows_per_source(read_info.skip_rows, read_info.num_rows);
    }
    // Simply move the number of rows per file if reading all at once
    else {
      // Move is okay here as we are reading in one go.
      out_metadata.num_rows_per_source = std::move(_file_itm_data.num_rows_per_source);
    }
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
  for (size_t i = out_columns.size(); i < _output_buffers.size(); ++i) {
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

  // advance output chunk/subpass/pass info for non-empty tables if and only if we are in bounds
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes()) {
    auto& pass    = *_pass_itm_data;
    auto& subpass = *pass.subpass;
    subpass.current_output_chunk++;
  }

  // increment the output chunk count
  _file_itm_data._output_chunk_count++;

  // Create a table from the output columns.
  auto read_table = std::make_unique<table>(std::move(out_columns));

  // If reading filter columns, compute the predicate, apply it to the table, and update the input
  // row mask to reflect the final surviving rows.
  if constexpr (std::is_same_v<RowMaskView, cudf::mutable_column_view>) {
    CUDF_EXPECTS(read_columns_mode == read_columns_mode::FILTER_COLUMNS, "Invalid read mode");
    // Apply the row selection predicate on the read table to get the final row mask
    auto final_row_mask =
      cudf::detail::compute_column(*read_table,
                                   _expr_conv.get_converted_expr().value().get(),
                                   _stream,
                                   cudf::get_current_device_resource_ref());
    CUDF_EXPECTS(final_row_mask->view().type().id() == type_id::BOOL8,
                 "Predicate filter should return a boolean");

    // Apply the final row mask to get the final output table
    auto output_table =
      cudf::detail::apply_boolean_mask(read_table->view(), *final_row_mask, _stream, _mr);

    // Update the input row mask to reflect the final row mask.
    update_row_mask(final_row_mask->view(), row_mask, _stream);

    // Return the final output table and metadata
    return {std::move(output_table), std::move(out_metadata)};
  }
  // Otherwise, simply apply the input row mask to the table.
  else {
    CUDF_EXPECTS(read_columns_mode == read_columns_mode::PAYLOAD_COLUMNS, "Invalid read mode");
    CUDF_EXPECTS(row_mask.type().id() == type_id::BOOL8,
                 "Predicate filter should return a boolean");
    CUDF_EXPECTS(row_mask.size() == read_table->num_rows(),
                 "Encountered invalid sized row mask to apply");
    auto output_table =
      cudf::detail::apply_boolean_mask(read_table->view(), row_mask, _stream, _mr);
    return {std::move(output_table), std::move(out_metadata)};
  }
}

void hybrid_scan_reader_impl::set_pass_page_mask(
  cudf::host_span<std::vector<bool> const> data_page_mask)
{
  CUDF_FUNC_RANGE();

  auto const& pass   = _pass_itm_data;
  auto const& chunks = pass->chunks;

  _pass_page_mask        = cudf::detail::make_empty_host_vector<bool>(pass->pages.size(), _stream);
  auto const num_columns = _input_columns.size();

  // Handle the empty page mask case
  if (data_page_mask.empty()) {
    std::fill(_pass_page_mask.begin(), _pass_page_mask.end(), true);
    return;
  }

  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(_input_columns.size()),
    [&](auto col_idx) {
      auto const& col_page_mask      = data_page_mask[col_idx];
      size_t num_inserted_data_pages = 0;

      for (size_t chunk_idx = col_idx; chunk_idx < chunks.size(); chunk_idx += num_columns) {
        // Insert a true value for each dictionary page
        if (chunks[chunk_idx].num_dict_pages > 0) { _pass_page_mask.push_back(true); }

        // Number of data pages in this column chunk
        auto const num_data_pages_this_col_chunk = chunks[chunk_idx].num_data_pages;

        // Make sure we have enough page mask for this column chunk
        CUDF_EXPECTS(
          col_page_mask.size() >= num_inserted_data_pages + num_data_pages_this_col_chunk,
          "Encountered invalid data page mask size");

        // Insert page mask for this column chunk
        _pass_page_mask.insert(
          _pass_page_mask.end(),
          col_page_mask.begin() + num_inserted_data_pages,
          col_page_mask.begin() + num_inserted_data_pages + num_data_pages_this_col_chunk);

        // Update the number of inserted data pages
        num_inserted_data_pages += num_data_pages_this_col_chunk;
      }
      // Make sure we inserted exactly the number of data pages for this column
      CUDF_EXPECTS(num_inserted_data_pages == col_page_mask.size(),
                   "Encountered mismatch in number of data pages and page mask size");
    });

  // Make sure we inserted exactly the number of pages for this pass
  CUDF_EXPECTS(_pass_page_mask.size() == pass->pages.size(),
               "Encountered mismatch in number of pass pages and page mask size");
}

}  // namespace cudf::io::parquet::experimental::detail
