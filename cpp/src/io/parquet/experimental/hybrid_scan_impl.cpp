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

#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
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
}

FileMetaData hybrid_scan_reader_impl::parquet_metadata() const
{
  return _metadata->parquet_metadata();
}

byte_range_info hybrid_scan_reader_impl::page_index_byte_range() const
{
  return _metadata->page_index_byte_range();
}

void hybrid_scan_reader_impl::setup_page_index(
  cudf::host_span<uint8_t const> page_index_bytes) const
{
  _metadata->setup_page_index(page_index_bytes);
}

void hybrid_scan_reader_impl::select_columns(read_mode read_mode,
                                             parquet_reader_options const& options)
{
  // Strings may be returned as either string or categorical columns
  auto const strings_to_categorical = options.is_enabled_convert_strings_to_categories();
  auto const use_pandas_metadata    = options.is_enabled_use_pandas_metadata();
  auto const timestamp_type_id      = options.get_timestamp_type().id();

  // Select only columns required by the filter
  if (read_mode == read_mode::FILTER_COLUMNS) {
    if (_is_filter_columns_selected) { return; }
    // list, struct, dictionary are not supported by AST filter yet.
    _filter_columns_names =
      cudf::io::parquet::detail::get_column_names_in_expression(options.get_filter(), {});
    // Select only filter columns using the base `select_columns` method
    std::tie(_input_columns, _output_buffers, _output_column_schemas) = _metadata->select_columns(
      _filter_columns_names, {}, use_pandas_metadata, strings_to_categorical, timestamp_type_id);

    _is_filter_columns_selected  = true;
    _is_payload_columns_selected = false;
  } else {
    if (_is_payload_columns_selected) { return; }

    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _metadata->select_payload_columns(options.get_columns(),
                                        _filter_columns_names,
                                        use_pandas_metadata,
                                        strings_to_categorical,
                                        timestamp_type_id);

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
  auto const num_row_groups = _metadata->get_num_row_groups();
  auto row_groups_indices   = std::vector<size_type>(num_row_groups);
  std::iota(row_groups_indices.begin(), row_groups_indices.end(), size_type{0});
  return row_groups_indices;
}

size_type hybrid_scan_reader_impl::total_rows_in_row_groups(
  cudf::host_span<std::vector<size_type> const> row_group_indices) const
{
  return _metadata->total_rows_in_row_groups(row_group_indices);
}

std::vector<std::vector<size_type>> hybrid_scan_reader_impl::filter_row_groups_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  CUDF_EXPECTS(options.get_filter().has_value(), "Encountered empty converted filter expression");

  select_columns(read_mode::FILTER_COLUMNS, options);

  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);
  CUDF_EXPECTS(expr_conv.get_converted_expr().has_value(),
               "Columns names in filter expression must be convertible to index references");
  auto output_dtypes = get_output_types(_output_buffers_template);

  return _metadata->filter_row_groups_with_stats(row_group_indices,
                                                 output_dtypes,
                                                 _output_column_schemas,
                                                 expr_conv.get_converted_expr(),
                                                 stream);
}

std::pair<std::vector<byte_range_info>, std::vector<byte_range_info>>
hybrid_scan_reader_impl::secondary_filters_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options)
{
  return {};
}

std::vector<std::vector<size_type>>
hybrid_scan_reader_impl::filter_row_groups_with_dictionary_pages(
  cudf::host_span<rmm::device_buffer> dictionary_page_data,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  return {};
}

std::vector<std::vector<size_type>> hybrid_scan_reader_impl::filter_row_groups_with_bloom_filters(
  cudf::host_span<rmm::device_buffer> bloom_filter_data,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  return {};
}

std::pair<std::unique_ptr<cudf::column>, std::vector<thrust::host_vector<bool>>>
hybrid_scan_reader_impl::filter_data_pages_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return {};
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::get_input_column_chunk_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices) const
{
  return {};
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::filter_column_chunks_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options)
{
  return {};
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::payload_column_chunks_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options)
{
  return {};
}

table_with_metadata hybrid_scan_reader_impl::materialize_filter_columns(
  cudf::host_span<thrust::host_vector<bool> const> data_page_mask,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::mutable_column_view row_mask,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  return {};
}

table_with_metadata hybrid_scan_reader_impl::materialize_payload_columns(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::column_view row_mask,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  return {};
}

void hybrid_scan_reader_impl::populate_metadata(table_metadata& out_metadata) const
{
  // Return column names
  out_metadata.schema_info.resize(_output_buffers.size());
  for (size_t i = 0; i < _output_column_schemas.size(); i++) {
    auto const& schema               = _metadata->get_schema(_output_column_schemas[i]);
    out_metadata.schema_info[i].name = schema.name;
    out_metadata.schema_info[i].is_nullable =
      schema.repetition_type != cudf::io::parquet::FieldRepetitionType::REQUIRED;
  }

  // Return user metadata
  out_metadata.per_file_user_data = _metadata->get_key_value_metadata();
  out_metadata.user_data          = {out_metadata.per_file_user_data[0].begin(),
                                     out_metadata.per_file_user_data[0].end()};
}

bool hybrid_scan_reader_impl::has_more_work() const
{
  return _file_itm_data.num_passes() > 0 &&
         _file_itm_data._current_input_pass < _file_itm_data.num_passes();
}

bool hybrid_scan_reader_impl::is_first_output_chunk() const
{
  return _file_itm_data._output_chunk_count == 0;
}

}  // namespace cudf::io::parquet::experimental::detail
