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

#include "hybrid_scan_helpers.hpp"

#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <bitset>
#include <iterator>
#include <limits>
#include <numeric>

namespace cudf::experimental::io::parquet::detail {

impl::impl(cudf::host_span<uint8_t const> footer_bytes,
           cudf::host_span<uint8_t const> page_index_bytes,
           cudf::io::parquet_reader_options const& options)
{
  // Open and parse the source dataset metadata
  _metadata = std::make_unique<aggregate_reader_metadata>(
    footer_bytes,
    page_index_bytes,
    options.is_enabled_use_arrow_schema(),
    options.get_columns().has_value() and options.is_enabled_allow_mismatched_pq_schemas());

  // Strings may be returned as either string or categorical columns
  auto const strings_to_categorical = options.is_enabled_convert_strings_to_categories();

  // Select only columns required by the filter
  std::optional<std::vector<std::string>> filter_columns_names;
  if (options.get_filter().has_value() and options.get_columns().has_value()) {
    // list, struct, dictionary are not supported by AST filter yet.
    // extract columns not present in get_columns() & keep count to remove at end.
    filter_columns_names = cudf::io::parquet::detail::get_column_names_in_expression(
      options.get_filter(), *(options.get_columns()));
    _num_filter_only_columns = filter_columns_names->size();
  }

  // Only need to select columns if filter is available
  if (options.get_filter().has_value()) {
    if (not filter_columns_names.has_value()) {
      filter_columns_names =
        cudf::io::parquet::detail::get_column_names_in_expression(options.get_filter(), {});
    }

    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _metadata->select_filter_columns(filter_columns_names,
                                       options.is_enabled_use_pandas_metadata(),
                                       strings_to_categorical,
                                       options.get_timestamp_type().id());

    // Save the states of the output buffers for reuse.
    for (auto const& buff : _output_buffers) {
      _output_buffers_template.emplace_back(
        cudf::io::detail::inline_column_buffer::empty_like(buff));
    }
  }
}

std::vector<size_type> impl::get_valid_row_groups(
  cudf::io::parquet_reader_options const& options) const
{
  auto const num_row_groups = _metadata->get_num_row_groups();
  auto row_groups_indices   = std::vector<size_type>(num_row_groups);
  std::iota(row_groups_indices.begin(), row_groups_indices.end(), size_type{0});
  return row_groups_indices;
}

std::vector<std::vector<size_type>> impl::filter_row_groups_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  // Save the name to reference converter to extract output filter AST in
  // `preprocess_file()` and `finalize_output()`
  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  std::vector<data_type> output_dtypes;
  if (expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });
  }

  return _metadata->filter_row_groups_with_stats(row_group_indices,
                                                 output_dtypes,
                                                 _output_column_schemas,
                                                 expr_conv.get_converted_expr(),
                                                 stream);
}

std::pair<std::vector<cudf::io::text::byte_range_info>,
          std::vector<cudf::io::text::byte_range_info>>
impl::get_secondary_filters(cudf::host_span<std::vector<size_type> const> row_group_indices,
                            cudf::io::parquet_reader_options const& options) const
{
  // Save the name to reference converter to extract output filter AST in
  // `preprocess_file()` and `finalize_output()`
  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  std::vector<data_type> output_dtypes;
  if (expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });
  }

  auto const bloom_filter_bytes = _metadata->get_bloom_filter_bytes(
    row_group_indices, output_dtypes, _output_column_schemas, expr_conv.get_converted_expr());
  auto const dictionary_page_bytes = _metadata->get_dictionary_page_bytes(
    row_group_indices, output_dtypes, _output_column_schemas, expr_conv.get_converted_expr());

  return {bloom_filter_bytes, dictionary_page_bytes};
}

std::vector<std::vector<size_type>> impl::filter_row_groups_with_dictionary_pages(
  std::vector<rmm::device_buffer>& dictionary_page_data,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  // Save the name to reference converter to extract output filter AST in
  // `preprocess_file()` and `finalize_output()`
  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  std::vector<data_type> output_dtypes;
  if (expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });
  }

  return _metadata->filter_row_groups_with_dictionary_pages(dictionary_page_data,
                                                            row_group_indices,
                                                            output_dtypes,
                                                            _output_column_schemas,
                                                            expr_conv.get_converted_expr(),
                                                            stream);
}

std::vector<std::vector<size_type>> impl::filter_row_groups_with_bloom_filters(
  std::vector<rmm::device_buffer>& bloom_filter_data,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  // Save the name to reference converter to extract output filter AST in
  // `preprocess_file()` and `finalize_output()`
  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  std::vector<data_type> output_dtypes;
  if (expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });
  }

  return _metadata->filter_row_groups_with_bloom_filters(bloom_filter_data,
                                                         row_group_indices,
                                                         output_dtypes,
                                                         _output_column_schemas,
                                                         expr_conv.get_converted_expr(),
                                                         stream);
}

std::unique_ptr<cudf::column> impl::filter_data_pages_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (not _file_preprocessed) { prepare_row_groups(row_group_indices, options); }

  // Make sure we haven't gone past the input passes
  CUDF_EXPECTS(_file_itm_data._current_input_pass < _file_itm_data.num_passes(), "");

  // Save the name to reference converter to extract output filter AST in
  // `preprocess_file()` and `finalize_output()`
  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  std::vector<data_type> output_dtypes;
  if (expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });
  }
  return _metadata->filter_data_pages_with_stats(row_group_indices,
                                                 output_dtypes,
                                                 _output_column_schemas,
                                                 expr_conv.get_converted_expr(),
                                                 stream,
                                                 mr);
}

std::vector<std::vector<cudf::io::text::byte_range_info>> impl::get_filter_columns_data_pages(
  cudf::column_view input_rows,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  // Save the name to reference converter to extract output filter AST in
  // `preprocess_file()` and `finalize_output()`
  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  std::vector<data_type> output_dtypes;
  if (expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });
  }

  return _metadata->get_filter_columns_data_pages(
    input_rows, row_group_indices, output_dtypes, _output_column_schemas, stream);
}

std::unique_ptr<cudf::table> impl::materialize_filter_columns(
  cudf::mutable_column_view input_rows,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::vector<rmm::device_buffer>& data_pages_bytes,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  if (not _file_preprocessed) {
    // setup file level information
    // - read row group information
    // - setup information on (parquet) chunks
    // - compute schedule of input passes
    prepare_row_groups(row_group_indices, options);
  }

  // Make sure we haven't gone past the input passes
  CUDF_EXPECTS(_file_itm_data._current_input_pass < _file_itm_data.num_passes(), "");
  return {};
}

void impl::populate_metadata(table_metadata& out_metadata) const
{
  // Return column names
  out_metadata.schema_info.resize(_output_buffers.size());
  for (size_t i = 0; i < _output_column_schemas.size(); i++) {
    auto const& schema               = _metadata->get_schema(_output_column_schemas[i]);
    out_metadata.schema_info[i].name = schema.name;
    out_metadata.schema_info[i].is_nullable =
      schema.repetition_type != cudf::io::parquet::detail::REQUIRED;
  }

  // Return user metadata
  out_metadata.per_file_user_data = _metadata->get_key_value_metadata();
  out_metadata.user_data          = {out_metadata.per_file_user_data[0].begin(),
                                     out_metadata.per_file_user_data[0].end()};
}

void impl::prepare_data(cudf::host_span<std::vector<size_type> const> row_group_indices,
                        std::vector<rmm::device_buffer>& data_pages_bytes,
                        cudf::io::parquet_reader_options const& options)
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
    handle_chunking(data_pages_bytes, options);
  }
}

}  // namespace cudf::experimental::io::parquet::detail
