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

#include "hybrid_scan_helpers.hpp"

#include "io/parquet/compact_protocol_reader.hpp"
#include "io/parquet/reader_impl_helpers.hpp"
#include "io/utilities/row_selection.hpp"

#include <cudf/logger.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>

namespace cudf::experimental::io::parquet::detail {

using aggregate_reader_metadata_base = cudf::io::parquet::detail::aggregate_reader_metadata;
using ColumnIndex                    = cudf::io::parquet::ColumnIndex;
using column_name_info               = cudf::io::column_name_info;
using CompactProtocolReader          = cudf::io::parquet::detail::CompactProtocolReader;
using equality_literals_collector    = cudf::io::parquet::detail::equality_literals_collector;
using FieldRepetitionType            = cudf::io::parquet::FieldRepetitionType;
using inline_column_buffer           = cudf::io::detail::inline_column_buffer;
using input_column_info              = cudf::io::parquet::detail::input_column_info;
using metadata_base                  = cudf::io::parquet::detail::metadata;
using OffsetIndex                    = cudf::io::parquet::OffsetIndex;
using row_group_info                 = cudf::io::parquet::detail::row_group_info;
using SchemaElement                  = cudf::io::parquet::SchemaElement;

metadata::metadata(cudf::host_span<uint8_t const> footer_bytes) {}

aggregate_reader_metadata::aggregate_reader_metadata(cudf::host_span<uint8_t const> footer_bytes,
                                                     bool use_arrow_schema,
                                                     bool has_cols_from_mismatched_srcs)
  : aggregate_reader_metadata_base({}, false, false)
{
}

cudf::io::text::byte_range_info aggregate_reader_metadata::get_page_index_bytes() const
{
  return {};
}

FileMetadata const& aggregate_reader_metadata::get_parquet_metadata() const
{
  return per_file_metadata.front();
}

void aggregate_reader_metadata::setup_page_index(cudf::host_span<uint8_t const> page_index_bytes) {}

std::tuple<std::vector<input_column_info>,
           std::vector<inline_column_buffer>,
           std::vector<cudf::size_type>>
aggregate_reader_metadata::select_payload_columns(
  std::optional<std::vector<std::string>> const& use_names,
  std::optional<std::vector<std::string>> const& filter_columns_names,
  bool include_index,
  bool strings_to_categorical,
  type_id timestamp_type_id)
{
  return {};
}

std::tuple<std::vector<input_column_info>,
           std::vector<inline_column_buffer>,
           std::vector<cudf::size_type>>
aggregate_reader_metadata::select_filter_columns(
  std::optional<std::vector<std::string>> const& filter_columns_names,
  bool include_index,
  bool strings_to_categorical,
  type_id timestamp_type_id)
{
  return {};
}

std::tuple<int64_t, cudf::size_type, std::vector<row_group_info>>
aggregate_reader_metadata::select_row_groups(
  host_span<std::vector<cudf::size_type> const> row_group_indices,
  int64_t row_start,
  std::optional<cudf::size_type> const& row_count)
{
  return {};
}

std::vector<std::vector<cudf::size_type>> aggregate_reader_metadata::filter_row_groups_with_stats(
  host_span<std::vector<cudf::size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter,
  rmm::cuda_stream_view stream) const
{
  return {};
}

std::vector<cudf::io::text::byte_range_info> aggregate_reader_metadata::get_bloom_filter_bytes(
  cudf::host_span<std::vector<cudf::size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter)
{
  return {};
}

std::vector<cudf::io::text::byte_range_info> aggregate_reader_metadata::get_dictionary_page_bytes(
  cudf::host_span<std::vector<cudf::size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter)
{
  return {};
}

std::vector<std::vector<cudf::size_type>>
aggregate_reader_metadata::filter_row_groups_with_dictionary_pages(
  std::vector<rmm::device_buffer>& dictionary_page_data,
  host_span<std::vector<cudf::size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter,
  rmm::cuda_stream_view stream) const
{
  return {};
}

std::vector<std::vector<cudf::size_type>>
aggregate_reader_metadata::filter_row_groups_with_bloom_filters(
  std::vector<rmm::device_buffer>& bloom_filter_data,
  host_span<std::vector<cudf::size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter,
  rmm::cuda_stream_view stream) const
{
  return {};
}

}  // namespace cudf::experimental::io::parquet::detail
