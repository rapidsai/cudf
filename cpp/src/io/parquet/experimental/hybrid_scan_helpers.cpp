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
#include <unordered_set>

namespace cudf::io::parquet::experimental::detail {

using aggregate_reader_metadata_base = parquet::detail::aggregate_reader_metadata;
using metadata_base                  = parquet::detail::metadata;

using io::detail::inline_column_buffer;
using parquet::detail::CompactProtocolReader;
using parquet::detail::equality_literals_collector;
using parquet::detail::input_column_info;
using parquet::detail::row_group_info;

namespace {

// Construct a vector of all row group indices from the input vectors
[[nodiscard]] auto all_row_group_indices(
  host_span<std::vector<cudf::size_type> const> row_group_indices)
{
  return std::vector<std::vector<cudf::size_type>>(row_group_indices.begin(),
                                                   row_group_indices.end());
}

// Compute total number of input row groups
[[nodiscard]] cudf::size_type compute_total_row_groups(
  host_span<std::vector<cudf::size_type> const> row_group_indices)
{
  auto const total_row_groups = std::accumulate(
    row_group_indices.begin(), row_group_indices.end(), size_t{0}, [](auto sum, auto const& pfm) {
      return sum + pfm.size();
    });

  // Check if we have less than 2B total row groups.
  CUDF_EXPECTS(total_row_groups <= std::numeric_limits<cudf::size_type>::max(),
               "Total number of row groups exceed the cudf::size_type's limit");
  return static_cast<cudf::size_type>(total_row_groups);
}

}  // namespace

metadata::metadata(cudf::host_span<uint8_t const> footer_bytes)
{
  CompactProtocolReader cp(footer_bytes.data(), footer_bytes.size());
  cp.read(this);
  CUDF_EXPECTS(cp.InitSchema(this), "Cannot initialize schema");
  sanitize_schema();
}

aggregate_reader_metadata::aggregate_reader_metadata(cudf::host_span<uint8_t const> footer_bytes,
                                                     bool use_arrow_schema,
                                                     bool has_cols_from_mismatched_srcs)
  : aggregate_reader_metadata_base({}, false, false)
{
  // Re-initialize internal variables here as base class was initialized without a source
  per_file_metadata = std::vector<metadata_base>{metadata{footer_bytes}.get_file_metadata()};
  keyval_maps       = collect_keyval_metadata();
  schema_idx_maps   = init_schema_idx_maps(has_cols_from_mismatched_srcs);
  num_rows          = calc_num_rows();
  num_row_groups    = calc_num_row_groups();

  // Force all columns to be nullable
  auto& schema = per_file_metadata.front().schema;
  std::for_each(schema.begin(), schema.end(), [](auto& col) {
    col.repetition_type = FieldRepetitionType::OPTIONAL;
  });

  // Collect and apply arrow:schema from Parquet's key value metadata section
  if (use_arrow_schema) {
    apply_arrow_schema();

    // Erase ARROW_SCHEMA_KEY from the output pfm if exists
    std::for_each(keyval_maps.begin(), keyval_maps.end(), [](auto& pfm) {
      pfm.erase(cudf::io::parquet::detail::ARROW_SCHEMA_KEY);
    });
  }
}

text::byte_range_info aggregate_reader_metadata::page_index_byte_range() const
{
  auto& schema     = per_file_metadata.front();
  auto& row_groups = schema.row_groups;

  if (row_groups.size() and row_groups.front().columns.size()) {
    auto const min_offset = schema.row_groups.front().columns.front().column_index_offset;
    auto const& last_col  = schema.row_groups.back().columns.back();
    auto const max_offset = last_col.offset_index_offset + last_col.offset_index_length;
    return {min_offset, (max_offset - min_offset)};
  }

  return {};
}

FileMetaData aggregate_reader_metadata::parquet_metadata() const
{
  return per_file_metadata.front();
}

void aggregate_reader_metadata::setup_page_index(cudf::host_span<uint8_t const> page_index_bytes)
{
  // Return early if empty page index buffer span
  if (page_index_bytes.empty()) {
    CUDF_LOG_WARN("Hybrid scan reader encountered empty page index buffer");
    return;
  }

  auto& row_groups = per_file_metadata.front().row_groups;

  CUDF_EXPECTS(not row_groups.empty() and not row_groups.front().columns.empty(),
               "No column chunks in Parquet schema to read page index for");

  CompactProtocolReader cp(page_index_bytes.data(), page_index_bytes.size());

  // Set the first ColumnChunk's offset of ColumnIndex as the adjusted zero offset
  int64_t const min_offset = row_groups.front().columns.front().column_index_offset;
  // now loop over row groups
  for (auto& rg : row_groups) {
    for (auto& col : rg.columns) {
      // Read the ColumnIndex for this ColumnChunk
      if (col.column_index_length > 0 && col.column_index_offset > 0) {
        int64_t const offset = col.column_index_offset - min_offset;
        cp.init(page_index_bytes.data() + offset, col.column_index_length);
        ColumnIndex ci;
        cp.read(&ci);
        col.column_index = std::move(ci);
      }
      // Read the OffsetIndex for this ColumnChunk
      if (col.offset_index_length > 0 && col.offset_index_offset > 0) {
        int64_t const offset = col.offset_index_offset - min_offset;
        cp.init(page_index_bytes.data() + offset, col.offset_index_length);
        OffsetIndex oi;
        cp.read(&oi);
        col.offset_index = std::move(oi);
      }
    }
  }
}

size_type aggregate_reader_metadata::total_rows_in_row_groups(
  cudf::host_span<std::vector<size_type> const> row_group_indices) const
{
  size_t total_rows = 0;

  std::for_each(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator(row_group_indices.size()),
                [&](auto const src_idx) {
                  auto const& pfm = per_file_metadata[src_idx];
                  for (auto const row_group_idx : row_group_indices[src_idx]) {
                    CUDF_EXPECTS(
                      row_group_idx < static_cast<cudf::size_type>(pfm.row_groups.size()),
                      "Row group index out of bounds");
                    total_rows += pfm.row_groups[row_group_idx].num_rows;
                  }
                });
  CUDF_EXPECTS(total_rows <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
               "Total number of rows exceeds cudf::size_type's limit");

  return static_cast<size_type>(total_rows);
}

std::tuple<std::vector<input_column_info>,
           std::vector<inline_column_buffer>,
           std::vector<cudf::size_type>>
aggregate_reader_metadata::select_payload_columns(
  std::optional<std::vector<std::string>> const& payload_column_names,
  std::optional<std::vector<std::string>> const& filter_column_names,
  bool include_index,
  bool strings_to_categorical,
  type_id timestamp_type_id)
{
  // If neither payload nor filter columns are specified, select all columns
  if (not payload_column_names.has_value() and not filter_column_names.has_value()) {
    // Call the base `select_columns()` method without specifying any columns
    return select_columns({}, {}, include_index, strings_to_categorical, timestamp_type_id);
  }

  std::vector<std::string> valid_payload_columns;

  // If payload columns are specified, only select payload columns that do not appear in the filter
  // expression
  if (payload_column_names.has_value()) {
    valid_payload_columns = *payload_column_names;
    // Remove filter columns from the provided payload column names
    if (filter_column_names.has_value() and not filter_column_names->empty()) {
      // Add filter column names to a hash set for faster lookup
      std::unordered_set<std::string> filter_columns_set(filter_column_names->begin(),
                                                         filter_column_names->end());
      // Remove a payload column name if it is also present in the hash set
      valid_payload_columns.erase(std::remove_if(valid_payload_columns.begin(),
                                                 valid_payload_columns.end(),
                                                 [&filter_columns_set](auto const& col) {
                                                   return filter_columns_set.count(col) > 0;
                                                 }),
                                  valid_payload_columns.end());
    }
    // Call the base `select_columns()` method with valid payload columns
    return select_columns(
      valid_payload_columns, {}, include_index, strings_to_categorical, timestamp_type_id);
  }

  // Else if only filter columns are specified, select all columns that do not appear in the
  // filter expression

  // Add filter column names to a hash set for faster lookup
  std::unordered_set<std::string> filter_columns_set(filter_column_names->begin(),
                                                     filter_column_names->end());

  std::function<void(std::string, int)> add_column_path = [&](std::string path_till_now,
                                                              int schema_idx) {
    auto const& schema_elem     = get_schema(schema_idx);
    std::string const curr_path = path_till_now + schema_elem.name;
    // If the current path is not a filter column, then add it and its children to the list of valid
    // payload columns
    if (filter_columns_set.count(curr_path) == 0) {
      valid_payload_columns.push_back(curr_path);
      // Add all children as well
      for (auto const& child_idx : schema_elem.children_idx) {
        add_column_path(curr_path + ".", child_idx);
      }
    }
  };

  // Add all but filter columns to valid payload columns
  if (not filter_column_names->empty()) {
    for (auto const& child_idx : get_schema(0).children_idx) {
      add_column_path("", child_idx);
    }
  }

  // Call the base `select_columns()` method with all but filter columns
  return select_columns(
    valid_payload_columns, {}, include_index, strings_to_categorical, timestamp_type_id);
}

std::tuple<int64_t, cudf::size_type, std::vector<row_group_info>>
aggregate_reader_metadata::select_row_groups(
  host_span<std::vector<cudf::size_type> const> row_group_indices,
  int64_t row_start,
  std::optional<cudf::size_type> const& row_count)
{
  // Compute the number of rows to read and skip
  auto [rows_to_skip, rows_to_read] = [&]() {
    if (not row_group_indices.empty()) { return std::pair<int64_t, cudf::size_type>{}; }
    auto const from_opts =
      cudf::io::detail::skip_rows_num_rows_from_options(row_start, row_count, get_num_rows());
    CUDF_EXPECTS(
      from_opts.second <= static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
      "Number of reading rows exceeds cudf's column size limit.");
    return std::pair{static_cast<int64_t>(from_opts.first),
                     static_cast<cudf::size_type>(from_opts.second)};
  }();

  // Vector to hold the `row_group_info` of selected row groups
  std::vector<row_group_info> selection;
  // Number of rows in each data source
  std::vector<size_t> num_rows_per_source(per_file_metadata.size(), 0);

  CUDF_EXPECTS(row_group_indices.size() == per_file_metadata.size(),
               "Must specify row groups for each source");

  auto total_row_groups = 0;
  for (size_t src_idx = 0; src_idx < row_group_indices.size(); ++src_idx) {
    auto const& fmd = per_file_metadata[src_idx];
    for (auto const& rowgroup_idx : row_group_indices[src_idx]) {
      CUDF_EXPECTS(
        rowgroup_idx >= 0 && rowgroup_idx < static_cast<cudf::size_type>(fmd.row_groups.size()),
        "Invalid rowgroup index");
      total_row_groups++;
      selection.emplace_back(rowgroup_idx, rows_to_read, src_idx);
      // if page-level indexes are present, then collect extra chunk and page info.
      column_info_for_row_group(selection.back(), 0);
      auto const rows_this_rg = get_row_group(rowgroup_idx, src_idx).num_rows;
      rows_to_read += rows_this_rg;
      num_rows_per_source[src_idx] += rows_this_rg;
    }
  }

  CUDF_EXPECTS(total_row_groups > 0, "No row groups added");

  return {rows_to_skip, rows_to_read, std::move(selection)};
}

std::vector<std::vector<cudf::size_type>> aggregate_reader_metadata::filter_row_groups_with_stats(
  host_span<std::vector<cudf::size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream) const
{
  // Compute total number of input row groups
  auto const total_row_groups = compute_total_row_groups(row_group_indices);

  // Filter stats table with StatsAST expression and collect filtered row group indices
  auto const stats_filtered_row_group_indices = apply_stats_filters(
    row_group_indices, total_row_groups, output_dtypes, output_column_schemas, filter, stream);

  return stats_filtered_row_group_indices.value_or(all_row_group_indices(row_group_indices));
}

std::vector<std::vector<cudf::size_type>>
aggregate_reader_metadata::filter_row_groups_with_dictionary_pages(
  cudf::detail::hostdevice_span<parquet::detail::ColumnChunkDesc const> chunks,
  cudf::detail::hostdevice_span<parquet::detail::PageInfo const> pages,
  cudf::host_span<std::vector<cudf::size_type> const> row_group_indices,
  cudf::host_span<std::vector<ast::literal*> const> literals,
  cudf::host_span<std::vector<ast::ast_operator> const> operators,
  cudf::host_span<data_type const> output_dtypes,
  cudf::host_span<int const> dictionary_col_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream) const
{
  // Number of input row groups
  auto const total_row_groups = static_cast<size_t>(compute_total_row_groups(row_group_indices));

  // NYI: Filter row groups using dictionaries
  auto const dictionary_filtered_row_groups = apply_dictionary_filter(chunks,
                                                                      pages,
                                                                      row_group_indices,
                                                                      literals,
                                                                      operators,
                                                                      total_row_groups,
                                                                      output_dtypes,
                                                                      dictionary_col_schemas,
                                                                      filter,
                                                                      stream);

  // Return all_row_group_indices as dictionary filtering not yet implemented
  return dictionary_filtered_row_groups.value_or(all_row_group_indices(row_group_indices));
}

}  // namespace cudf::io::parquet::experimental::detail
