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
using text::byte_range_info;

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

  // Force all leaf columns to be nullable
  auto& schema = per_file_metadata.front().schema;
  std::for_each(schema.begin(), schema.end(), [](auto& col) {
    // Modifying the repetition type of lists converts them to structs, so we must skip that
    auto const is_leaf_col =
      not(col.type == Type::UNDEFINED or col.is_stub() or col.is_list() or col.is_struct());
    if (is_leaf_col) { col.repetition_type = FieldRepetitionType::OPTIONAL; }
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
                    CUDF_EXPECTS(std::cmp_less(row_group_idx, pfm.row_groups.size()),
                                 "Row group index out of bounds");
                    total_rows += pfm.row_groups[row_group_idx].num_rows;
                  }
                });
  CUDF_EXPECTS(std::cmp_less_equal(total_rows, std::numeric_limits<size_type>::max()),
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
    // Add the current path to the list of valid payload columns if it is not a filter column
    // TODO: Add children when AST filter expressions start supporting nested struct columns
    if (filter_columns_set.count(curr_path) == 0) { valid_payload_columns.push_back(curr_path); }
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

std::vector<std::vector<cudf::size_type>> aggregate_reader_metadata::filter_row_groups_with_stats(
  host_span<std::vector<cudf::size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<cudf::size_type const> output_column_schemas,
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

std::vector<byte_range_info> aggregate_reader_metadata::get_bloom_filter_bytes(
  cudf::host_span<std::vector<cudf::size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<cudf::size_type const> output_column_schemas,
  std::reference_wrapper<ast::expression const> filter)
{
  // Collect equality literals for each input table column
  auto const literals =
    equality_literals_collector{filter.get(), static_cast<cudf::size_type>(output_dtypes.size())}
      .get_literals();

  // Collect schema indices of columns with equality predicate(s)
  std::vector<cudf::size_type> bloom_filter_col_schemas;
  thrust::copy_if(thrust::host,
                  output_column_schemas.begin(),
                  output_column_schemas.end(),
                  literals.begin(),
                  std::back_inserter(bloom_filter_col_schemas),
                  [](auto& bloom_filter_literals) { return not bloom_filter_literals.empty(); });

  // No equality literals found, return empty vector
  if (bloom_filter_col_schemas.empty()) { return {}; }

  // Compute total number of input row groups
  auto const total_row_groups = compute_total_row_groups(row_group_indices);

  // Descriptors for all the chunks that make up the selected columns
  auto const num_bloom_filter_columns = bloom_filter_col_schemas.size();
  auto const num_chunks               = total_row_groups * num_bloom_filter_columns;

  std::vector<byte_range_info> bloom_filter_bytes;
  bloom_filter_bytes.reserve(num_chunks);

  // Flag to check if we have at least one valid bloom filter offset
  auto have_bloom_filters = false;

  // For all sources
  std::for_each(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator(row_group_indices.size()),
                [&](auto const src_index) {
                  // Get all row group indices in the data source
                  auto const& rg_indices = row_group_indices[src_index];
                  // For all row groups
                  std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto const rg_index) {
                    // For all column chunks
                    std::for_each(
                      bloom_filter_col_schemas.begin(),
                      bloom_filter_col_schemas.end(),
                      [&](auto const schema_idx) {
                        auto& col_meta = get_column_metadata(rg_index, src_index, schema_idx);
                        // Get bloom filter offsets and sizes
                        bloom_filter_bytes.emplace_back(col_meta.bloom_filter_offset.value_or(0),
                                                        col_meta.bloom_filter_length.value_or(0));

                        // Set `have_bloom_filters` if `bloom_filter_offset` is valid
                        if (col_meta.bloom_filter_offset.has_value()) { have_bloom_filters = true; }
                      });
                  });
                });

  if (not have_bloom_filters) { return {}; }

  return bloom_filter_bytes;
}

std::vector<byte_range_info> aggregate_reader_metadata::get_dictionary_page_bytes(
  cudf::host_span<std::vector<cudf::size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<cudf::size_type const> output_column_schemas,
  std::reference_wrapper<ast::expression const> filter)
{
  // Collect (in)equality literals for each input table column
  auto const literals =
    dictionary_literals_collector{filter.get(), static_cast<cudf::size_type>(output_dtypes.size())}
      .get_literals();

  auto iter = thrust::make_zip_iterator(thrust::counting_iterator<cudf::size_type>(0),
                                        output_column_schemas.begin());

  // Collect schema indices of columns with equality predicate(s)
  std::vector<thrust::tuple<cudf::size_type, cudf::size_type>> dictionary_col_schemas;
  thrust::copy_if(thrust::host,
                  iter,
                  iter + output_column_schemas.size(),
                  literals.begin(),
                  std::back_inserter(dictionary_col_schemas),
                  [](auto& dict_literals) { return not dict_literals.empty(); });

  // No (in)equality literals found, return empty vector
  if (dictionary_col_schemas.empty()) { return {}; }

  // Compute total number of input row groups
  auto const total_row_groups = compute_total_row_groups(row_group_indices);

  // Descriptors for all the chunks that make up the selected columns
  auto const num_dictionary_columns = dictionary_col_schemas.size();
  auto const num_chunks             = total_row_groups * num_dictionary_columns;

  std::vector<byte_range_info> dictionary_page_bytes;
  dictionary_page_bytes.reserve(num_chunks);

  // Flag to check if we have at least one valid dictionary page
  auto have_dictionary_pages = false;

  // For all sources
  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(row_group_indices.size()),
    [&](auto const src_index) {
      // Get all row group indices in the data source
      auto const& rg_indices = row_group_indices[src_index];
      // For all row groups
      std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto const rg_index) {
        auto const& rg = per_file_metadata[0].row_groups[rg_index];
        // For all column chunks
        std::for_each(
          dictionary_col_schemas.begin(),
          dictionary_col_schemas.end(),
          [&](auto const& schema_col_idx_pair) {
            auto const [input_col_idx, schema_idx] = schema_col_idx_pair;
            auto& col_meta        = get_column_metadata(rg_index, src_index, schema_idx);
            auto const& col_chunk = rg.columns[input_col_idx];

            auto dictionary_offset = int64_t{0};
            auto dictionary_size   = int64_t{0};

            // If any columns lack the page indexes then just return without modifying the
            // row_group_info.
            if (col_chunk.offset_index.has_value() and col_chunk.column_index.has_value()) {
              auto const& offset_index = col_chunk.offset_index.value();
              auto const num_pages     = offset_index.page_locations.size();

              // There is a bug in older versions of parquet-mr where the first data page offset
              // really points to the dictionary page. The first possible offset in a file is 4
              // (after the "PAR1" header), so check to see if the dictionary_page_offset is > 0.
              // If it is, then we haven't encountered the bug.
              if (col_meta.dictionary_page_offset > 0) {
                dictionary_offset     = col_meta.dictionary_page_offset;
                dictionary_size       = col_meta.data_page_offset - dictionary_offset;
                have_dictionary_pages = true;
              } else {
                // dictionary_page_offset is 0, so check to see if the data_page_offset does not
                // match the first offset in the offset index.  If they don't match, then
                // data_page_offset points to the dictionary page.
                if (num_pages > 0 &&
                    col_meta.data_page_offset < offset_index.page_locations[0].offset) {
                  dictionary_offset = col_meta.data_page_offset;
                  dictionary_size =
                    offset_index.page_locations[0].offset - col_meta.data_page_offset;
                  have_dictionary_pages = true;
                }
              }
            }

            dictionary_page_bytes.emplace_back(dictionary_offset, dictionary_size);
          });
      });
    });

  if (not have_dictionary_pages) { return {}; }

  return dictionary_page_bytes;
}

std::vector<std::vector<cudf::size_type>>
aggregate_reader_metadata::filter_row_groups_with_dictionary_pages(
  cudf::detail::hostdevice_span<parquet::detail::ColumnChunkDesc const> chunks,
  cudf::detail::hostdevice_span<parquet::detail::PageInfo const> pages,
  cudf::host_span<std::vector<cudf::size_type> const> row_group_indices,
  cudf::host_span<std::vector<ast::literal*> const> literals,
  cudf::host_span<std::vector<ast::ast_operator> const> operators,
  cudf::host_span<data_type const> output_dtypes,
  cudf::host_span<cudf::size_type const> dictionary_col_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream) const
{
  // Compute total number of input row groups
  auto const total_row_groups = static_cast<size_t>(compute_total_row_groups(row_group_indices));

  // Filter row groups using column chunk dictionaries
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

  return dictionary_filtered_row_groups.value_or(all_row_group_indices(row_group_indices));
}

std::vector<std::vector<cudf::size_type>>
aggregate_reader_metadata::filter_row_groups_with_bloom_filters(
  cudf::host_span<rmm::device_buffer> bloom_filter_data,
  host_span<std::vector<cudf::size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<cudf::size_type const> output_column_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream) const
{
  // Collect equality literals for each input table column
  auto const literals =
    equality_literals_collector{filter.get(), static_cast<cudf::size_type>(output_dtypes.size())}
      .get_literals();

  // Collect schema indices of columns with equality predicate(s)
  std::vector<cudf::size_type> bloom_filter_col_schemas;
  thrust::copy_if(thrust::host,
                  output_column_schemas.begin(),
                  output_column_schemas.end(),
                  literals.begin(),
                  std::back_inserter(bloom_filter_col_schemas),
                  [](auto& eq_literals) { return not eq_literals.empty(); });

  // Return all row groups if no column with equality predicate(s)
  if (bloom_filter_col_schemas.empty()) { return all_row_group_indices(row_group_indices); }

  // Compute total number of input row groups
  auto const total_row_groups = compute_total_row_groups(row_group_indices);

  auto const bloom_filtered_row_groups = apply_bloom_filters(bloom_filter_data,
                                                             row_group_indices,
                                                             literals,
                                                             total_row_groups,
                                                             output_dtypes,
                                                             bloom_filter_col_schemas,
                                                             filter,
                                                             stream);

  return bloom_filtered_row_groups.value_or(all_row_group_indices(row_group_indices));
}

}  // namespace cudf::io::parquet::experimental::detail
