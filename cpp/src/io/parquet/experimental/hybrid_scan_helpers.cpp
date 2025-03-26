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

using CompactProtocolReader          = cudf::io::parquet::detail::CompactProtocolReader;
using ColumnIndex                    = cudf::io::parquet::detail::ColumnIndex;
using OffsetIndex                    = cudf::io::parquet::detail::OffsetIndex;
using row_group_info                 = cudf::io::parquet::detail::row_group_info;
using aggregate_reader_metadata_base = cudf::io::parquet::detail::aggregate_reader_metadata;
using metadata_base                  = cudf::io::parquet::detail::metadata;
using input_column_info              = cudf::io::parquet::detail::input_column_info;
using inline_column_buffer           = cudf::io::detail::inline_column_buffer;
using equality_literals_collector    = cudf::io::parquet::detail::equality_literals_collector;

metadata::metadata(cudf::host_span<uint8_t const> footer_bytes,
                   cudf::host_span<uint8_t const> page_index_bytes)
{
  CompactProtocolReader cp(footer_bytes.data(), footer_bytes.size());
  cp.read(this);
  CUDF_EXPECTS(cp.InitSchema(this), "Cannot initialize schema");

  // column index and offset index are encoded back to back.
  // the first column of the first row group will have the first column index, the last
  // column of the last row group will have the final offset index.

  // FIXME: Remove this. Only temporary stuff to get PageIndex offsets.
  if (page_index_bytes.empty() and row_groups.size() and row_groups.front().columns.size()) {
    int64_t const min_offset = row_groups.front().columns.front().column_index_offset;
    auto const& last_col     = row_groups.back().columns.back();
    int64_t const max_offset = last_col.offset_index_offset + last_col.offset_index_length;
    std::cout << "min_offset: " << min_offset << " max_offset: " << max_offset << std::endl;
    CUDF_EXPECTS(!page_index_bytes.empty(), "Empty page index");
  }

  // Check if we have page_index buffer, and non-zero row groups and columnchunks
  // MH: TODO: We will be passed a span of the page index buffer instead of the whole file, so we
  // can uncomment the use of `min_offset`
  if (page_index_bytes.size() and row_groups.size() and row_groups.front().columns.size()) {
    // Set the first ColumnChunk's offset of ColumnIndex as the adjusted zero offset
    // int64_t const min_offset = row_groups.front().columns.front().column_index_offset;
    // now loop over row groups
    for (auto& rg : row_groups) {
      for (auto& col : rg.columns) {
        // Read the ColumnIndex for this ColumnChunk
        if (col.column_index_length > 0 && col.column_index_offset > 0) {
          int64_t const offset = col.column_index_offset;  // - min_offset;
          cp.init(page_index_bytes.data() + offset, col.column_index_length);
          ColumnIndex ci;
          cp.read(&ci);
          col.column_index = std::move(ci);
        }
        // Read the OffsetIndex for this ColumnChunk
        if (col.offset_index_length > 0 && col.offset_index_offset > 0) {
          int64_t const offset = col.offset_index_offset;  // - min_offset;
          cp.init(page_index_bytes.data() + offset, col.offset_index_length);
          OffsetIndex oi;
          cp.read(&oi);
          col.offset_index = std::move(oi);
        }
      }
    }
  }

  sanitize_schema();
}

aggregate_reader_metadata::aggregate_reader_metadata(
  cudf::host_span<uint8_t const> footer_bytes,
  cudf::host_span<uint8_t const> page_index_bytes,
  bool use_arrow_schema,
  bool has_cols_from_mismatched_srcs)
  : aggregate_reader_metadata_base({}, false, false)
{
  // Re-initialize internal variables here as base class was initialized without a source
  per_file_metadata =
    std::vector<metadata_base>{metadata{footer_bytes, page_index_bytes}.get_file_metadata()};
  keyval_maps     = collect_keyval_metadata();
  schema_idx_maps = init_schema_idx_maps(has_cols_from_mismatched_srcs);
  num_rows        = calc_num_rows();
  num_row_groups  = calc_num_row_groups();

  // Force all columns to be nullable
  auto& schema = per_file_metadata.front().schema;
  std::for_each(schema.begin(), schema.end(), [](auto& col) {
    col.repetition_type = cudf::io::parquet::detail::OPTIONAL;
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

std::
  tuple<std::vector<input_column_info>, std::vector<inline_column_buffer>, std::vector<size_type>>
  aggregate_reader_metadata::select_filter_columns(
    std::optional<std::vector<std::string>> const& filter_columns_names,
    bool include_index,
    bool strings_to_categorical,
    type_id timestamp_type_id)
{
  // Only extract filter columns
  return select_columns(filter_columns_names,
                        std::vector<std::string>{},
                        include_index,
                        strings_to_categorical,
                        timestamp_type_id);
}
std::tuple<int64_t, size_type, std::vector<row_group_info>>
aggregate_reader_metadata::add_row_groups(host_span<std::vector<size_type> const> row_group_indices,
                                          int64_t row_start,
                                          std::optional<size_type> const& row_count)
{
  // Compute the number of rows to read and skip
  auto [rows_to_skip, rows_to_read] = [&]() {
    if (not row_group_indices.empty()) { return std::pair<int64_t, size_type>{}; }
    auto const from_opts =
      cudf::io::detail::skip_rows_num_rows_from_options(row_start, row_count, get_num_rows());
    CUDF_EXPECTS(from_opts.second <= static_cast<int64_t>(std::numeric_limits<size_type>::max()),
                 "Number of reading rows exceeds cudf's column size limit.");
    return std::pair{static_cast<int64_t>(from_opts.first),
                     static_cast<size_type>(from_opts.second)};
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
        rowgroup_idx >= 0 && rowgroup_idx < static_cast<size_type>(fmd.row_groups.size()),
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

std::vector<std::vector<size_type>> aggregate_reader_metadata::filter_row_groups_with_stats(
  host_span<std::vector<size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter,
  rmm::cuda_stream_view stream) const
{
  std::vector<std::vector<size_type>> all_row_group_indices;
  std::transform(per_file_metadata.cbegin(),
                 per_file_metadata.cend(),
                 std::back_inserter(all_row_group_indices),
                 [](auto const& file_meta) {
                   std::vector<size_type> rg_idx(file_meta.row_groups.size());
                   std::iota(rg_idx.begin(), rg_idx.end(), 0);
                   return rg_idx;
                 });

  if (not filter.has_value()) { return all_row_group_indices; }

  // Compute total number of input row groups
  size_type total_row_groups = [&]() {
    if (not row_group_indices.empty()) {
      size_t const total_row_groups =
        std::accumulate(row_group_indices.begin(),
                        row_group_indices.end(),
                        size_t{0},
                        [](size_t& sum, auto const& pfm) { return sum + pfm.size(); });

      // Check if we have less than 2B total row groups.
      CUDF_EXPECTS(total_row_groups <= std::numeric_limits<cudf::size_type>::max(),
                   "Total number of row groups exceed the size_type's limit");
      return static_cast<size_type>(total_row_groups);
    } else {
      return num_row_groups;
    }
  }();

  // Span of input row group indices for predicate pushdown
  host_span<std::vector<size_type> const> input_row_group_indices;
  if (row_group_indices.empty()) {
    std::transform(per_file_metadata.cbegin(),
                   per_file_metadata.cend(),
                   std::back_inserter(all_row_group_indices),
                   [](auto const& file_meta) {
                     std::vector<size_type> rg_idx(file_meta.row_groups.size());
                     std::iota(rg_idx.begin(), rg_idx.end(), 0);
                     return rg_idx;
                   });
    input_row_group_indices = host_span<std::vector<size_type> const>(all_row_group_indices);
  } else {
    input_row_group_indices = row_group_indices;
  }

  // Filter stats table with StatsAST expression and collect filtered row group indices
  auto const stats_filtered_row_group_indices = apply_stats_filters(input_row_group_indices,
                                                                    total_row_groups,
                                                                    output_dtypes,
                                                                    output_column_schemas,
                                                                    filter.value(),
                                                                    stream);

  return stats_filtered_row_group_indices.value_or(all_row_group_indices);
}

std::vector<cudf::io::text::byte_range_info> aggregate_reader_metadata::get_bloom_filter_bytes(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter)
{
  // Number of surviving row groups after applying stats filter
  auto const total_row_groups = std::accumulate(
    row_group_indices.begin(),
    row_group_indices.end(),
    size_type{0},
    [](auto& sum, auto const& per_file_row_groups) { return sum + per_file_row_groups.size(); });

  // Collect equality literals for each input table column
  auto const equality_literals =
    equality_literals_collector{filter.value().get(),
                                static_cast<cudf::size_type>(output_dtypes.size())}
      .get_literals();

  // Collect schema indices of columns with equality predicate(s)
  std::vector<cudf::size_type> equality_col_schemas;
  thrust::copy_if(thrust::host,
                  output_column_schemas.begin(),
                  output_column_schemas.end(),
                  equality_literals.begin(),
                  std::back_inserter(equality_col_schemas),
                  [](auto& eq_literals) { return not eq_literals.empty(); });

  // Descriptors for all the chunks that make up the selected columns
  auto const num_equality_columns = equality_col_schemas.size();
  auto const num_chunks           = total_row_groups * num_equality_columns;

  std::vector<cudf::io::text::byte_range_info> bloom_filter_bytes;
  bloom_filter_bytes.reserve(num_chunks);

  // Flag to check if we have at least one valid bloom filter offset
  auto have_bloom_filters = false;

  // For all sources
  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(row_group_indices.size()),
    [&](auto const src_index) {
      // Get all row group indices in the data source
      auto const& rg_indices = row_group_indices[src_index];
      // For all row groups
      std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto const rg_index) {
        // For all column chunks
        std::for_each(
          equality_col_schemas.begin(), equality_col_schemas.end(), [&](auto const schema_idx) {
            auto& col_meta = get_column_metadata(rg_index, src_index, schema_idx);
            // Get bloom filter offsets and sizes
            bloom_filter_bytes.emplace_back(col_meta.bloom_filter_offset.value_or(0),
                                            col_meta.bloom_filter_length.value_or(0));

            // Set `have_bloom_filters` if `bloom_filter_offset` is valid
            if (col_meta.bloom_filter_offset.has_value()) { have_bloom_filters = true; }
          });
      });
    });

  // Clear vectors if found nothing
  if (not have_bloom_filters) { bloom_filter_bytes.clear(); }

  return bloom_filter_bytes;
}

std::vector<cudf::io::text::byte_range_info> aggregate_reader_metadata::get_dictionary_page_bytes(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter)
{
  // Number of surviving row groups after applying stats filter
  auto const total_row_groups = std::accumulate(
    row_group_indices.begin(),
    row_group_indices.end(),
    size_type{0},
    [](auto& sum, auto const& per_file_row_groups) { return sum + per_file_row_groups.size(); });

  // Collect equality literals for each input table column
  auto const [literals, _] =
    dictionary_literals_and_operators_collector{filter.value().get(),
                                                static_cast<cudf::size_type>(output_dtypes.size())}
      .get_literals_and_operators();

  // Collect schema indices of columns with equality predicate(s)
  std::vector<cudf::size_type> dictionary_col_schemas;
  thrust::copy_if(thrust::host,
                  output_column_schemas.begin(),
                  output_column_schemas.end(),
                  literals.begin(),
                  std::back_inserter(dictionary_col_schemas),
                  [](auto& dict_literals) { return not dict_literals.empty(); });

  // Descriptors for all the chunks that make up the selected columns
  auto const num_equality_columns = dictionary_col_schemas.size();
  auto const num_chunks           = total_row_groups * num_equality_columns;

  std::vector<cudf::io::text::byte_range_info> dictionary_page_bytes;
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
          dictionary_col_schemas.begin(), dictionary_col_schemas.end(), [&](auto const schema_idx) {
            auto& col_meta        = get_column_metadata(rg_index, src_index, schema_idx);
            auto const& col_chunk = rg.columns[schema_idx];

            auto dictionary_offset = int64_t{0};
            auto dictionary_size   = int64_t{0};

            // If any columns lack the page indexes then just return without modifying the
            // row_group_info.
            if (col_chunk.offset_index.has_value() and col_chunk.column_index.has_value()) {
              auto const& offset_index = col_chunk.offset_index.value();
              auto const num_pages     = offset_index.page_locations.size();

              // There is a bug in older versions of parquet-mr where the first data page offset
              // really points to the dictionary page. The first possible offset in a file is 4
              // (after the "PAR1" header), so check to see if the dictionary_page_offset is > 0. If
              // it is, then we haven't encountered the bug.
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

  // Clear vectors if found nothing
  if (not have_dictionary_pages) { dictionary_page_bytes.clear(); }

  return dictionary_page_bytes;
}

std::vector<std::vector<size_type>>
aggregate_reader_metadata::filter_row_groups_with_dictionary_pages(
  std::vector<rmm::device_buffer>& dictionary_page_data,
  host_span<std::vector<size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter,
  rmm::cuda_stream_view stream) const
{
  std::vector<std::vector<size_type>> all_row_group_indices;
  std::transform(row_group_indices.begin(),
                 row_group_indices.end(),
                 std::back_inserter(all_row_group_indices),
                 [](auto const& row_group) {
                   std::vector<size_type> rg_idx(row_group.size());
                   std::iota(rg_idx.begin(), rg_idx.end(), 0);
                   return rg_idx;
                 });

  // Number of surviving row groups after applying stats filter
  auto const total_row_groups = std::accumulate(
    row_group_indices.begin(),
    row_group_indices.end(),
    size_type{0},
    [](auto& sum, auto const& per_file_row_groups) { return sum + per_file_row_groups.size(); });

  // Collect literals and operators for dictionary page filtering for each input table column
  auto const [literals, operators] =
    dictionary_literals_and_operators_collector{filter.value().get(),
                                                static_cast<cudf::size_type>(output_dtypes.size())}
      .get_literals_and_operators();

  // Collect schema indices of columns with equality predicate(s)
  std::vector<cudf::size_type> dictionary_col_schemas;
  thrust::copy_if(thrust::host,
                  output_column_schemas.begin(),
                  output_column_schemas.end(),
                  literals.begin(),
                  std::back_inserter(dictionary_col_schemas),
                  [](auto& dict_literals) { return not dict_literals.empty(); });

  // Return early if no column with equality predicate(s)
  if (dictionary_col_schemas.empty()) { return all_row_group_indices; }

  // TODO: Decode dictionary pages and filter row groups based on dictionary pages
  auto dictionaries = materialize_dictionaries(
    dictionary_page_data, row_group_indices, output_dtypes, dictionary_col_schemas, stream);

  // TODO: Probe the dictionaries to get surviving row groups
  auto const dictionary_filtered_row_groups = apply_dictionary_filter(dictionaries,
                                                                      row_group_indices,
                                                                      literals,
                                                                      operators,
                                                                      total_row_groups,
                                                                      output_dtypes,
                                                                      dictionary_col_schemas,
                                                                      filter.value(),
                                                                      stream);

  // return dictionary_filtered_row_groups.value_or(all_row_group_indices);
  return dictionary_filtered_row_groups.value_or(all_row_group_indices);
}

std::vector<std::vector<size_type>> aggregate_reader_metadata::filter_row_groups_with_bloom_filters(
  std::vector<rmm::device_buffer>& bloom_filter_data,
  host_span<std::vector<size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter,
  rmm::cuda_stream_view stream) const
{
  std::vector<std::vector<size_type>> all_row_group_indices;
  std::transform(row_group_indices.begin(),
                 row_group_indices.end(),
                 std::back_inserter(all_row_group_indices),
                 [](auto const& row_group) {
                   std::vector<size_type> rg_idx(row_group.size());
                   std::iota(rg_idx.begin(), rg_idx.end(), 0);
                   return rg_idx;
                 });

  // Number of surviving row groups after applying stats filter
  auto const total_row_groups = std::accumulate(
    row_group_indices.begin(),
    row_group_indices.end(),
    size_type{0},
    [](auto& sum, auto const& per_file_row_groups) { return sum + per_file_row_groups.size(); });

  // Collect equality literals for each input table column
  auto const equality_literals =
    equality_literals_collector{filter.value().get(),
                                static_cast<cudf::size_type>(output_dtypes.size())}
      .get_literals();

  // Collect schema indices of columns with equality predicate(s)
  std::vector<cudf::size_type> equality_col_schemas;
  thrust::copy_if(thrust::host,
                  output_column_schemas.begin(),
                  output_column_schemas.end(),
                  equality_literals.begin(),
                  std::back_inserter(equality_col_schemas),
                  [](auto& eq_literals) { return not eq_literals.empty(); });

  // Return early if no column with equality predicate(s)
  if (equality_col_schemas.empty()) { return all_row_group_indices; }

  auto const bloom_filtered_row_groups = apply_bloom_filters(bloom_filter_data,
                                                             row_group_indices,
                                                             equality_literals,
                                                             total_row_groups,
                                                             output_dtypes,
                                                             equality_col_schemas,
                                                             filter.value(),
                                                             stream);

  return bloom_filtered_row_groups.value_or(all_row_group_indices);
}

}  // namespace cudf::experimental::io::parquet::detail
