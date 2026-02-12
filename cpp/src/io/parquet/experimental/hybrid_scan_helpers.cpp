/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_helpers.hpp"

#include "io/parquet/compact_protocol_reader.hpp"
#include "io/parquet/reader_impl_helpers.hpp"
#include "io/utilities/row_selection.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
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
  CUDF_FUNC_RANGE();

  CompactProtocolReader cp(footer_bytes.data(), footer_bytes.size());
  cp.read(this);
  CUDF_EXPECTS(cp.InitSchema(this), "Cannot initialize schema");
  sanitize_schema();
}

aggregate_reader_metadata::aggregate_reader_metadata(FileMetaData const& parquet_metadata,
                                                     bool use_arrow_schema,
                                                     bool has_cols_from_mismatched_srcs)
  : aggregate_reader_metadata_base(host_span<std::unique_ptr<datasource> const>{}, false, false)
{
  // Just copy over the FileMetaData struct to the internal metadata struct
  per_file_metadata.emplace_back(metadata{parquet_metadata});
  initialize_internals(use_arrow_schema, has_cols_from_mismatched_srcs);
}

aggregate_reader_metadata::aggregate_reader_metadata(cudf::host_span<uint8_t const> footer_bytes,
                                                     bool use_arrow_schema,
                                                     bool has_cols_from_mismatched_srcs)
  : aggregate_reader_metadata_base(host_span<std::unique_ptr<datasource> const>{}, false, false)
{
  // Re-initialize internal variables here as base class was initialized without a source
  per_file_metadata.emplace_back(metadata{footer_bytes});
  initialize_internals(use_arrow_schema, has_cols_from_mismatched_srcs);
}

void aggregate_reader_metadata::initialize_internals(bool use_arrow_schema,
                                                     bool has_cols_from_mismatched_srcs)
{
  keyval_maps     = collect_keyval_metadata();
  schema_idx_maps = init_schema_idx_maps(has_cols_from_mismatched_srcs);
  num_rows        = calc_num_rows();
  num_row_groups  = calc_num_row_groups();

  // Force all non-nullable (REQUIRED) columns to be nullable without modifying REPEATED columns to
  // preserve list structures
  auto& schema = per_file_metadata.front().schema;
  std::for_each(schema.begin() + 1, schema.end(), [](auto& col) {
    // TODO: Store information of whichever column schema we modified here and restore it to
    // `REQUIRED` if we end up not pruning any pages out of it
    if (col.repetition_type == FieldRepetitionType::REQUIRED) {
      col.repetition_type = FieldRepetitionType::OPTIONAL;
    }
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

  // Get the file metadata and setup the page index
  auto& file_metadata    = per_file_metadata.front();
  auto const& row_groups = file_metadata.row_groups;

  // Check for empty parquet file
  CUDF_EXPECTS(not row_groups.empty() and not row_groups.front().columns.empty(),
               "No column chunks in Parquet schema to read page index for");

  // Set the first ColumnChunk's offset of ColumnIndex as the adjusted zero offset
  int64_t const min_offset = row_groups.front().columns.front().column_index_offset;

  // Check if the page index buffer is valid
  {
    auto const& last_col  = row_groups.back().columns.back();
    auto const max_offset = last_col.offset_index_offset + last_col.offset_index_length;
    CUDF_EXPECTS(max_offset > min_offset, "Encountered an invalid page index buffer");
  }

  file_metadata.setup_page_index(page_index_bytes, min_offset);
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
  bool ignore_missing_columns,
  type_id timestamp_type_id)
{
  // If neither payload nor filter columns are specified, select all columns
  if (not payload_column_names.has_value() and not filter_column_names.has_value()) {
    // Call the base `select_columns()` method without specifying any columns
    return select_columns(
      {}, {}, include_index, strings_to_categorical, ignore_missing_columns, timestamp_type_id);
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
    return select_columns(valid_payload_columns,
                          {},
                          include_index,
                          strings_to_categorical,
                          ignore_missing_columns,
                          timestamp_type_id);
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
  return select_columns(valid_payload_columns,
                        {},
                        include_index,
                        strings_to_categorical,
                        ignore_missing_columns,
                        timestamp_type_id);
}

std::vector<std::vector<cudf::size_type>>
aggregate_reader_metadata::filter_row_groups_with_byte_range(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::size_t bytes_to_skip,
  std::optional<std::size_t> const& bytes_to_read) const
{
  return apply_byte_bounds_filter(row_group_indices, bytes_to_skip, bytes_to_read);
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

  // Collect schema indices of columns with equality predicate(s)
  std::vector<cudf::size_type> dictionary_col_schemas;
  thrust::copy_if(thrust::host,
                  output_column_schemas.begin(),
                  output_column_schemas.end(),
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
      std::optional<size_type> colchunk_iter_offset{};
      // For all row groups
      std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto const rg_index) {
        auto const& row_group = per_file_metadata[src_index].row_groups[rg_index];
        // For all column chunks
        std::for_each(
          dictionary_col_schemas.begin(),
          dictionary_col_schemas.end(),
          [&](auto const& schema_idx) {
            // Get the column chunk iterator
            if (not colchunk_iter_offset.has_value() or
                row_group.columns[colchunk_iter_offset.value()].schema_idx != schema_idx) {
              auto const& colchunk_iter = std::find_if(
                row_group.columns.begin(), row_group.columns.end(), [schema_idx](auto const& col) {
                  return col.schema_idx == schema_idx;
                });
              CUDF_EXPECTS(colchunk_iter != row_group.columns.end(),
                           "Column chunk with schema index " + std::to_string(schema_idx) +
                             " not found in row group",
                           std::invalid_argument);
              colchunk_iter_offset = std::distance(row_group.columns.begin(), colchunk_iter);
            }
            auto const colchunk_iter = row_group.columns.begin() + colchunk_iter_offset.value();
            auto const& col_chunk    = *colchunk_iter;
            auto const& col_meta     = col_chunk.meta_data;

            // Make sure that we have page index and the column chunk doesn't have any
            // non-dictionary encoded pages
            auto const has_page_index_and_only_dict_encoded_pages = [&]() {
              auto const has_page_index =
                col_chunk.offset_index.has_value() and col_chunk.column_index.has_value();

              if (has_page_index and not col_meta.encoding_stats.has_value()) {
                CUDF_LOG_WARN(
                  "Skipping the column chunk because it does not have encoding stats "
                  "needed to determine if all pages are dictionary encoded");
                return false;
              }

              return has_page_index and
                     std::all_of(
                       col_meta.encoding_stats.value().cbegin(),
                       col_meta.encoding_stats.value().cend(),
                       [](auto const& page_encoding_stats) {
                         return page_encoding_stats.page_type == PageType::DICTIONARY_PAGE or
                                page_encoding_stats.encoding == Encoding::PLAIN_DICTIONARY or
                                page_encoding_stats.encoding == Encoding::RLE_DICTIONARY;
                       });
            }();

            auto dictionary_offset = int64_t{0};
            auto dictionary_size   = int64_t{0};

            if (has_page_index_and_only_dict_encoded_pages) {
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
  cudf::host_span<cudf::device_span<uint8_t const> const> bloom_filter_data,
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

  // Transform bloom filter data to cuda::std::byte type for apply_bloom_filters
  std::vector<cudf::device_span<cuda::std::byte const>> transformed_bloom_filter_data;
  transformed_bloom_filter_data.reserve(bloom_filter_data.size());
  std::transform(bloom_filter_data.begin(),
                 bloom_filter_data.end(),
                 std::back_inserter(transformed_bloom_filter_data),
                 [](auto const& data) {
                   return cudf::device_span<cuda::std::byte const>{
                     reinterpret_cast<cuda::std::byte const*>(data.data()), data.size()};
                 });

  auto const bloom_filtered_row_groups = apply_bloom_filters(transformed_bloom_filter_data,
                                                             row_group_indices,
                                                             literals,
                                                             total_row_groups,
                                                             output_dtypes,
                                                             bloom_filter_col_schemas,
                                                             filter,
                                                             stream);

  return bloom_filtered_row_groups.value_or(all_row_group_indices(row_group_indices));
}

/**
 * @brief Converts column named expression to column index reference expression
 */
named_to_reference_converter::named_to_reference_converter(
  std::optional<std::reference_wrapper<ast::expression const>> expr,
  table_metadata const& metadata,
  std::vector<SchemaElement> const& schema_tree,
  cudf::io::parquet_reader_options const& options)
{
  if (!expr.has_value()) { return; }

  _column_indices_to_names =
    cudf::io::parquet::detail::map_column_indices_to_names(options, schema_tree);

  // Map column names to their indices
  std::transform(metadata.schema_info.cbegin(),
                 metadata.schema_info.cend(),
                 thrust::counting_iterator<size_t>(0),
                 std::inserter(_column_name_to_index, _column_name_to_index.end()),
                 [](auto const& sch, auto index) { return std::make_pair(sch.name, index); });

  expr.value().get().accept(*this);
}

std::reference_wrapper<ast::expression const> named_to_reference_converter::visit(
  ast::column_reference const& expr)
{
  // Map the column index to its name
  auto const col_name = _column_indices_to_names[expr.get_column_index()];
  // Check if the column name exists in the metadata and map it to its new column index
  auto col_index_it = _column_name_to_index.find(col_name);
  if (col_index_it == _column_name_to_index.end()) {
    CUDF_FAIL("Column name not found in metadata");
  }
  auto col_index = col_index_it->second;
  // Create a new column reference
  _col_ref.emplace_back(col_index);
  _converted_expr = std::reference_wrapper<ast::expression const>(_col_ref.back());
  return std::reference_wrapper<ast::expression const>(_col_ref.back());
}

}  // namespace cudf::io::parquet::experimental::detail
