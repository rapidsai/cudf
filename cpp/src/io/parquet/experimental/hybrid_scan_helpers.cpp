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
using SchemaElement                  = cudf::io::parquet::detail::SchemaElement;
using column_name_info               = cudf::io::column_name_info;
using inline_column_buffer           = cudf::io::detail::inline_column_buffer;
using input_column_info              = cudf::io::parquet::detail::input_column_info;
using size_type                      = cudf::size_type;

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

cudf::io::text::byte_range_info aggregate_reader_metadata::get_page_index_bytes() const
{
  auto& schema = per_file_metadata.front();

  if (schema.row_groups.size() and schema.row_groups.front().columns.size()) {
    int64_t const min_offset = schema.row_groups.front().columns.front().column_index_offset;
    auto const& last_col     = schema.row_groups.back().columns.back();
    int64_t const max_offset = last_col.offset_index_offset + last_col.offset_index_length;
    return {min_offset, (max_offset - min_offset)};
  }

  return {};
}

void aggregate_reader_metadata::setup_page_index(cudf::host_span<uint8_t const> page_index_bytes)
{
  auto& schema     = per_file_metadata.front();
  auto& row_groups = schema.row_groups;

  // Check if we have page_index buffer, and non-zero row groups and columnchunks
  // MH: TODO: We will be passed a span of the page index buffer instead of the whole file, so we
  // can uncomment the use of `min_offset`
  if (page_index_bytes.size() and row_groups.size() and row_groups.front().columns.size()) {
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
}

std::
  tuple<std::vector<input_column_info>, std::vector<inline_column_buffer>, std::vector<size_type>>
  aggregate_reader_metadata::select_payload_columns(
    std::optional<std::vector<std::string>> const& use_names,
    std::optional<std::vector<std::string>> const& filter_columns_names,
    bool include_index,
    bool strings_to_categorical,
    type_id timestamp_type_id)
{
  auto const find_schema_child =
    [&](SchemaElement const& schema_elem, std::string const& name, int const pfm_idx = 0) {
      auto const& col_schema_idx = std::find_if(
        schema_elem.children_idx.cbegin(),
        schema_elem.children_idx.cend(),
        [&](size_t col_schema_idx) { return get_schema(col_schema_idx, pfm_idx).name == name; });

      return (col_schema_idx != schema_elem.children_idx.end())
               ? static_cast<size_type>(*col_schema_idx)
               : -1;
    };

  std::vector<cudf::io::detail::inline_column_buffer> output_columns;
  std::vector<input_column_info> input_columns;
  std::vector<int> nesting;

  // Return true if column path is valid. e.g. if the path is {"struct1", "child1"}, then it is
  // valid if "struct1.child1" exists in this file's schema. If "struct1" exists but "child1" is
  // not a child of "struct1" then the function will return false for "struct1"
  std::function<bool(
    column_name_info const*, int, std::vector<cudf::io::detail::inline_column_buffer>&, bool)>
    build_column = [&](column_name_info const* col_name_info,
                       int schema_idx,
                       std::vector<cudf::io::detail::inline_column_buffer>& out_col_array,
                       bool has_list_parent) {
      auto const& schema_elem = get_schema(schema_idx);

      // if schema_elem is a stub then it does not exist in the column_name_info and column_buffer
      // hierarchy. So continue on
      if (schema_elem.is_stub()) {
        // is this legit?
        CUDF_EXPECTS(schema_elem.num_children == 1, "Unexpected number of children for stub");
        auto const child_col_name_info = col_name_info ? &col_name_info->children[0] : nullptr;
        return build_column(
          child_col_name_info, schema_elem.children_idx[0], out_col_array, has_list_parent);
      }

      auto const one_level_list = schema_elem.is_one_level_list(get_schema(schema_elem.parent_idx));

      // if we're at the root, this is a new output column
      auto const col_type = one_level_list
                              ? type_id::LIST
                              : to_type_id(schema_elem, strings_to_categorical, timestamp_type_id);
      auto const dtype    = to_data_type(col_type, schema_elem);

      cudf::io::detail::inline_column_buffer output_col(
        dtype, schema_elem.repetition_type == cudf::io::parquet::detail::OPTIONAL);
      if (has_list_parent) {
        output_col.user_data |=
          cudf::io::parquet::detail::PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT;
      }
      // store the index of this element if inserted in out_col_array
      nesting.push_back(static_cast<int>(out_col_array.size()));
      output_col.name = schema_elem.name;

      // build each child
      bool path_is_valid = false;
      if (col_name_info == nullptr or col_name_info->children.empty()) {
        // add all children of schema_elem.
        // At this point, we can no longer pass a col_name_info to build_column
        for (int idx = 0; idx < schema_elem.num_children; idx++) {
          path_is_valid |= build_column(nullptr,
                                        schema_elem.children_idx[idx],
                                        output_col.children,
                                        has_list_parent || col_type == type_id::LIST);
        }
      } else {
        for (const auto& idx : col_name_info->children) {
          path_is_valid |= build_column(&idx,
                                        find_schema_child(schema_elem, idx.name),
                                        output_col.children,
                                        has_list_parent || col_type == type_id::LIST);
        }
      }

      // if I have no children, we're at a leaf and I'm an input column (that is, one with actual
      // data stored) so add me to the list.
      if (schema_elem.num_children == 0) {
        input_column_info& input_col = input_columns.emplace_back(
          schema_idx, schema_elem.name, schema_elem.max_repetition_level > 0);

        // set up child output column for one-level encoding list
        if (one_level_list) {
          // determine the element data type
          auto const element_type =
            to_type_id(schema_elem, strings_to_categorical, timestamp_type_id);
          auto const element_dtype = to_data_type(element_type, schema_elem);

          cudf::io::detail::inline_column_buffer element_col(
            element_dtype, schema_elem.repetition_type == cudf::io::parquet::detail::OPTIONAL);
          if (has_list_parent || col_type == type_id::LIST) {
            element_col.user_data |=
              cudf::io::parquet::detail::PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT;
          }
          // store the index of this element
          nesting.push_back(static_cast<int>(output_col.children.size()));
          // TODO: not sure if we should assign a name or leave it blank
          element_col.name = "element";

          output_col.children.push_back(std::move(element_col));
        }

        std::copy(nesting.cbegin(), nesting.cend(), std::back_inserter(input_col.nesting));

        // pop off the extra nesting element.
        if (one_level_list) { nesting.pop_back(); }

        path_is_valid = true;  // If we're able to reach leaf then path is valid
      }

      if (path_is_valid) { out_col_array.push_back(std::move(output_col)); }

      nesting.pop_back();
      return path_is_valid;
    };

  // Compares two schema elements to be equal except their number of children
  auto const equal_to_except_num_children = [](SchemaElement const& lhs, SchemaElement const& rhs) {
    return lhs.type == rhs.type and lhs.converted_type == rhs.converted_type and
           lhs.type_length == rhs.type_length and lhs.name == rhs.name and
           lhs.decimal_scale == rhs.decimal_scale and
           lhs.decimal_precision == rhs.decimal_precision and lhs.field_id == rhs.field_id;
  };

  // Maps a projected column's schema_idx in the zeroth per_file_metadata (source) to the
  // corresponding schema_idx in pfm_idx'th per_file_metadata (destination). The projected
  // column's path must match across sources, else an appropriate exception is thrown.
  std::function<void(column_name_info const*, int const, int const, int const)> map_column =
    [&](column_name_info const* col_name_info,
        int const src_schema_idx,
        int const dst_schema_idx,
        int const pfm_idx) {
      auto const& src_schema_elem = get_schema(src_schema_idx);
      auto const& dst_schema_elem = get_schema(dst_schema_idx, pfm_idx);

      // Check the schema elements to be equal except their number of children as we only care about
      // the specific column paths in the schema trees. Raise an invalid_argument error if the
      // schema elements don't match.
      CUDF_EXPECTS(equal_to_except_num_children(src_schema_elem, dst_schema_elem),
                   "Encountered mismatching SchemaElement properties for a column in "
                   "the selected path",
                   std::invalid_argument);

      // Get the schema_idx_map for this data source (pfm)
      auto& schema_idx_map = schema_idx_maps[pfm_idx - 1];
      // Map the schema index from 0th tree (src) to the one in the current (dst) tree.
      schema_idx_map[src_schema_idx] = dst_schema_idx;

      // If src_schema_elem is a stub, it does not exist in the column_name_info and column_buffer
      // hierarchy. So continue on with mapping.
      if (src_schema_elem.is_stub()) {
        // Check if dst_schema_elem is also a stub i.e. has num_children == 1 that we didn't
        // previously check. Raise an invalid_argument error if dst_schema_elem is not a stub.
        CUDF_EXPECTS(dst_schema_elem.is_stub(),
                     "Encountered mismatching schemas for stub.",
                     std::invalid_argument);
        auto const child_col_name_info = col_name_info ? &col_name_info->children[0] : nullptr;
        return map_column(child_col_name_info,
                          src_schema_elem.children_idx[0],
                          dst_schema_elem.children_idx[0],
                          pfm_idx);
      }

      // The path ends here. If this is a list/struct col (has children), then map all its children
      // which must be identical.
      if (col_name_info == nullptr or col_name_info->children.empty()) {
        // Check the number of children to be equal to be mapped. An out_of_range error if the
        // number of children isn't equal.
        CUDF_EXPECTS(src_schema_elem.num_children == dst_schema_elem.num_children,
                     "Encountered mismatching number of children for a "
                     "column in the selected path",
                     std::out_of_range);

        std::for_each(thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(src_schema_elem.num_children),
                      [&](auto const child_idx) {
                        map_column(nullptr,
                                   src_schema_elem.children_idx[child_idx],
                                   dst_schema_elem.children_idx[child_idx],
                                   pfm_idx);
                      });
      }
      // The path goes further down to specific child(ren) of this column so map only those
      // children.
      else {
        std::for_each(
          col_name_info->children.cbegin(),
          col_name_info->children.cend(),
          [&](auto const& child_col_name_info) {
            // Ensure that each named child column exists in the destination schema tree for the
            // paths to align up. An out_of_range error otherwise.
            CUDF_EXPECTS(
              find_schema_child(dst_schema_elem, child_col_name_info.name, pfm_idx) != -1,
              "Encountered mismatching schema tree depths across data sources",
              std::out_of_range);
            map_column(&child_col_name_info,
                       find_schema_child(src_schema_elem, child_col_name_info.name),
                       find_schema_child(dst_schema_elem, child_col_name_info.name, pfm_idx),
                       pfm_idx);
          });
      }
    };

  std::vector<int> output_column_schemas;

  //
  // there is not necessarily a 1:1 mapping between input columns and output columns.
  // For example, parquet does not explicitly store a ColumnChunkDesc for struct columns.
  // The "structiness" is simply implied by the schema.  For example, this schema:
  //  required group field_id=1 name {
  //    required binary field_id=2 firstname (String);
  //    required binary field_id=3 middlename (String);
  //    required binary field_id=4 lastname (String);
  // }
  // will only contain 3 internal columns of data (firstname, middlename, lastname).  But of
  // course "name" is ultimately the struct column we want to return.
  //
  // "firstname", "middlename" and "lastname" represent the input columns in the file that we
  // process to produce the final cudf "name" column.
  //
  // A user can ask for a single field out of the struct e.g. firstname.
  // In this case they'll pass a fully qualified name to the schema element like
  // ["name", "firstname"]
  //
  auto const& root = get_schema(0);
  if (not use_names.has_value() and not filter_columns_names.has_value()) {
    for (auto const& schema_idx : root.children_idx) {
      build_column(nullptr, schema_idx, output_columns, false);
      output_column_schemas.push_back(schema_idx);
    }
  } else {
    struct path_info {
      std::string full_path;
      int schema_idx;
    };

    // Convert schema into a vector of all possible payload column paths
    std::vector<path_info> payload_column_paths;
    std::vector<std::reference_wrapper<std::vector<std::string> const>> const filter_column_names{
      *filter_columns_names};
    std::function<void(std::string, int)> add_path = [&](std::string path_till_now,
                                                         int schema_idx) {
      auto const& schema_elem     = get_schema(schema_idx);
      std::string const curr_path = path_till_now + schema_elem.name;
      // Do not push to payload_column_paths if the current path is in filter_column_names
      if (std::none_of(filter_column_names[0].get().cbegin(),
                       filter_column_names[0].get().cend(),
                       [&](auto const& name) { return name == curr_path; })) {
        payload_column_paths.push_back({curr_path, schema_idx});
        for (auto const& child_idx : schema_elem.children_idx) {
          add_path(curr_path + ".", child_idx);
        }
      }
    };
    for (auto const& child_idx : get_schema(0).children_idx) {
      add_path("", child_idx);
    }

    // Now construct paths as vector of strings for further consumption
    std::vector<std::vector<std::string>> use_names3;
    std::transform(payload_column_paths.cbegin(),
                   payload_column_paths.cend(),
                   std::back_inserter(use_names3),
                   [&](path_info const& valid_path) {
                     auto schema_idx = valid_path.schema_idx;
                     std::vector<std::string> result_path;
                     do {
                       SchemaElement const& elem = get_schema(schema_idx);
                       result_path.push_back(elem.name);
                       schema_idx = elem.parent_idx;
                     } while (schema_idx > 0);
                     return std::vector<std::string>(result_path.rbegin(), result_path.rend());
                   });

    std::vector<column_name_info> selected_columns;
    if (include_index) {
      std::vector<std::string> const index_names = get_pandas_index_names();
      std::transform(index_names.cbegin(),
                     index_names.cend(),
                     std::back_inserter(selected_columns),
                     [](std::string const& name) { return column_name_info(name); });
    }
    // Merge the vector use_names into a set of hierarchical column_name_info objects
    /* This is because if we have columns like this:
     *     col1
     *      / \
     *    s3   f4
     *   / \
     * f5   f6
     *
     * there may be common paths in use_names like:
     * {"col1", "s3", "f5"}, {"col1", "f4"}
     * which means we want the output to contain
     *     col1
     *      / \
     *    s3   f4
     *   /
     * f5
     *
     * rather than
     *  col1   col1
     *   |      |
     *   s3     f4
     *   |
     *   f5
     */
    for (auto const& path : use_names3) {
      auto array_to_find_in = &selected_columns;
      for (auto const& name_to_find : path) {
        // Check if the path exists in our selected_columns and if not, add it.
        auto found_col = std::find_if(
          array_to_find_in->begin(),
          array_to_find_in->end(),
          [&name_to_find](column_name_info const& col) { return col.name == name_to_find; });
        if (found_col == array_to_find_in->end()) {
          auto& col        = array_to_find_in->emplace_back(name_to_find);
          array_to_find_in = &col.children;
        } else {
          // Path exists. go down further.
          array_to_find_in = &found_col->children;
        }
      }
    }
    for (auto& col : selected_columns) {
      auto const& top_level_col_schema_idx = find_schema_child(root, col.name);
      bool const valid_column = build_column(&col, top_level_col_schema_idx, output_columns, false);
      if (valid_column) {
        output_column_schemas.push_back(top_level_col_schema_idx);

        // Map the column's schema_idx across the rest of the data sources if required.
        if (per_file_metadata.size() > 1 and not schema_idx_maps.empty()) {
          std::for_each(thrust::make_counting_iterator(static_cast<size_t>(1)),
                        thrust::make_counting_iterator(per_file_metadata.size()),
                        [&](auto const pfm_idx) {
                          auto const& dst_root = get_schema(0, pfm_idx);
                          // Ensure that each top level column exists in the destination schema
                          // tree. An out_of_range error is thrown otherwise.
                          CUDF_EXPECTS(
                            find_schema_child(dst_root, col.name, pfm_idx) != -1,
                            "Encountered mismatching schema tree depths across data sources",
                            std::out_of_range);
                          map_column(&col,
                                     top_level_col_schema_idx,
                                     find_schema_child(dst_root, col.name, pfm_idx),
                                     pfm_idx);
                        });
        }
      }
    }
  }

  return std::make_tuple(
    std::move(input_columns), std::move(output_columns), std::move(output_column_schemas));
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
aggregate_reader_metadata::select_row_groups(
  host_span<std::vector<size_type> const> row_group_indices,
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
