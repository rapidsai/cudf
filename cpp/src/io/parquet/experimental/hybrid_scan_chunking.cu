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
#include "hybrid_scan_impl.hpp"
#include "io/comp/comp.hpp"
#include "io/comp/gpuinflate.hpp"
#include "io/comp/nvcomp_adapter.hpp"
#include "io/parquet/compact_protocol_reader.hpp"
#include "io/parquet/reader_impl_chunking.hpp"
#include "io/utilities/time_utils.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#include <numeric>

namespace cudf::experimental::io::parquet::detail {

namespace {

/**
 * @brief Converts cuDF units to Parquet units.
 *
 * @return A tuple of Parquet clock rate and Parquet decimal type.
 */
[[nodiscard]] std::tuple<int32_t, std::optional<cudf::io::parquet::detail::LogicalType>>
conversion_info(type_id column_type_id,
                type_id timestamp_type_id,
                cudf::io::parquet::detail::Type physical,
                std::optional<cudf::io::parquet::detail::LogicalType> logical_type)
{
  int32_t const clock_rate =
    is_chrono(data_type{column_type_id}) ? cudf::io::to_clockrate(timestamp_type_id) : 0;

  // TODO(ets): this is leftover from the original code, but will we ever output decimal as
  // anything but fixed point?
  if (logical_type.has_value() and
      logical_type->type == cudf::io::parquet::detail::LogicalType::DECIMAL) {
    // if decimal but not outputting as float or decimal, then convert to no logical type
    if (column_type_id != type_id::FLOAT64 and
        not cudf::is_fixed_point(data_type{column_type_id})) {
      return {clock_rate, std::nullopt};
    }
  }

  return {clock_rate, std::move(logical_type)};
}

/**
 * @brief Return the required number of bits to store a value.
 */
template <typename T = uint8_t>
[[nodiscard]] T required_bits(uint32_t max_level)
{
  return static_cast<T>(
    cudf::io::parquet::detail::CompactProtocolReader::NumRequiredBits(max_level));
}

}  // namespace

void impl::create_global_chunk_info(cudf::io::parquet_reader_options const& options)
{
  auto const num_rows         = _file_itm_data.global_num_rows;
  auto const& row_groups_info = _file_itm_data.row_groups;
  auto& chunks                = _file_itm_data.chunks;

  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_chunks        = row_groups_info.size() * num_input_columns;

  // Mapping of input column to page index column
  std::vector<size_type> column_mapping;

  if (_has_page_index and not row_groups_info.empty()) {
    // use first row group to define mappings (assumes same schema for each file)
    auto const& rg      = row_groups_info[0];
    auto const& columns = _metadata->get_row_group(rg.index, rg.source_index).columns;
    column_mapping.resize(num_input_columns);
    std::transform(
      _input_columns.begin(), _input_columns.end(), column_mapping.begin(), [&](auto const& col) {
        // translate schema_idx into something we can use for the page indexes
        if (auto it = std::find_if(columns.begin(),
                                   columns.end(),
                                   [&](auto const& col_chunk) {
                                     return col_chunk.schema_idx ==
                                            _metadata->map_schema_index(col.schema_idx,
                                                                        rg.source_index);
                                   });
            it != columns.end()) {
          return std::distance(columns.begin(), it);
        }
        CUDF_FAIL("cannot find column mapping");
      });
  }

  // Initialize column chunk information
  auto remaining_rows = num_rows;
  auto skip_rows      = _file_itm_data.global_skip_rows;
  for (auto const& rg : row_groups_info) {
    auto const& row_group      = _metadata->get_row_group(rg.index, rg.source_index);
    auto const row_group_start = rg.start_row;
    auto const row_group_rows  = std::min<int>(remaining_rows, row_group.num_rows);

    // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
    for (size_t i = 0; i < num_input_columns; ++i) {
      auto col = _input_columns[i];
      // look up metadata
      auto& col_meta = _metadata->get_column_metadata(rg.index, rg.source_index, col.schema_idx);
      auto& schema   = _metadata->get_schema(
        _metadata->map_schema_index(col.schema_idx, rg.source_index), rg.source_index);

      auto [clock_rate, logical_type] =
        conversion_info(to_type_id(schema,
                                   options.is_enabled_convert_strings_to_categories(),
                                   options.get_timestamp_type().id()),
                        options.get_timestamp_type().id(),
                        schema.type,
                        schema.logical_type);

      // for lists, estimate the number of bytes per row. this is used by the subpass reader to
      // determine where to split the decompression boundaries
      float const list_bytes_per_row_est =
        schema.max_repetition_level > 0 && row_group.num_rows > 0
          ? static_cast<float>(col_meta.total_uncompressed_size) /
              static_cast<float>(row_group.num_rows)
          : 0.0f;

      // grab the column_chunk_info for each chunk (if it exists)
      cudf::io::parquet::detail::column_chunk_info const* const chunk_info =
        _has_page_index ? &rg.column_chunks.value()[column_mapping[i]] : nullptr;

      chunks.push_back(cudf::io::parquet::detail::ColumnChunkDesc(
        col_meta.total_compressed_size,
        nullptr,
        col_meta.num_values,
        schema.type,
        schema.type_length,
        row_group_start,
        row_group_rows,
        schema.max_definition_level,
        schema.max_repetition_level,
        _metadata->get_output_nesting_depth(col.schema_idx),
        required_bits(schema.max_definition_level),
        required_bits(schema.max_repetition_level),
        col_meta.codec,
        logical_type,
        clock_rate,
        i,
        col.schema_idx,
        chunk_info,
        list_bytes_per_row_est,
        schema.type == cudf::io::parquet::detail::BYTE_ARRAY and
          options.is_enabled_convert_strings_to_categories(),
        rg.source_index));
    }
    // Adjust for skip_rows when updating the remaining rows after the first group
    remaining_rows -=
      (skip_rows) ? std::min<int>(rg.start_row + row_group.num_rows - skip_rows, remaining_rows)
                  : row_group_rows;
    // Set skip_rows = 0 as it is no longer needed for subsequent row_groups
    skip_rows = 0;
  }
}

void impl::compute_input_passes()
{
  // at this point, row_groups has already been filtered down to just the row groups we need to
  // handle optional skip_rows/num_rows parameters.
  auto const& row_groups_info = _file_itm_data.row_groups;

  // read everything in a single pass.
  _file_itm_data.input_pass_row_group_offsets.push_back(0);
  _file_itm_data.input_pass_row_group_offsets.push_back(row_groups_info.size());
  _file_itm_data.input_pass_start_row_count.push_back(0);
  auto rg_row_count = cudf::detail::make_counting_transform_iterator(0, [&](size_t i) {
    auto const& rgi       = row_groups_info[i];
    auto const& row_group = _metadata->get_row_group(rgi.index, rgi.source_index);
    return row_group.num_rows;
  });
  _file_itm_data.input_pass_start_row_count.push_back(
    std::reduce(rg_row_count, rg_row_count + row_groups_info.size()));
  return;
}

void impl::compute_output_chunks_for_subpass()
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  // simple case : no chunk size, no splits
  subpass.output_chunk_read_info.push_back({subpass.skip_rows, subpass.num_rows});
  return;
}

}  // namespace cudf::experimental::io::parquet::detail
