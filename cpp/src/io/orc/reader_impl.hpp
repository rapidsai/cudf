/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#pragma once

#include "aggregate_orc_metadata.hpp"
#include "orc.hpp"
#include "orc_gpu.hpp"

#include <io/utilities/column_buffer.hpp>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/orc.hpp>
#include <cudf/io/orc.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cudf {
namespace io {
namespace detail {
namespace orc {
using namespace cudf::io::orc;

// Forward declarations
class metadata;
namespace {
struct orc_stream_info;
struct stripe_source_mapping;
}  // namespace

/**
 * @brief Keeps track of orc mapping and child column details.
 */
struct reader_column_meta {
  std::vector<std::vector<size_type>>
    orc_col_map;                         // Mapping between column id in orc to processing order.
  std::vector<uint32_t> num_child_rows;  // number of rows in child columns

  std::vector<column_validity_info>
    parent_column_data;  // consists of parent column valid_map and null count
  std::vector<size_type> parent_column_index;

  std::vector<uint32_t> child_start_row;  // start row of child columns [stripe][column]
  std::vector<uint32_t>
    num_child_rows_per_stripe;  // number of rows of child columns [stripe][column]
  struct row_group_meta {
    uint32_t num_rows;   // number of rows in a column in a row group
    uint32_t start_row;  // start row in a column in a row group
  };
  // num_rowgroups * num_columns
  std::vector<row_group_meta> rwgrp_meta;  // rowgroup metadata [rowgroup][column]
};

/**
 * @brief Implementation for ORC reader
 */
class reader::impl {
 public:
  /**
   * @brief Constructor from a dataset source with reader options.
   *
   * @param sources Dataset sources
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::vector<std::unique_ptr<datasource>>&& sources,
                orc_reader_options const& options,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Read an entire set or a subset of data and returns a set of columns
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to read
   * @param stripes Indices of individual stripes to load if non-empty
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return The set of columns along with metadata
   */
  table_with_metadata read(size_type skip_rows,
                           size_type num_rows,
                           const std::vector<std::vector<size_type>>& stripes,
                           rmm::cuda_stream_view stream);

 private:
  /**
   * @brief Decompresses the stripe data, at stream granularity
   *
   * @param chunks Vector of list of column chunk descriptors
   * @param stripe_data List of source stripe column data
   * @param decompressor Block decompressor
   * @param stream_info List of stream to column mappings
   * @param num_stripes Number of stripes making up column chunks
   * @param row_groups Vector of list of row index descriptors
   * @param row_index_stride Distance between each row index
   * @param use_base_stride Whether to use base stride obtained from meta or use the computed value
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Device buffer to decompressed page data
   */
  rmm::device_buffer decompress_stripe_data(
    cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>& chunks,
    const std::vector<rmm::device_buffer>& stripe_data,
    OrcDecompressor const& decompressor,
    std::vector<orc_stream_info>& stream_info,
    size_t num_stripes,
    cudf::detail::hostdevice_2dvector<gpu::RowGroup>& row_groups,
    size_t row_index_stride,
    bool use_base_stride,
    rmm::cuda_stream_view stream);

  /**
   * @brief Converts the stripe column data and outputs to columns
   *
   * @param chunks Vector of list of column chunk descriptors
   * @param num_dicts Number of dictionary entries required
   * @param skip_rows Number of rows to offset from start
   * @param tz_table Local time to UTC conversion table
   * @param row_groups Vector of list of row index descriptors
   * @param row_index_stride Distance between each row index
   * @param out_buffers Output columns' device buffers
   * @param level Current nesting level being processed
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void decode_stream_data(cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>& chunks,
                          size_t num_dicts,
                          size_t skip_rows,
                          table_device_view tz_table,
                          cudf::detail::hostdevice_2dvector<gpu::RowGroup>& row_groups,
                          size_t row_index_stride,
                          std::vector<column_buffer>& out_buffers,
                          size_t level,
                          rmm::cuda_stream_view stream);

  /**
   * @brief Aggregate child metadata from parent column chunks.
   *
   * @param chunks Vector of list of parent column chunks.
   * @param row_groups Vector of list of row index descriptors
   * @param out_buffers Column buffers for columns.
   * @param list_col Vector of column metadata of list type parent columns.
   * @param level Current nesting level being processed.
   */
  void aggregate_child_meta(cudf::detail::host_2dspan<gpu::ColumnDesc> chunks,
                            cudf::detail::host_2dspan<gpu::RowGroup> row_groups,
                            std::vector<column_buffer>& out_buffers,
                            std::vector<orc_column_meta> const& list_col,
                            const int32_t level);

  /**
   * @brief Assemble the buffer with child columns.
   *
   * @param orc_col_id Column id in orc.
   * @param col_buffers Column buffers for columns and children.
   * @param level Current nesting level.
   */
  column_buffer&& assemble_buffer(const size_type orc_col_id,
                                  std::vector<std::vector<column_buffer>>& col_buffers,
                                  const size_t level,
                                  rmm::cuda_stream_view stream);

  /**
   * @brief Create columns and respective schema information from the buffer.
   *
   * @param col_buffers Column buffers for columns and children.
   * @param out_columns Vector of columns formed from column buffers.
   * @param schema_info Vector of schema information formed from column buffers.
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void create_columns(std::vector<std::vector<column_buffer>>&& col_buffers,
                      std::vector<std::unique_ptr<column>>& out_columns,
                      std::vector<column_name_info>& schema_info,
                      rmm::cuda_stream_view stream);

  /**
   * @brief Create empty columns and respective schema information from the buffer.
   *
   * @param col_buffers Column buffers for columns and children.
   * @param schema_info Vector of schema information formed from column buffers.
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return An empty column equivalent to orc column type.
   */
  std::unique_ptr<column> create_empty_column(const size_type orc_col_id,
                                              column_name_info& schema_info,
                                              rmm::cuda_stream_view stream);

  /**
   * @brief Setup table for converting timestamp columns from local to UTC time
   *
   * @return Timezone table with timestamp offsets
   */
  std::unique_ptr<table> compute_timezone_table(
    const std::vector<cudf::io::orc::metadata::stripe_source_mapping>& selected_stripes,
    rmm::cuda_stream_view stream);

 private:
  rmm::mr::device_memory_resource* _mr = nullptr;
  std::vector<std::unique_ptr<datasource>> _sources;
  cudf::io::orc::detail::aggregate_orc_metadata _metadata;
  cudf::io::orc::detail::column_hierarchy selected_columns;

  bool _use_index{true};
  bool _use_np_dtypes{true};
  std::vector<std::string> decimal128_columns;
  data_type _timestamp_type{type_id::EMPTY};
  reader_column_meta _col_meta{};
};

}  // namespace orc
}  // namespace detail
}  // namespace io
}  // namespace cudf
