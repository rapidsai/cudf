/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/cudf.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/legacy/table.hpp>
#include <io/utilities/datasource.hpp>
#include <io/utilities/legacy/wrapper_utils.hpp>

namespace cudf {
namespace io {
namespace parquet {

// Forward declare Parquet metadata parser
class ParquetMetadata;

/**
 * @brief Implementation for Parquet reader
 **/
class reader::Impl {
 public:
  /**
   * @brief Constructor from a dataset source with reader options.
   **/
  explicit Impl(std::unique_ptr<datasource> source, reader_options const &options);

  /**
   * @brief Returns the PANDAS-specific index column derived from the metadata.
   *
   * @return std::string Name of the column
   **/
  std::string get_index_column() const { return pandas_index_col_; }

  /**
   * @brief Read an entire set or a subset of data from the source and returns
   * an array of gdf_columns.
   *
   * @param[in] skip_rows Number of rows to skip from the start
   * @param[in] num_rows Number of rows to read; use `0` for all remaining data
   * @param[in] row_group Row group index to select
   *
   * @return cudf::table Object that contains the array of gdf_columns
   **/
  table read(int skip_rows, int num_rows, int row_group);

 private:
  /**
   * @brief Align a size such that aligned 32-bit loads within a memory block
   * won't read bytes beyond the unaligned cuda-memcheck limit
   *
   * @param[in] size in bytes
   *
   * @return size_t Size aligned to the next multiple of bytes needed by parquet kernels
   **/
  size_t align_size(size_t v) const { return util::round_up_safe(v, sizeof(uint32_t)); }

  /**
   * @brief Returns the number of total pages from the given column chunks
   *
   * @param[in] chunks List of column chunk descriptors
   *
   * @return size_t The total number of pages
   **/
  size_t count_page_headers(hostdevice_vector<parquet::gpu::ColumnChunkDesc> &chunks);

  /**
   * @brief Returns the page information from the given column chunks.
   *
   * @param[in] chunks List of column chunk descriptors
   * @param[in] pages List of page information
   **/
  void decode_page_headers(hostdevice_vector<parquet::gpu::ColumnChunkDesc> &chunks,
                           hostdevice_vector<parquet::gpu::PageInfo> &pages);

  /**
   * @brief Decompresses the page data, at page granularity.
   *
   * @param[in] chunks List of column chunk descriptors
   * @param[in] pages List of page information
   *
   * @return rmm::device_buffer Device buffer to decompressed page data
   **/
  rmm::device_buffer decompress_page_data(hostdevice_vector<parquet::gpu::ColumnChunkDesc> &chunks,
                                          hostdevice_vector<parquet::gpu::PageInfo> &pages);

  /**
   * @brief Converts the page data and outputs to gdf_columns.
   *
   * @param[in] chunks List of column chunk descriptors
   * @param[in] pages List of page information
   * @param[in] chunk_map Mapping between column chunk and gdf_column
   * @param[in] min_row Minimum number of rows to read from start
   * @param[in] total_rows Total number of rows to output
   **/
  void decode_page_data(hostdevice_vector<parquet::gpu::ColumnChunkDesc> &chunks,
                        hostdevice_vector<parquet::gpu::PageInfo> &pages,
                        const std::vector<gdf_column *> &chunk_map,
                        size_t min_row,
                        size_t total_rows);

 private:
  std::unique_ptr<datasource> source_;
  std::unique_ptr<ParquetMetadata> md_;

  std::string pandas_index_col_;
  std::vector<std::pair<int, std::string>> selected_cols_;
  bool strings_to_categorical_  = false;
  gdf_time_unit timestamp_unit_ = TIME_UNIT_NONE;
};

}  // namespace parquet
}  // namespace io
}  // namespace cudf
