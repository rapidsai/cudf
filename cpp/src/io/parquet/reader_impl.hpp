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

/**
 * @file reader_impl.hpp
 * @brief cuDF-IO Parquet reader class implementation header
 */

#pragma once

#include "parquet.h"
#include "parquet_gpu.h"

#include <io/utilities/column_buffer.hpp>
#include <io/utilities/datasource.hpp>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/io/readers.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cudf {
namespace experimental {
namespace io {
namespace detail {
namespace parquet {

using namespace cudf::io::parquet;
using namespace cudf::io;

// Forward declarations
class metadata;

/**
 * @brief Implementation for Parquet reader
 */
class reader::impl {
 public:
  /**
   * @brief Constructor from a dataset source with reader options.
   *
   * @param source Dataset source
   * @param options Settings for controlling reading behavior
   * @param mr Resource to use for device memory allocation
   */
  explicit impl(std::unique_ptr<datasource> source,
                reader_options const &options,
                rmm::mr::device_memory_resource *mr);

  /**
   * @brief Returns the PANDAS-specific index column derived from the metadata.
   *
   * @return Name of the column
   */
  std::string get_pandas_index() const { return _pandas_index; }

  /**
   * @brief Read an entire set or a subset of data and returns a set of columns
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to read
   * @param row_group Row group index to select
   * @param stream Stream to use for memory allocation and kernels
   *
   * @return The set of columns along with metadata
   */
  table_with_metadata read(int skip_rows, int num_rows, int row_group,
                           cudaStream_t stream);

 private:
  /**
   * @brief Returns the number of total pages from the given column chunks
   *
   * @param chunks List of column chunk descriptors
   * @param stream Stream to use for memory allocation and kernels
   *
   * @return The total number of pages
   */
  size_t count_page_headers(
      const hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
      cudaStream_t stream);

  /**
   * @brief Returns the page information from the given column chunks.
   *
   * @param chunks List of column chunk descriptors
   * @param pages List of page information
   * @param stream Stream to use for memory allocation and kernels
   */
  void decode_page_headers(
      const hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
      const hostdevice_vector<gpu::PageInfo> &pages, cudaStream_t stream);

  /**
   * @brief Decompresses the page data, at page granularity.
   *
   * @param chunks List of column chunk descriptors
   * @param pages List of page information
   * @param stream Stream to use for memory allocation and kernels
   *
   * @return Device buffer to decompressed page data
   */
  rmm::device_buffer decompress_page_data(
      const hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
      const hostdevice_vector<gpu::PageInfo> &pages, cudaStream_t stream);

  /**
   * @brief Converts the page data and outputs to columns.
   *
   * @param chunks List of column chunk descriptors
   * @param pages List of page information
   * @param min_row Minimum number of rows from start
   * @param total_rows Number of rows to output
   * @param chunk_map Mapping between chunk and column
   * @param out_buffers Output columns' device buffers
   * @param stream Stream to use for memory allocation and kernels
   */
  void decode_page_data(const hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                        const hostdevice_vector<gpu::PageInfo> &pages,
                        size_t min_row, size_t total_rows,
                        const std::vector<int> &chunk_map,
                        std::vector<column_buffer> &out_buffers,
                        cudaStream_t stream);

 private:
  rmm::mr::device_memory_resource *_mr = nullptr;
  std::unique_ptr<datasource> _source;
  std::unique_ptr<metadata> _metadata;

  std::vector<std::pair<int, std::string>> _selected_columns;
  std::string _pandas_index;
  bool _strings_to_categorical = false;
  data_type _timestamp_type{type_id::EMPTY};
};

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
