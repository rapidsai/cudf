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
 * @brief cuDF-IO ORC reader class implementation header
 */

#pragma once

#include "orc.h"
#include "orc_gpu.h"

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
namespace orc {

using namespace cudf::io::orc;
using namespace cudf::io;

// Forward declarations
class metadata;
namespace {
class orc_stream_info;
}

/**
 * @brief Implementation for ORC reader
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
   * @brief Read an entire set or a subset of data and returns a set of columns
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to read
   * @param stripe Stripe index to select
   * @param stream Stream to use for memory allocation and kernels
   *
   * @return The set of columns along with metadata
   */
  table_with_metadata read(int skip_rows, int num_rows, int stripe,
                           cudaStream_t stream);

 private:
  /**
   * @brief Decompresses the stripe data, at stream granularity
   *
   * @param chunks List of column chunk descriptors
   * @param stripe_data List of source stripe column data
   * @param decompressor Originally host decompressor
   * @param stream_info List of stream to column mappings
   * @param num_stripes Number of stripes making up column chunks
   * @param row_groups List of row index descriptors
   * @param row_index_stride Distance between each row index
   * @param stream Stream to use for memory allocation and kernels
   *
   * @return Device buffer to decompressed page data
   */
  rmm::device_buffer decompress_stripe_data(
      const hostdevice_vector<gpu::ColumnDesc> &chunks,
      const std::vector<rmm::device_buffer> &stripe_data,
      const OrcDecompressor *decompressor,
      std::vector<orc_stream_info> &stream_info, size_t num_stripes,
      rmm::device_vector<gpu::RowGroup> &row_groups, size_t row_index_stride,
      cudaStream_t stream);

  /**
   * @brief Converts the stripe column data and outputs to columns
   *
   * @param chunks List of column chunk descriptors
   * @param num_dicts Number of dictionary entries required
   * @param skip_rows Number of rows to offset from start
   * @param num_rows Number of rows to output
   * @param timezone_table Local time to UTC conversion table
   * @param row_groups List of row index descriptors
   * @param row_index_stride Distance between each row index
   * @param out_buffers Output columns' device buffers
   * @param stream Stream to use for memory allocation and kernels
   */
  void decode_stream_data(const hostdevice_vector<gpu::ColumnDesc> &chunks,
                          size_t num_dicts, size_t skip_rows, size_t num_rows,
                          const std::vector<int64_t> &timezone_table,
                          const rmm::device_vector<gpu::RowGroup> &row_groups,
                          size_t row_index_stride,
                          std::vector<column_buffer> &out_buffers,
                          cudaStream_t stream);

 private:
  rmm::mr::device_memory_resource *_mr = nullptr;
  std::unique_ptr<datasource> _source;
  std::unique_ptr<metadata> _metadata;

  std::vector<int> _selected_columns;
  bool _use_index = true;
  bool _use_np_dtypes = true;
  bool _has_timestamp_column = false;
  bool _decimals_as_float = true;
  int _decimals_as_int_scale = -1;
  data_type _timestamp_type{type_id::EMPTY};
};

}  // namespace orc
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
