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
 * @brief cuDF-IO Avro reader class implementation header
 */

#pragma once

#include "avro.h"
#include "avro_gpu.h"

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
namespace avro {

using namespace cudf::io::avro;
using namespace cudf::io;

// Forward declarations
class metadata;

/**
 * @brief Implementation for Avro reader
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
   * @param stream Stream to use for memory allocation and kernels
   *
   * @return The set of columns along with metadata
   */
  table_with_metadata read(int skip_rows, int num_rows,
                           cudaStream_t stream);

 private:
  /**
   * @brief Decompresses the block data.
   *
   * @param comp_block_data Compressed block data
   * @param stream Stream to use for memory allocation and kernels
   *
   * @return Device buffer to decompressed block data
   */
  rmm::device_buffer decompress_data(const rmm::device_buffer &comp_block_data,
                                     cudaStream_t stream);

  /**
   * @brief Convert the avro row-based block data and outputs to columns
   *
   * @param block_data Uncompressed block data
   * @param dict Dictionary entries
   * @param global_dictionary Dictionary allocation
   * @param total_dictionary_entries Number of dictionary entries
   * @param out_buffers Output columns' device buffers
   * @param stream Stream to use for memory allocation and kernels
   */
  void decode_data(const rmm::device_buffer &block_data,
                   const std::vector<std::pair<uint32_t, uint32_t>> &dict,
                   const hostdevice_vector<uint8_t> &global_dictionary,
                   size_t total_dictionary_entries, size_t num_rows,
                   std::vector<std::pair<int, std::string>> columns,
                   std::vector<column_buffer> &out_buffers,
                   cudaStream_t stream);

 private:
  rmm::mr::device_memory_resource *_mr = nullptr;
  std::unique_ptr<datasource> _source;
  std::unique_ptr<metadata> _metadata;

  std::vector<std::string> _columns;
};

}  // namespace avro
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
