/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_buffer.hpp>

namespace cudf {
namespace io {
namespace text {

/**
 * @brief represents a possibly-shared view over device memory.
 */
struct data_chunk {
  data_chunk(device_span<char const> data) : _data(data) {}

  operator cudf::device_span<char const>() { return _data; }

  uint32_t size() const { return _data.size(); }

 private:
  device_span<char const> _data;
};

/**
 * @brief a reader capable of producing views over device memory
 *
 */
class data_chunk_reader {
 public:
  /**
   * @brief Get the next chunk of bytes from the data source
   *
   * Performs any necessary work to read and prepare the underlying data source for consumption as a
   * view over device memory. Common implementations may read from a file, copy data from host
   * memory, allocate temporary memory, perform iterative decompression, or even launch device
   * kernels.
   *
   * @param size desired number of bytes
   * @param stream stream to associate allocations or perform work required to obtain chunk
   * @return a chunk of data up to @param size bytes, or less if no more data is avaialable
   */
  virtual data_chunk get_next_chunk(uint32_t size, rmm::cuda_stream_view stream) = 0;
};

/**
 * @brief a data source capable of creating a reader which can produce views of the data source in
 * device memory.
 *
 */
class data_chunk_source {
 public:
  virtual std::unique_ptr<data_chunk_reader> create_reader() = 0;
};

}  // namespace text
}  // namespace io
}  // namespace cudf
