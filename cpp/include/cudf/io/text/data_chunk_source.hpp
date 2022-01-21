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
 * @brief A contract guaranteeing stream-ordered memory access to the underlying device data.
 *
 * This class guarantees access to the underlying data for the stream on which the data was
 * allocated. Possible implementations may own the device data, or may only have a view over the
 * data. Any work enqueued to the stream on which this data was allocated is guaranteed to be
 * performed prior to the destruction of the underlying data, but otherwise no guarantees are made
 * regarding if or when the underlying data gets destroyed.
 */
class device_data_chunk {
 public:
  [[nodiscard]] virtual char const* data() const   = 0;
  [[nodiscard]] virtual std::size_t size() const   = 0;
  virtual operator device_span<char const>() const = 0;
};

/**
 * @brief a reader capable of producing views over device memory.
 *
 * The data chunk reader API encapsulates the idea of statefully traversing and loading a data
 * source. A data source may be a file, a region of device memory, or a region of host memory.
 * Reading data from these data sources efficiently requires different strategies dependings on the
 * type of data source, type of compression, capabilities of the host and device, the data's
 * destination. Whole-file decompression should be hidden behind this interface.
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
   * @param size number of bytes to read.
   * @param stream stream to associate allocations or perform work required to obtain chunk
   * @return a chunk of data up to @p size bytes. May return less than @p size bytes if
   * reader reaches end of underlying data source. Returned data must be accessed in stream order
   * relative to the specified @p stream
   */
  virtual std::unique_ptr<device_data_chunk> get_next_chunk(std::size_t size,
                                                            rmm::cuda_stream_view stream) = 0;
};

/**
 * @brief a data source capable of creating a reader which can produce views of the data source in
 * device memory.
 */
class data_chunk_source {
 public:
  [[nodiscard]] virtual std::unique_ptr<data_chunk_reader> create_reader() const = 0;
};

}  // namespace text
}  // namespace io
}  // namespace cudf
