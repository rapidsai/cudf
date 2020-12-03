/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cudf {
//! IO interfaces
namespace io {
/**
 * @brief Interface class for storing the output data from the writers
 **/
class data_sink {
 public:
  /**
   * @brief Create a sink from a file path
   *
   * @param[in] filepath Path to the file to use
   **/
  static std::unique_ptr<data_sink> create(const std::string& filepath);

  /**
   * @brief Create a sink from a std::vector
   *
   * @param[in,out] buffer Pointer to the output vector
   **/
  static std::unique_ptr<data_sink> create(std::vector<char>* buffer);

  /**
   * @brief Create a void sink (one that does no actual io)
   *
   * A useful code path for benchmarking, to eliminate physical
   * hardware randomness from profiling.
   *
   **/
  static std::unique_ptr<data_sink> create();

  /**
   * @brief Create a wrapped custom user data sink
   *
   * @param[in] User-provided data sink (typically custom class)
   *
   * The data sink returned here is not the one passed by the user. It is an internal
   * class that wraps the user pointer.  The principle is to allow the user to declare
   * a custom sink instance and use it across multiple write() calls.
   *
   **/
  static std::unique_ptr<data_sink> create(cudf::io::data_sink* const user_sink);

  /**
   * @brief Base class destructor
   **/
  virtual ~data_sink(){};

  /**
   * @brief Append the buffer content to the sink
   *
   * @param[in] data Pointer to the buffer to be written into the sink object
   * @param[in] size Number of bytes to write
   *
   * @return void
   **/
  virtual void host_write(void const* data, size_t size) = 0;

  /**
   * @brief Whether or not this sink supports writing from gpu memory addresses.
   *
   * Internal to some of the file format writers, we have code that does things like
   *
   * tmp_buffer = alloc_temp_buffer();
   * cudaMemcpy(tmp_buffer, device_buffer, size);
   * sink->write(tmp_buffer, size);
   *
   * In the case where the sink type is itself a memory buffered write, this ends up
   * being effectively a second memcpy.  So a useful optimization for a "smart"
   * custom data_sink is to do it's own internal management of the movement
   * of data between cpu and gpu; turning the internals of the writer into simply
   *
   * sink->device_write(device_buffer, size)
   *
   * If this function returns true, the data_sink will receive calls to device_write()
   * instead of write() when possible.  However, it is still possible to receive
   * write() calls as well.
   *
   * @return bool If this writer supports device_write() calls.
   **/
  virtual bool supports_device_write() const { return false; }

  /**
   * @brief Append the buffer content to the sink from a gpu address
   *
   * @param[in] data Pointer to the buffer to be written into the sink object
   * @param[in] size Number of bytes to write
   *
   * @return void
   **/
  virtual void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream)
  {
    CUDF_FAIL("data_sink classes that support device_write must override this function.");
  }

  /**
   * @brief Flush the data written into the sink
   *
   * @return void
   */
  virtual void flush() = 0;

  /**
   * @brief Returns the total number of bytes written into this sink
   *
   * @return size_t Total number of bytes written into this sink
   **/
  virtual size_t bytes_written() = 0;
};

}  // namespace io
}  // namespace cudf
