/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

namespace cudf::detail {

/**
 * @brief A pool of CUDA stream objects
 *
 * Provides efficient access to a collection of asynchronous (i.e. non-default) CUDA stream objects.
 *
 * The default implementation uses an underlying `rmm::cuda_stream_pool`. The only other
 * implementation at present is a debugging version that always returns the stream returned by
 * `cudf::get_default_stream()`. To use this debugging version, set the environment variable
 * `LIBCUDF_USE_DEBUG_STREAM_POOL`.
 *
 * Access to the global `cuda_stream_pool` is granted via `cudf::detail::global_cuda_stream_pool()`.
 *
 * Example usage:
 * @code{.cpp}
 * auto stream = cudf::get_default_stream();
 * auto const num_streams = 2;
 * // do work on stream
 * // allocate streams and wait for an event on stream before executing on any of streams
 * auto streams = cudf::detail::fork_streams(stream, num_streams);
 * // do work on streams[0] and streams[1]
 * // wait for event on streams before continuing to do work on stream
 * cudf::detail::join_streams(streams, stream);
 * @endcode
 */
class cuda_stream_pool {
 public:
  virtual ~cuda_stream_pool() = default;

  /**
   * @brief Get a `cuda_stream_view` of a stream in the pool.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return Stream view.
   */
  virtual rmm::cuda_stream_view get_stream() = 0;

  /**
   * @brief Get a `cuda_stream_view` of the stream associated with `stream_id`.
   *
   * Equivalent values of `stream_id` return a `cuda_stream_view` to the same underlying stream.
   * This function is thread safe with respect to other calls to the same function.
   *
   * @param stream_id Unique identifier for the desired stream
   * @return Requested stream view.
   */
  virtual rmm::cuda_stream_view get_stream(std::size_t stream_id) = 0;

  /**
   * @brief Get a set of `cuda_stream_view` objects from the pool.
   *
   * An attempt is made to ensure that the returned vector does not contain duplicate
   * streams, but this cannot be guaranteed if `count` is greater than the value returned by
   * `get_stream_pool_size()`.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @param count The number of stream views to return.
   * @return Vector containing `count` stream views.
   */
  virtual std::vector<rmm::cuda_stream_view> get_streams(uint32_t count) = 0;

  /**
   * @brief Get the number of stream objects in the pool.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return the number of stream objects in the pool
   */
  virtual std::size_t get_stream_pool_size() const = 0;
};

/**
 * @brief Return a reference to the global `cuda_stream_pool` object.
 *
 * @return The cuda_stream_pool singleton.
 */
cuda_stream_pool& global_cuda_stream_pool();

/**
 * @brief Acquire a set of `cuda_stream_view` objects and synchronize them to an event on another
 * stream.
 *
 * @param stream Stream to synchronize the returned streams to, usually the default stream.
 * @param count The number of stream views to return.
 * @return Vector containing `count` stream views.
 */
std::vector<rmm::cuda_stream_view> fork_streams(rmm::cuda_stream_view stream, uint32_t count);

/**
 * @brief Synchronize a stream to an event on a set of streams.
 *
 * @param streams Vector of streams to synchronize to.
 * @param stream Stream to synchronize, usually the default stream.
 */
void join_streams(host_span<rmm::cuda_stream_view> streams, rmm::cuda_stream_view stream);

}  // namespace cudf::detail
