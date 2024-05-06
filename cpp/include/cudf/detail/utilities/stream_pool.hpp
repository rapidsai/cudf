/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <vector>

namespace cudf::detail {

class cuda_stream_pool {
 public:
  // matching type used in rmm::cuda_stream_pool::get_stream(stream_id)
  using stream_id_type = std::size_t;

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
  virtual rmm::cuda_stream_view get_stream(stream_id_type stream_id) = 0;

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
  virtual std::vector<rmm::cuda_stream_view> get_streams(std::size_t count) = 0;

  /**
   * @brief Get the number of unique stream objects in the pool.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return the number of stream objects in the pool
   */
  virtual std::size_t get_stream_pool_size() const = 0;
};

/**
 * @brief Initialize global stream pool.
 */
cuda_stream_pool* create_global_cuda_stream_pool();

/**
 * @brief Get the global stream pool.
 */
cuda_stream_pool& global_cuda_stream_pool();

/**
 * @brief Acquire a set of `cuda_stream_view` objects and synchronize them to an event on another
 * stream.
 *
 * By default an underlying `rmm::cuda_stream_pool` is used to obtain the streams. The only other
 * implementation at present is a debugging version that always returns the stream returned by
 * `cudf::get_default_stream()`. To use this debugging version, set the environment variable
 * `LIBCUDF_USE_DEBUG_STREAM_POOL`.
 *
 * Example usage:
 * @code{.cpp}
 * auto stream = cudf::get_default_stream();
 * auto const num_streams = 2;
 * // do work on stream
 * // allocate streams and wait for an event on stream before executing on any of streams
 * auto streams = cudf::detail::fork_stream(stream, num_streams);
 * // do work on streams[0] and streams[1]
 * // wait for event on streams before continuing to do work on stream
 * cudf::detail::join_streams(streams, stream);
 * @endcode
 *
 * @param stream Stream that the returned streams will wait on.
 * @param count The number of `cuda_stream_view` objects to return.
 * @return Vector containing `count` stream views.
 */
[[nodiscard]] std::vector<rmm::cuda_stream_view> fork_streams(rmm::cuda_stream_view stream,
                                                              std::size_t count);

/**
 * @brief Synchronize a stream to an event on a set of streams.
 *
 * @param streams Streams to wait on.
 * @param stream Joined stream that synchronizes with the waited-on streams.
 */
void join_streams(host_span<rmm::cuda_stream_view const> streams, rmm::cuda_stream_view stream);

}  // namespace cudf::detail
