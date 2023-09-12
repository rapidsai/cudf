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

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <vector>

namespace cudf::detail {

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
