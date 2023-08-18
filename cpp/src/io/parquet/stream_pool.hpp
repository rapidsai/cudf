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

#include <rmm/cuda_stream_pool.hpp>

namespace cudf::io::detail::parquet {

/**
 * @brief Get a `cuda_stream_view` of a stream in the pool.
 *
 * This function is thread safe with respect to other calls to the same function.
 *
 * @return Stream view.
 */
rmm::cuda_stream_view get_stream();

/**
 * @brief Get a `cuda_stream_view` of the stream associated with `stream_id`.
 *
 * Equivalent values of `stream_id` return a stream_view to the same underlying stream.
 * This function is thread safe with respect to other calls to the same function.
 *
 * @param stream_id Unique identifier for the desired stream
 * @return Requested stream view.
 */
rmm::cuda_stream_view get_stream(std::size_t stream_id);

/**
 * @brief Get a set of `cuda_stream_view` objects from the pool.
 *
 * This function is thread safe with respect to other calls to the same function.
 *
 * @param count The number of stream views to return.
 * @return Vector containing `count` stream views.
 */
std::vector<rmm::cuda_stream_view> get_streams(uint32_t count);

/**
 * @brief Synchronize a set of streams to an event on another stream.
 *
 * @param streams Vector of streams to synchronize on.
 * @param stream Stream to synchronize the other streams to, usually the default stream.
 */
void fork_streams(std::vector<rmm::cuda_stream_view>& streams, rmm::cuda_stream_view stream);

/**
 * @brief Synchronize a stream to an event on a set of streams.
 *
 * @param streams Vector of streams to synchronize on.
 * @param stream Stream to synchronize the other streams to, usually the default stream.
 */
void join_streams(std::vector<rmm::cuda_stream_view>& streams, rmm::cuda_stream_view stream);

/**
 * @brief Get the number of streams in the pool.
 *
 * This function is thread safe with respect to other calls to the same function.
 *
 * @return the number of streams in the pool
 */
std::size_t get_stream_pool_size();

}  // namespace cudf::io::detail::parquet
