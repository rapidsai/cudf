/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <mutex>

namespace cudf {

namespace detail {

/**
 * @brief Default stream for cudf
 *
 * Use this value to ensure the correct stream is used when compiled with per
 * thread default stream.
 */
extern rmm::cuda_stream_view default_stream_value;

/**
 * @brief Lock for setting the stream.
 */
inline std::mutex& stream_lock()
{
  static std::mutex _stream_lock;
  return _stream_lock;
}

/**
 * @brief Lock for setting the stream.
 *
 * @param new_default_stream The stream view to use as the new cudf default.
 */
inline void set_default_stream(rmm::cuda_stream_view new_default_stream)
{
  std::lock_guard<std::mutex> lock{stream_lock()};
  default_stream_value = new_default_stream;
}

}  // namespace detail

}  // namespace cudf
