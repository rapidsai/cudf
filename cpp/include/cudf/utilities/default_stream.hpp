/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

namespace cudf {

namespace detail {

// TODO: I've moved the default_stream_value to the detail namespace now and
// only expose the getter publicly. Assuming that reviewers are OK with this
// structure, we probably should move these to a different header file.

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

// TODO: For now, we only support setting a single stream for all threads. Do
// we want to support a stream per thread (manual PTDS)?
// TODO: Should this function be responsible for accepting a stream (not a
// view) and keeping that stream alive?
// TODO: I don't think we actually want to support letting users set a new
// default stream yet, so I'm putting this in detail for now.

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

/**
 * @brief Get the current default stream
 *
 * @return The current default stream.
 */
inline rmm::cuda_stream_view const get_default_stream()
{
  std::lock_guard<std::mutex> lock{detail::stream_lock()};
  return detail::default_stream_value;
}

/**
 * @brief Check if per-thread default stream is enabled.
 *
 * @return true if PTDS is enabled, false otherwise.
 */
bool is_ptds_enabled();

}  // namespace cudf
