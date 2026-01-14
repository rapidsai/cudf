/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/utilities/default_stream.hpp>

#include <cstdlib>

namespace cudf {

namespace detail {

#if defined(CUDF_USE_PER_THREAD_DEFAULT_STREAM)
rmm::cuda_stream_view const default_stream_value{rmm::cuda_stream_per_thread};
#else
rmm::cuda_stream_view const default_stream_value{};
#endif

}  // namespace detail

/**
 * @brief Check if per-thread default stream is enabled.
 *
 * @return true if PTDS is enabled, false otherwise.
 */
bool is_ptds_enabled()
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return true;
#else
  return false;
#endif
}

rmm::cuda_stream_view const get_default_stream()
{
  static auto const default_stream = []() {
    if (std::getenv("CUDF_PER_THREAD_STREAM") != nullptr) {
      return rmm::cuda_stream_per_thread;
    } else {
      return detail::default_stream_value;
    }
  }();
  return default_stream;
}
}  // namespace cudf
