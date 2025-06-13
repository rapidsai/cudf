/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/io/thread.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>

#include <kvikio/defaults.hpp>

namespace cudf::io::detail {
void set_num_io_threads(unsigned int num_io_threads)
{
  CUDF_EXPECTS(num_io_threads > 0, "The number of I/O threads must be positive.");
  auto old_setting = kvikio::defaults::thread_pool_nthreads();
  if (num_io_threads != old_setting) {
    kvikio::defaults::set_thread_pool_nthreads(num_io_threads);
    CUDF_LOG_INFO(
      "Set the number of I/O threads. Old value: %u, new value: %u. Thread pool re-created.",
      old_setting,
      num_io_threads);
  } else {
    CUDF_LOG_INFO(
      "Set the number of I/O threads. Setting unchanged: %u threads. Reusing the existing thread "
      "pool",
      old_setting);
  }
}

[[nodiscard]] unsigned int num_io_threads() { return kvikio::defaults::thread_pool_nthreads(); }

}  // namespace cudf::io::detail
