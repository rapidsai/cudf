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

#include "io/utilities/getenv_or.hpp"

#include <cudf/detail/utilities/host_worker_pool.hpp>

namespace cudf::detail {

BS::thread_pool& host_worker_pool()
{
  static const std::size_t default_pool_size =
    std::min(32u, std::thread::hardware_concurrency() / 2);
  static const std::size_t pool_size = getenv_or("LIBCUDF_NUM_HOST_WORKERS", default_pool_size);
  static BS::thread_pool pool(pool_size);
  return pool;
}

}  // namespace cudf::detail
