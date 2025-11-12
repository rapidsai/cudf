/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
