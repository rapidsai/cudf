/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <BS_thread_pool.hpp>

namespace cudf::detail {

/**
 * @brief Retrieves a reference to the global host worker thread pool.
 *
 * This function returns a reference to a thread pool that can be used for executing host-only
 * tasks. The pool size is potentially not optimal for tasks that include device operations, like
 * copies between host and device and kernel calls.
 *
 * @return A reference to the host worker thread pool.
 */
BS::thread_pool& host_worker_pool();

}  // namespace cudf::detail
