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
