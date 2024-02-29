/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>

#include <cuco/static_map.cuh>
#include <cuda/std/atomic>

#include <limits>

namespace cudf {
namespace detail {

using hash_map_type = cuco::legacy::
  static_map<size_type, size_type, cuda::thread_scope_device, cudf::detail::cuco_allocator>;

}  // namespace detail
}  // namespace cudf
