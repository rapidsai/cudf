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

#include "compute_column_kernel.cuh"
#include "compute_column_kernel.hpp"

namespace cudf::detail {
template void launch_compute_column_kernel<true, false>(
  table_device_view const& table_device,
  ast::detail::expression_device_view device_expression_data,
  mutable_column_device_view& mutable_output_device,
  cudf::detail::grid_1d const& config,
  size_t shmem_per_block,
  rmm::cuda_stream_view stream);
}  // namespace cudf::detail
