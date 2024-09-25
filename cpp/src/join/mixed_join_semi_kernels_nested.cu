/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "join/mixed_join_common_utils.cuh"
#include "join/mixed_join_semi_kernels.cuh"
#include "join/mixed_join_semi_kernels.hpp"

namespace cudf {
namespace detail {
template void launch_mixed_join_semi<row_hash>(
  bool has_nulls,
  table_device_view left_table,
  table_device_view right_table,
  table_device_view probe,
  table_device_view build,
  row_hash const hash_probe,
  row_equality const equality_probe,
  cudf::detail::semi_map_type::device_view hash_table_view,
  cudf::device_span<bool> left_table_keep_mask,
  cudf::ast::detail::expression_device_view device_expression_data,
  detail::grid_1d const config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream);
}  // namespace detail
}  // namespace cudf
