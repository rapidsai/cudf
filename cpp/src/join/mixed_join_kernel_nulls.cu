/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "mixed_join_kernel.cuh"
#include "mixed_join_kernel.hpp"

namespace cudf::detail {

template void launch_mixed_join<true>(
  table_device_view left_table,
  table_device_view right_table,
  bool is_outer_join,
  bool swap_tables,
  row_equality equality_probe,
  cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
  cuco::pair<hash_value_type, cudf::size_type> const* input_pairs,
  cuda::std::pair<uint32_t, uint32_t> const* hash_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  size_type* join_output_l,
  size_type* join_output_r,
  cudf::size_type const* join_result_offsets,
  detail::grid_1d config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream);

}  // namespace cudf::detail
