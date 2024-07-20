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

#include "mixed_join_size_kernel.cuh"

namespace cudf {
namespace detail {

template __global__ void compute_mixed_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, true>(
  table_device_view left_table,
  table_device_view right_table,
  table_device_view probe,
  table_device_view build,
  row_hash const hash_probe,
  row_equality const equality_probe,
  join_kind const join_type,
  cudf::detail::mixed_multimap_type::device_view hash_table_view,
  cudf::ast::expression_device_view device_expression_data,
  bool const swap_tables,
  std::size_t* output_size,
  cudf::device_span<cudf::size_type> matches_per_row);

}  // namespace detail
}  // namespace cudf
