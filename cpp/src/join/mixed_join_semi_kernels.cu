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

#include "mixed_join_common_utils.cuh"
#include "mixed_join_semi_kernels.cuh"
#include "mixed_join_semi_kernels.hpp"

#include <cudf/table/experimental/row_operators.cuh>

#include <cuco/static_set.cuh>

namespace {
using row_comparator_type = cudf::experimental::row::equality::strong_index_comparator_adapter<
  cudf::experimental::row::equality::device_row_comparator<
    false,
    cudf::nullate::DYNAMIC,
    cudf::experimental::row::equality::nan_equal_physical_equality_comparator,
    cudf::experimental::dispatch_void_if_compound_t>>;
using hash_set_ref_type = cuco::static_set<
  cudf::size_type,
  cuco::extent<size_t>,
  cuda::thread_scope_device,
  cudf::detail::double_row_equality_comparator<row_comparator_type, row_comparator_type>,
  cuco::linear_probing<cudf::detail::DEFAULT_MIXED_JOIN_CG_SIZE,
                       cudf::detail::row_hash_no_compound>,
  cudf::detail::cuco_allocator<char>,
  cuco::storage<1>>::ref_type<cuco::contains_tag>;
}  // namespace

namespace cudf {
namespace detail {
template void launch_mixed_join_semi<hash_set_ref_type>(
  bool has_nulls,
  table_device_view left_table,
  table_device_view right_table,
  table_device_view probe,
  table_device_view build,
  row_equality const equality_probe,
  hash_set_ref_type const& set_ref,
  cudf::device_span<bool> left_table_keep_mask,
  cudf::ast::detail::expression_device_view device_expression_data,
  detail::grid_1d const config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace cudf
