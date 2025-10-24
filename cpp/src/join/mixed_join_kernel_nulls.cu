/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
  cuda::std::pair<hash_value_type, hash_value_type> const* hash_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  size_type* join_output_l,
  size_type* join_output_r,
  cudf::size_type const* join_result_offsets,
  detail::grid_1d config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream);

}  // namespace cudf::detail
