/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mixed_join_size_kernel.cuh"
#include "mixed_join_size_kernel.hpp"

namespace cudf {
namespace detail {

template std::size_t launch_compute_mixed_join_output_size<false>(
  cudf::table_device_view left_table,
  cudf::table_device_view right_table,
  cudf::table_device_view probe,
  cudf::table_device_view build,
  row_hash const hash_probe,
  row_equality const equality_probe,
  join_kind const join_type,
  cudf::detail::mixed_multimap_type::device_view hash_table_view,
  ast::detail::expression_device_view device_expression_data,
  bool const swap_tables,
  cudf::device_span<cudf::size_type> matches_per_row,
  detail::grid_1d const config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
