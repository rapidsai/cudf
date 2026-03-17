/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../join_common_utils.cuh"
#include "common.cuh"

#include <cudf/detail/algorithms/reduce.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/scatter.h>
#include <thrust/uninitialized_fill.h>

#include <memory>

namespace cudf::detail {

std::size_t compute_left_join_complement_size(
  std::unique_ptr<rmm::device_uvector<size_type>>& right_indices,
  size_type left_table_row_count,
  size_type right_table_row_count,
  rmm::cuda_stream_view stream)
{
  if (left_table_row_count == 0) { return right_table_row_count; }

  auto invalid_index_map =
    std::make_unique<rmm::device_uvector<size_type>>(right_table_row_count, stream);
  thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                             invalid_index_map->begin(),
                             invalid_index_map->end(),
                             int32_t{1});

  valid_range<size_type> valid(0, right_table_row_count);

  thrust::scatter_if(rmm::exec_policy_nosync(stream),
                     cuda::make_constant_iterator(0),
                     cuda::make_constant_iterator(0) + right_indices->size(),
                     right_indices->begin(),
                     right_indices->begin(),
                     invalid_index_map->begin(),
                     valid);

  return cudf::detail::count_if(
    invalid_index_map->begin(), invalid_index_map->end(), cuda::std::identity{}, stream);
}

}  // namespace cudf::detail
