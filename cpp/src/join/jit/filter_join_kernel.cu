/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/std/cstddef>

#include <jit/accessors.cuh>
#include <jit/span.cuh>

// clang-format off
// This header is an inlined header that defines the GENERIC_JOIN_FILTER_OP function. It is placed here
// so the symbols in the headers above can be used by it.
#include <cudf/detail/operation-udf.hpp>
// clang-format on

namespace cudf {
namespace join {
namespace jit {

template <bool has_user_data, typename... InputAccessors>
CUDF_KERNEL void filter_join_kernel(cudf::device_span<cudf::size_type const> left_indices,
                                   cudf::device_span<cudf::size_type const> right_indices,
                                   cudf::column_device_view_core const* left_tables,
                                   cudf::column_device_view_core const* right_tables,
                                   bool* predicate_results,
                                   void* user_data)
{
  auto const start = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  auto const size = left_indices.size();

  for (auto i = start; i < size; i += stride) {
    auto const left_idx = left_indices[i];
    auto const right_idx = right_indices[i];
    
    // Skip if either index is JoinNoMatch
    if (left_idx == cudf::JoinNoMatch || right_idx == cudf::JoinNoMatch) {
      predicate_results[i] = false;
      continue;
    }

    bool result = false;
    
    if constexpr (has_user_data) {
      GENERIC_JOIN_FILTER_OP(user_data, i, &result, 
                            InputAccessors::element(left_tables, left_idx, i)...,
                            InputAccessors::element(right_tables, right_idx, i)...);
    } else {
      GENERIC_JOIN_FILTER_OP(&result,
                           InputAccessors::element(left_tables, left_idx, i)...,
                           InputAccessors::element(right_tables, right_idx, i)...);
    }
    
    predicate_results[i] = result;
  }
}

}  // namespace jit
}  // namespace join
}  // namespace cudf
