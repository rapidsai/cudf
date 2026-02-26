/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/ast/detail/operator_functor.cuh>
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/limits>

#include <jit/accessors.cuh>
#include <jit/span.cuh>

#pragma nv_hdrstop  // The above headers are used by the kernel below and need to be included before
                    // it. Each UDF will have a different operation-udf.hpp generated for it, so we
                    // need to put this pragma before including it to avoid PCH mismatch.

// clang-format off
// This header is an inlined header that defines the GENERIC_JOIN_FILTER_OP function. It is placed here
// so the symbols in the headers above can be used by it.
#include <cudf/detail/operation-udf.hpp>
// clang-format on

namespace cudf::join::jit {

// TODO: Create a JIT-compatible header for JoinNoMatch to avoid this duplication.
// This must match the definition in cudf/join/join.hpp
constexpr cudf::size_type JoinNoMatch = cuda::std::numeric_limits<cudf::size_type>::min();

template <bool has_user_data, bool has_nulls, typename... InputAccessors>
CUDF_KERNEL void filter_join_kernel(cudf::jit::device_span<cudf::size_type const> left_indices,
                                    cudf::jit::device_span<cudf::size_type const> right_indices,
                                    cudf::column_device_view_core const* left_tables,
                                    cudf::column_device_view_core const* right_tables,
                                    bool* predicate_results,
                                    void* user_data)
{
  auto const start  = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  auto const size   = left_indices.size();

  for (auto i = start; i < size; i += stride) {
    auto const left_idx  = left_indices[i];
    auto const right_idx = right_indices[i];

    // Skip if either index is JoinNoMatch
    if (left_idx == JoinNoMatch || right_idx == JoinNoMatch) {
      predicate_results[i] = false;
      continue;
    }

    // Each accessor receives both tables and both indices, and internally selects
    // the appropriate table based on whether it's a left or right accessor.
    if constexpr (has_nulls) {
      // Null-aware path: pass optional<T> inputs, get optional<bool> result
      cuda::std::optional<bool> result{false};
      if constexpr (has_user_data) {
        GENERIC_JOIN_FILTER_OP(
          user_data,
          i,
          &result,
          InputAccessors::nullable_element(left_tables, right_tables, left_idx, right_idx, i)...);
      } else {
        GENERIC_JOIN_FILTER_OP(
          &result,
          InputAccessors::nullable_element(left_tables, right_tables, left_idx, right_idx, i)...);
      }
      predicate_results[i] = result.has_value() && result.value();
    } else {
      // Non-null-aware path: if any input is null, predicate is false
      if ((InputAccessors::is_null(left_tables, right_tables, left_idx, right_idx, i) || ...)) {
        predicate_results[i] = false;
        continue;
      }
      bool result = false;
      if constexpr (has_user_data) {
        GENERIC_JOIN_FILTER_OP(
          user_data,
          i,
          &result,
          InputAccessors::element(left_tables, right_tables, left_idx, right_idx, i)...);
      } else {
        GENERIC_JOIN_FILTER_OP(
          &result,
          InputAccessors::element(left_tables, right_tables, left_idx, right_idx, i)...);
      }
      predicate_results[i] = result;
    }
  }
}

}  // namespace cudf::join::jit
