/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <type_traits>

namespace cudf {
namespace detail {

/**
 * @brief Binary `argmin`/`argmax` operator
 *
 * @tparam T Type of the underlying column. Must support '<' operator.
 */
template <typename T>
struct element_argminmax_fn {
  column_device_view const d_col;
  bool const has_nulls;
  bool const arg_min;

  __device__ inline size_type operator()(size_type const& lhs_idx, size_type const& rhs_idx) const
  {
    // The extra bounds checking is due to issue github.com/rapidsai/cudf/9156 and
    // github.com/NVIDIA/thrust/issues/1525
    // where invalid random values may be passed here by thrust::reduce_by_key
    auto out_of_bound_or_null = [this] __device__(size_type const& idx) {
      return idx < 0 || idx >= this->d_col.size() ||
             (this->has_nulls && this->d_col.is_null_nocheck(idx));
    };
    if (out_of_bound_or_null(lhs_idx)) { return rhs_idx; }
    if (out_of_bound_or_null(rhs_idx)) { return lhs_idx; }

    // Return `lhs_idx` iff:
    //   row(lhs_idx) <  row(rhs_idx) and finding ArgMin, or
    //   row(lhs_idx) >= row(rhs_idx) and finding ArgMax.
    auto const less = d_col.element<T>(lhs_idx) < d_col.element<T>(rhs_idx);
    return less == arg_min ? lhs_idx : rhs_idx;
  }
};

}  // namespace detail
}  // namespace cudf
