/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/table/row_operators.cuh>

namespace cudf {
namespace reduction {
namespace detail {

/**
 * @brief Binary operator ArgMin/ArgMax with index values into the input table.
 *
 * @tparam T Type of the underlying data. This is the fallback for the cases when T does not support
 * '<' operator.
 */
template <bool has_nulls>
struct row_arg_minmax_fn {
  size_type const num_rows;
  row_lexicographic_comparator<has_nulls> const comp;
  bool const arg_min;

  row_arg_minmax_fn(size_type const num_rows_,
                    table_device_view const& table_,
                    null_order const* null_precedence_,
                    bool const arg_min_)
    : num_rows(num_rows_), comp(table_, table_, nullptr, null_precedence_), arg_min(arg_min_)
  {
  }

  // This function is explicitly prevented from inlining, because it calls to
  // `row_lexicographic_comparator::operator()` which is inlined and very heavy-weight. As a result,
  // instantiating this functor will result in huge code, and objects of this functor used with
  // `thrust::reduce_by_key` or `thrust::scan_by_key` will result in significant compile time.
  __attribute__((noinline)) __device__ auto operator()(size_type lhs_idx, size_type rhs_idx) const
  {
    // The extra bounds checking is due to issue github.com/rapidsai/cudf/9156 and
    // github.com/NVIDIA/thrust/issues/1525
    // where invalid random values may be passed here by thrust::reduce_by_key
    if (lhs_idx < 0 || lhs_idx >= num_rows) { return rhs_idx; }
    if (rhs_idx < 0 || rhs_idx >= num_rows) { return lhs_idx; }

    // Return `lhs_idx` iff:
    //   row(lhs_idx) <  row(rhs_idx) and finding ArgMin, or
    //   row(lhs_idx) >= row(rhs_idx) and finding ArgMax.
    return comp(lhs_idx, rhs_idx) == arg_min ? lhs_idx : rhs_idx;
  }
};

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
