/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/detail/reduction_operators.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/table/table_view.hpp>

#include <cudf/table/experimental/row_operators.cuh>

namespace cudf {
namespace reduction {
namespace detail {

using row_comparator = cudf::experimental::row::lexicographic::self_comparator;
using device_row_comparator =
  cudf::experimental::row::lexicographic::less_comparator<nullate::DYNAMIC>;
using column_device_view_ptr =
  std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>;

/**
 * @brief Binary operator ArgMin/ArgMax with index values into the input column.
 */
struct row_arg_minmax_fn {
  column_device_view const input;
  device_row_comparator const comp;
  bool const is_arg_min;

  row_arg_minmax_fn(column_device_view const& input_,
                    device_row_comparator&& comp_,
                    bool const is_arg_min_)
    : input(input_), comp(std::move(comp_)), is_arg_min(is_arg_min_)
  {
  }

  // This function is explicitly prevented from inlining, because it calls to
  // `row_lexicographic_comparator::operator()` which is inlined and very heavy-weight. As a result,
  // instantiating this functor will result in huge code, and objects of this functor used with
  // `thrust::reduce_by_key` or `thrust::scan_by_key` will result in significant compile time.
  __attribute__((noinline)) __device__ auto operator()(size_type lhs_idx, size_type rhs_idx) const
  {
    // The extra bounds checking is due to issue github.com/rapidsai/cudf/issues/9156 and
    // github.com/NVIDIA/thrust/issues/1525
    // where invalid random values may be passed here by thrust::reduce_by_key
    auto const num_rows = input.size();
    if (lhs_idx < 0 || lhs_idx >= num_rows) { return rhs_idx; }
    if (rhs_idx < 0 || rhs_idx >= num_rows) { return lhs_idx; }

    // Nulls at top level are excluded from the operation.
    // Thus, if the is one null, return index of the non-null element.
    // If both sides are nulls, just return any index.
    if (input.is_null(lhs_idx)) { return rhs_idx; }
    if (input.is_null(rhs_idx)) { return lhs_idx; }

    // Return `lhs_idx` iff:
    //   row(lhs_idx) <  row(rhs_idx) and finding ArgMin, or
    //   row(lhs_idx) >= row(rhs_idx) and finding ArgMax.
    return comp(lhs_idx, rhs_idx) == is_arg_min ? lhs_idx : rhs_idx;
  }
};

/**
 * @brief The null order when comparing a null with non-null elements. Currently support only the
 * default null order: nulls are compared as LESS than any other non-null elements.
 */
auto static constexpr DEFAULT_NULL_ORDER = cudf::null_order::BEFORE;

/**
 * @brief The utility class to provide a binary operator object for lexicographic comparison of
 * struct elements.
 */
class comparison_binop_generator {
 private:
  column_device_view_ptr const input_cdv_ptr;
  row_comparator const comparator;
  rmm::cuda_stream_view const stream;
  bool const has_nulls;
  bool const is_min_op;

  comparison_binop_generator(column_view const& input_,
                             rmm::cuda_stream_view stream_,
                             bool is_min_op_)
    : input_cdv_ptr(column_device_view::create(input_)),
      comparator(table_view{{input_}}, {}, std::vector<null_order>{DEFAULT_NULL_ORDER}, stream_),
      stream(stream_),
      has_nulls(has_nested_nulls(table_view{{input_}})),
      is_min_op(is_min_op_)
  {
  }

 public:
  auto binop() const
  {
    return row_arg_minmax_fn(
      *input_cdv_ptr, comparator.device_comparator(nullate::DYNAMIC{has_nulls}), is_min_op);
  }

  template <typename BinOp>
  static auto create(column_view const& input, rmm::cuda_stream_view stream)
  {
    return comparison_binop_generator(
      input,
      stream,
      std::is_same_v<BinOp, cudf::reduction::op::min> || std::is_same_v<BinOp, cudf::DeviceMin>);
  }

  template <cudf::aggregation::Kind K>
  static auto create(column_view const& input, rmm::cuda_stream_view stream)

  {
    return comparison_binop_generator(
      input, stream, K == cudf::aggregation::MIN || K == cudf::aggregation::ARGMIN);
  }
};

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
