/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/reduction/detail/reduction_operators.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <thrust/reduce.h>
#include <thrust/scan.h>

namespace cudf {
namespace reduction {
namespace detail {

/**
 * @brief Binary operator ArgMin/ArgMax with index values into the input table.
 */
struct row_arg_minmax_fn {
  size_type const num_rows;
  row_lexicographic_comparator<nullate::DYNAMIC> const comp;
  bool const is_arg_min;

  row_arg_minmax_fn(table_device_view const& table,
                    bool has_nulls,
                    null_order const* null_precedence,
                    bool const is_arg_min)
    : num_rows(table.num_rows()),
      comp(nullate::DYNAMIC{has_nulls}, table, table, nullptr, null_precedence),
      is_arg_min(is_arg_min)
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
    if (lhs_idx < 0 || lhs_idx >= num_rows) { return rhs_idx; }
    if (rhs_idx < 0 || rhs_idx >= num_rows) { return lhs_idx; }

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
 *
 * The input of this class is a structs column. Using the binary operator provided by this class,
 * nulls STRUCT are compared as larger than all other non-null STRUCT elements - if finding for
 * ARGMIN, or smaller than all other non-null STRUCT elements - if finding for ARGMAX. This helps
 * achieve the results of finding the min or max STRUCT element when nulls are excluded from the
 * operations, returning null only when all the input elements are nulls.
 */
class comparison_binop_generator {
 private:
  std::unique_ptr<cudf::structs::detail::flattened_table> const flattened_input;
  std::unique_ptr<table_device_view, std::function<void(table_device_view*)>> const
    d_flattened_input_ptr;
  bool const is_min_op;
  bool const has_nulls;

  std::vector<null_order> null_orders;
  rmm::device_uvector<null_order> null_orders_dvec;

  comparison_binop_generator(column_view const& input, rmm::cuda_stream_view stream, bool is_min_op)
    : flattened_input{cudf::structs::detail::flatten_nested_columns(
        table_view{{input}},
        {},
        std::vector<null_order>{DEFAULT_NULL_ORDER},
        cudf::structs::detail::column_nullability::MATCH_INCOMING,
        stream)},
      d_flattened_input_ptr{
        table_device_view::create(flattened_input->flattened_columns(), stream)},
      is_min_op(is_min_op),
      has_nulls{has_nested_nulls(table_view{{input}})},
      null_orders_dvec(0, stream)
  {
    if (is_min_op) {
      null_orders = flattened_input->null_orders();
      // If the input column has nulls (at the top level), null structs are excluded from the
      // operations, and that is equivalent to considering top-level nulls as larger than all other
      // non-null STRUCT elements (if finding for ARGMIN), or smaller than all other non-null STRUCT
      // elements (if finding for ARGMAX). Thus, we need to set a separate null order for the top
      // level structs column (which is stored at the first position in the null_orders array) to
      // achieve this purpose.
      if (input.has_nulls()) { null_orders.front() = cudf::null_order::AFTER; }
      null_orders_dvec = cudf::detail::make_device_uvector_async(
        null_orders, stream, rmm::mr::get_current_device_resource());
    }
    // else: Don't need to generate nulls order to copy to device memory if we have all null orders
    // are BEFORE (that happens when we have is_min_op == false).
  }

 public:
  auto binop() const
  {
    return row_arg_minmax_fn(*d_flattened_input_ptr, has_nulls, null_orders_dvec.data(), is_min_op);
  }

  template <typename BinOp>
  static auto create(column_view const& input, rmm::cuda_stream_view stream)
  {
    return comparison_binop_generator(input,
                                      stream,
                                      std::is_same_v<BinOp, cudf::reduction::detail::op::min> ||
                                        std::is_same_v<BinOp, cudf::DeviceMin>);
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
