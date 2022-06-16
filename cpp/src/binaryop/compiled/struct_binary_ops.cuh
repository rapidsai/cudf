/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include "binary_ops.hpp"
#include "operation.cuh"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf::binops::compiled {
namespace detail {
template <class T, class... Ts>
inline constexpr bool is_any_v = std::disjunction<std::is_same<T, Ts>...>::value;
}

template <class BinaryOperator, typename PhysicalElementComparator>
void apply_struct_binary_op(mutable_column_view& out,
                            column_view const& lhs,
                            column_view const& rhs,
                            bool is_lhs_scalar,
                            bool is_rhs_scalar,
                            rmm::cuda_stream_view stream,
                            PhysicalElementComparator c)
{
  auto compare_orders = std::vector<order>(
    lhs.size(),
    detail::is_any_v<BinaryOperator, ops::Greater, ops::GreaterEqual> ? order::DESCENDING
                                                                      : order::ASCENDING);
  auto tlhs             = table_view{{lhs}};
  auto trhs             = table_view{{rhs}};
  auto table_comparator = cudf::experimental::row::lexicographic::two_table_comparator{
    tlhs, trhs, compare_orders, {}, stream};
  auto outd = column_device_view::create(out, stream);
  auto optional_iter =
    cudf::detail::make_optional_iterator<bool>(*outd, nullate::DYNAMIC{out.has_nulls()});

  if (detail::is_any_v<BinaryOperator, ops::LessEqual, ops::GreaterEqual>) {
    auto device_comparator = table_comparator.less_equivalent(
      nullate::DYNAMIC{nullate::DYNAMIC{has_nested_nulls(tlhs) || has_nested_nulls(trhs)}}, c);
    thrust::tabulate(
      rmm::exec_policy(stream),
      out.begin<bool>(),
      out.end<bool>(),
      [optional_iter, is_lhs_scalar, is_rhs_scalar, device_comparator] __device__(size_type i) {
        return optional_iter[i].has_value() &&
               device_comparator(cudf::experimental::row::lhs_index_type{is_lhs_scalar ? 0 : i},
                                 cudf::experimental::row::rhs_index_type{is_rhs_scalar ? 0 : i});
      });

  } else {
    auto device_comparator = table_comparator.less(
      nullate::DYNAMIC{nullate::DYNAMIC{has_nested_nulls(tlhs) || has_nested_nulls(trhs)}}, c);
    thrust::tabulate(
      rmm::exec_policy(stream),
      out.begin<bool>(),
      out.end<bool>(),
      [optional_iter, is_lhs_scalar, is_rhs_scalar, device_comparator] __device__(size_type i) {
        return optional_iter[i].has_value() &&
               device_comparator(cudf::experimental::row::lhs_index_type{is_lhs_scalar ? 0 : i},
                                 cudf::experimental::row::rhs_index_type{is_rhs_scalar ? 0 : i});
      });
  }
  return;
}

template <typename PhysicalEqualityComparator>
void apply_struct_equality_op(mutable_column_view& out,
                              column_view const& lhs,
                              column_view const& rhs,
                              bool is_lhs_scalar,
                              bool is_rhs_scalar,
                              binary_operator op,
                              rmm::cuda_stream_view stream,
                              PhysicalEqualityComparator c)
{
  CUDF_EXPECTS(op == binary_operator::EQUAL || op == binary_operator::NOT_EQUAL,
               "Unsupported operator for these types");

  auto tlhs = table_view{{lhs}};
  auto trhs = table_view{{rhs}};
  auto table_comparator =
    cudf::experimental::row::equality::two_table_comparator{tlhs, trhs, stream};
  auto device_comparator = table_comparator.equal_to(
    nullate::DYNAMIC{nullate::DYNAMIC{has_nested_nulls(tlhs) || has_nested_nulls(trhs)}},
    null_equality::EQUAL,
    c);

  auto outd = column_device_view::create(out, stream);
  auto optional_iter =
    cudf::detail::make_optional_iterator<bool>(*outd, nullate::DYNAMIC{out.has_nulls()});
  thrust::tabulate(
    rmm::exec_policy(stream),
    out.begin<bool>(),
    out.end<bool>(),
    [optional_iter,
     is_lhs_scalar,
     is_rhs_scalar,
     flip_output = (op == binary_operator::NOT_EQUAL),
     device_comparator] __device__(size_type i) {
      auto lhs = cudf::experimental::row::lhs_index_type{is_lhs_scalar ? 0 : i};
      auto rhs = cudf::experimental::row::rhs_index_type{is_rhs_scalar ? 0 : i};
      return optional_iter[i].has_value() and
             (flip_output ? not device_comparator(lhs, rhs) : device_comparator(lhs, rhs));
    });
}
}  // namespace cudf::binops::compiled