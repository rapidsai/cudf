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

#include "binary_ops.hpp"
#include "operation.cuh"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

namespace cudf::binops::compiled::detail {
template <class T, class... Ts>
inline constexpr bool is_any_v = std::disjunction<std::is_same<T, Ts>...>::value;

template <typename OptionalIterator, typename DeviceComparator>
struct device_comparison_functor {
  // Explicit constructor definition required to avoid a "no instance of constructor" compilation
  // error
  device_comparison_functor(OptionalIterator const optional_iter,
                            bool const is_lhs_scalar,
                            bool const is_rhs_scalar,
                            DeviceComparator const& comparator)
    : _optional_iter(optional_iter),
      _is_lhs_scalar(is_lhs_scalar),
      _is_rhs_scalar(is_rhs_scalar),
      _comparator(comparator)
  {
  }

  bool __device__ operator()(size_type i)
  {
    return _optional_iter[i].has_value() &&
           _comparator(cudf::experimental::row::lhs_index_type{_is_lhs_scalar ? 0 : i},
                       cudf::experimental::row::rhs_index_type{_is_rhs_scalar ? 0 : i});
  }

  OptionalIterator const _optional_iter;
  bool const _is_lhs_scalar;
  bool const _is_rhs_scalar;
  DeviceComparator const _comparator;
};

template <class BinaryOperator,
          typename PhysicalElementComparator =
            cudf::experimental::row::lexicographic::sorting_physical_element_comparator>
void apply_struct_binary_op(mutable_column_view& out,
                            column_view const& lhs,
                            column_view const& rhs,
                            bool is_lhs_scalar,
                            bool is_rhs_scalar,
                            PhysicalElementComparator comparator = {},
                            rmm::cuda_stream_view stream         = cudf::default_stream_value)
{
  auto const compare_orders = std::vector<order>(
    lhs.size(),
    is_any_v<BinaryOperator, ops::Greater, ops::GreaterEqual> ? order::DESCENDING
                                                              : order::ASCENDING);
  auto const tlhs             = table_view{{lhs}};
  auto const trhs             = table_view{{rhs}};
  auto const table_comparator = cudf::experimental::row::lexicographic::two_table_comparator{
    tlhs, trhs, compare_orders, {}, stream};
  auto outd = column_device_view::create(out, stream);
  auto optional_iter =
    cudf::detail::make_optional_iterator<bool>(*outd, nullate::DYNAMIC{out.has_nulls()});
  auto const comparator_nulls = nullate::DYNAMIC{has_nested_nulls(tlhs) || has_nested_nulls(trhs)};

  auto tabulate_device_operator = [&](auto device_comparator) {
    thrust::tabulate(
      rmm::exec_policy(stream),
      out.begin<bool>(),
      out.end<bool>(),
      device_comparison_functor{optional_iter, is_lhs_scalar, is_rhs_scalar, device_comparator});
  };
  is_any_v<BinaryOperator, ops::LessEqual, ops::GreaterEqual>
    ? tabulate_device_operator(table_comparator.less_equivalent(comparator_nulls, comparator))
    : tabulate_device_operator(table_comparator.less(comparator_nulls, comparator));
}

template <typename PhysicalEqualityComparator =
            cudf::experimental::row::equality::physical_equality_comparator>
void apply_struct_equality_op(mutable_column_view& out,
                              column_view const& lhs,
                              column_view const& rhs,
                              bool is_lhs_scalar,
                              bool is_rhs_scalar,
                              binary_operator op,
                              PhysicalEqualityComparator comparator = {},
                              rmm::cuda_stream_view stream          = cudf::default_stream_value)
{
  CUDF_EXPECTS(op == binary_operator::EQUAL || op == binary_operator::NOT_EQUAL,
               "Unsupported operator for these types");

  auto tlhs = table_view{{lhs}};
  auto trhs = table_view{{rhs}};
  auto table_comparator =
    cudf::experimental::row::equality::two_table_comparator{tlhs, trhs, stream};
  auto device_comparator =
    table_comparator.equal_to(nullate::DYNAMIC{has_nested_nulls(tlhs) || has_nested_nulls(trhs)},
                              null_equality::EQUAL,
                              comparator);

  auto outd = column_device_view::create(out, stream);
  auto optional_iter =
    cudf::detail::make_optional_iterator<bool>(*outd, nullate::DYNAMIC{out.has_nulls()});
  thrust::tabulate(rmm::exec_policy(stream),
                   out.begin<bool>(),
                   out.end<bool>(),
                   [optional_iter,
                    is_lhs_scalar,
                    is_rhs_scalar,
                    preserve_output = (op != binary_operator::NOT_EQUAL),
                    device_comparator] __device__(size_type i) {
                     auto lhs = cudf::experimental::row::lhs_index_type{is_lhs_scalar ? 0 : i};
                     auto rhs = cudf::experimental::row::rhs_index_type{is_rhs_scalar ? 0 : i};
                     return optional_iter[i].has_value() and
                            (device_comparator(lhs, rhs) == preserve_output);
                   });
}
}  // namespace cudf::binops::compiled::detail
