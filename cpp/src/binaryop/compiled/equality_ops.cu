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

#include "binary_ops.cuh"

#include <cudf/detail/structs/utilities.hpp>
#include <cudf/table/row_operators.cuh>

namespace cudf::binops::compiled {
void dispatch_equality_op(mutable_column_view& out,
                          column_view const& lhs,
                          column_view const& rhs,
                          bool is_lhs_scalar,
                          bool is_rhs_scalar,
                          binary_operator op,
                          rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(op == binary_operator::EQUAL || op == binary_operator::NOT_EQUAL,
               "Unsupported operator for these types");
  if (is_struct(lhs.type()) && is_struct(rhs.type())) {
    auto const nullability =
      structs::detail::contains_null_structs(lhs) || structs::detail::contains_null_structs(rhs)
        ? structs::detail::column_nullability::FORCE
        : structs::detail::column_nullability::MATCH_INCOMING;
    auto const lhs_flattened =
      structs::detail::flatten_nested_columns(table_view{{lhs}}, {}, {}, nullability);
    auto const rhs_flattened =
      structs::detail::flatten_nested_columns(table_view{{rhs}}, {}, {}, nullability);
    auto lhsd       = table_device_view::create(lhs_flattened);
    auto rhsd       = table_device_view::create(rhs_flattened);
    auto comparator = row_equality_comparator{
      nullate::DYNAMIC{has_nested_nulls(lhs_flattened) || has_nested_nulls(rhs_flattened)},
      *lhsd,
      *rhsd,
      null_equality::EQUAL,
      nan_equality::UNEQUAL};

    auto outd = column_device_view::create(out, stream);
    auto optional_iter =
      cudf::detail::make_optional_iterator<bool>(*outd, nullate::DYNAMIC{out.has_nulls()});
    thrust::tabulate(rmm::exec_policy(stream),
                     out.begin<bool>(),
                     out.end<bool>(),
                     [optional_iter,
                      is_lhs_scalar,
                      is_rhs_scalar,
                      flip_output = (op == binary_operator::NOT_EQUAL),
                      comparator] __device__(size_type i) {
                       auto lhs = is_lhs_scalar ? 0 : i;
                       auto rhs = is_rhs_scalar ? 0 : i;
                       return optional_iter[i].has_value() and
                              (flip_output ? not comparator(lhs, rhs) : comparator(lhs, rhs));
                     });
  } else {
    auto common_dtype = get_common_type(out.type(), lhs.type(), rhs.type());
    auto outd         = mutable_column_device_view::create(out, stream);
    auto lhsd         = column_device_view::create(lhs, stream);
    auto rhsd         = column_device_view::create(rhs, stream);
    if (common_dtype) {
      if (op == binary_operator::EQUAL) {
        for_each(stream,
                 out.size(),
                 binary_op_device_dispatcher<ops::Equal>{
                   *common_dtype, *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
      } else if (op == binary_operator::NOT_EQUAL) {
        for_each(stream,
                 out.size(),
                 binary_op_device_dispatcher<ops::NotEqual>{
                   *common_dtype, *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
      }
    } else {
      if (op == binary_operator::EQUAL) {
        for_each(stream,
                 out.size(),
                 binary_op_double_device_dispatcher<ops::Equal>{
                   *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
      } else if (op == binary_operator::NOT_EQUAL) {
        for_each(stream,
                 out.size(),
                 binary_op_double_device_dispatcher<ops::NotEqual>{
                   *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
      }
    }
  }
}

}  // namespace cudf::binops::compiled
