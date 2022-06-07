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

#include "binary_ops.cuh"

#include <cudf/detail/structs/utilities.hpp>
#include <cudf/table/row_operators.cuh>

namespace cudf::binops::compiled {
namespace {
void struct_equality_op(mutable_column_view& out,
                        column_view const& lhs,
                        column_view const& rhs,
                        bool is_lhs_scalar,
                        bool is_rhs_scalar,
                        binary_operator op,
                        rmm::cuda_stream_view stream)
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
    cudf::experimental::row::equality::physical_equality_comparator{});

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
}  // namespace

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
    struct_equality_op(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, op, stream);
    return;
  }

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

}  // namespace cudf::binops::compiled
