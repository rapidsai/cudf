/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
               "Unsupported operator for these types",
               cudf::data_type_error);
  auto common_dtype = get_common_type(out.type(), lhs.type(), rhs.type());
  auto outd         = mutable_column_device_view::create(out, stream);
  auto lhsd         = column_device_view::create(lhs, stream);
  auto rhsd         = column_device_view::create(rhs, stream);
  if (common_dtype) {
    if (op == binary_operator::EQUAL) {
      thrust::for_each_n(rmm::exec_policy_nosync(stream),
                         thrust::counting_iterator<size_type>(0),
                         out.size(),
                         binary_op_device_dispatcher<ops::Equal>{
                           *common_dtype, *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
    } else if (op == binary_operator::NOT_EQUAL) {
      thrust::for_each_n(rmm::exec_policy_nosync(stream),
                         thrust::counting_iterator<size_type>(0),
                         out.size(),
                         binary_op_device_dispatcher<ops::NotEqual>{
                           *common_dtype, *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
    }
  } else {
    if (op == binary_operator::EQUAL) {
      thrust::for_each_n(rmm::exec_policy_nosync(stream),
                         thrust::counting_iterator<size_type>(0),
                         out.size(),
                         binary_op_double_device_dispatcher<ops::Equal>{
                           *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
    } else if (op == binary_operator::NOT_EQUAL) {
      thrust::for_each_n(rmm::exec_policy_nosync(stream),
                         thrust::counting_iterator<size_type>(0),
                         out.size(),
                         binary_op_double_device_dispatcher<ops::NotEqual>{
                           *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
    }
  }
}

}  // namespace cudf::binops::compiled
