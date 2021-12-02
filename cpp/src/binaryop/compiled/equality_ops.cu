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

namespace cudf::binops::compiled {
void dispatch_equality_op(mutable_column_device_view& outd,
                          column_device_view const& lhsd,
                          column_device_view const& rhsd,
                          bool is_lhs_scalar,
                          bool is_rhs_scalar,
                          binary_operator op,
                          rmm::cuda_stream_view stream)
{
  auto common_dtype = get_common_type(outd.type(), lhsd.type(), rhsd.type());

  // Execute it on every element

  if (common_dtype) {
    if (op == binary_operator::EQUAL) {
      for_each(stream,
               outd.size(),
               binary_op_device_dispatcher<ops::Equal>{
                 *common_dtype, outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar});
    } else if (op == binary_operator::NOT_EQUAL) {
      for_each(stream,
               outd.size(),
               binary_op_device_dispatcher<ops::NotEqual>{
                 *common_dtype, outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar});
    }
  } else {
    if (op == binary_operator::EQUAL) {
      for_each(stream,
               outd.size(),
               binary_op_double_device_dispatcher<ops::Equal>{
                 outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar});
    } else if (op == binary_operator::NOT_EQUAL) {
      for_each(stream,
               outd.size(),
               binary_op_double_device_dispatcher<ops::NotEqual>{
                 outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar});
    }
  }
}

}  // namespace cudf::binops::compiled
