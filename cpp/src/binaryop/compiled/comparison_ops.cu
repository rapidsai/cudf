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

#include "binops_custom.cuh"

namespace cudf::binops::compiled {
void dispatch_comparison_op(mutable_column_device_view& outd,
                            column_device_view const& lhsd,
                            column_device_view const& rhsd,
                            binary_operator op,
                            rmm::cuda_stream_view stream)
{
  auto common_dtype = get_common_type(outd.type(), lhsd.type(), rhsd.type());

  // Execute it on every element
  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(outd.size()),
    [op, outd, lhsd, rhsd, common_dtype] __device__(size_type i) {
      // clang-format off
      switch (op) {
      // Similar enabled template types should go together (performance)
      case binary_operator::LESS:          device_type_dispatcher<ops::Less, true>{outd, lhsd, rhsd, common_dtype}(i); break;
      case binary_operator::GREATER:       device_type_dispatcher<ops::Greater, true>{outd, lhsd, rhsd, common_dtype}(i); break;
      case binary_operator::LESS_EQUAL:    device_type_dispatcher<ops::LessEqual, true>{outd, lhsd, rhsd, common_dtype}(i); break;
      case binary_operator::GREATER_EQUAL: device_type_dispatcher<ops::GreaterEqual, true>{outd, lhsd, rhsd, common_dtype}(i); break;
      default:;
      }
      // clang-format on
    });
  //"cudf::binops::jit::kernel_v_v")  //TODO v_s, s_v.
}
}  // namespace cudf::binops::compiled
