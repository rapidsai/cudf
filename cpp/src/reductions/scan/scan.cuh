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

#include <cudf/column/column.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/reduction.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

// logical-and scan of the null mask of the input view
rmm::device_buffer mask_scan(column_view const& input_view,
                             scan_type inclusive,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr);

template <template <typename> typename DispatchFn>
std::unique_ptr<column> scan_agg_dispatch(const column_view& input,
                                          scan_aggregation const& agg,
                                          null_policy null_handling,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  switch (agg.kind) {
    case aggregation::SUM:
      return type_dispatcher<dispatch_storage_type>(
        input.type(), DispatchFn<DeviceSum>(), input, null_handling, stream, mr);
    case aggregation::MIN:
      return type_dispatcher<dispatch_storage_type>(
        input.type(), DispatchFn<DeviceMin>(), input, null_handling, stream, mr);
    case aggregation::MAX:
      return type_dispatcher<dispatch_storage_type>(
        input.type(), DispatchFn<DeviceMax>(), input, null_handling, stream, mr);
    case aggregation::PRODUCT:
      // a product scan on a decimal type with non-zero scale would result in each element having
      // a different scale, and because scale is stored once per column, this is not possible
      if (is_fixed_point(input.type())) CUDF_FAIL("decimal32/64/128 cannot support product scan");
      return type_dispatcher<dispatch_storage_type>(
        input.type(), DispatchFn<DeviceProduct>(), input, null_handling, stream, mr);
    default: CUDF_FAIL("Unsupported aggregation operator for scan");
  }
}

}  // namespace detail
}  // namespace cudf
