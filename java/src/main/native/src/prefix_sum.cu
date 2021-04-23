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

#include <thrust/scan.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/exec_policy.hpp>


namespace cudf {
namespace jni {

std::unique_ptr<column> prefix_sum(column_view const &value_column,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource *mr) {
  // Defensive checks.
  CUDF_EXPECTS(value_column.type().id() == type_id::INT64, "Only longs are supported.");
  CUDF_EXPECTS(!value_column.has_nulls(), "NULLS are not supported");

  auto result = make_numeric_column(value_column.type(), value_column.size(),
                                    mask_state::ALL_VALID, stream, mr);

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         value_column.begin<int64_t>(),
                         value_column.end<int64_t>(),
                         result->mutable_view().begin<int64_t>());

  return result;
}
} // namespace jni
} // namespace cudf
