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

#include "scan.cuh"
#include "scan.hpp"

#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/reduction.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

std::unique_ptr<column> scan_max(const column_view& input,
                                 scan_type inclusive,
                                 null_policy null_handling,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  return cudf::type_dispatcher<dispatch_storage_type>(
    input.type(), scan_dispatcher<cudf::DeviceMax>(), input, inclusive, null_handling, stream, mr);
}
}  // namespace detail
}  // namespace cudf
