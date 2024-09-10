/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "groupby/sort/group_single_pass_reduction_util.cuh"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_max(column_view const& values,
                                  size_type num_groups,
                                  cudf::device_span<size_type const> group_labels,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  auto values_type = cudf::is_dictionary(values.type())
                       ? dictionary_column_view(values).keys().type()
                       : values.type();
  return type_dispatcher(values_type,
                         group_reduction_dispatcher<aggregation::MAX>{},
                         values,
                         num_groups,
                         group_labels,
                         stream,
                         mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
