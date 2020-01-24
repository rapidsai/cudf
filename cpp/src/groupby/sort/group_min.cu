/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <groupby/sort/group_single_pass_reduction_util.cuh>

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {

std::unique_ptr<column> group_min(
    column_view const& values,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_labels,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
  return type_dispatcher(values.type(), reduce_functor<aggregation::MIN>{},
                         values, group_sizes, group_labels, mr, stream);
}

}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
