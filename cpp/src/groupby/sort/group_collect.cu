/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_collect(column_view const &values,
                                      rmm::device_vector<size_type> const &group_offsets,
                                      size_type num_groups,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource *mr)
{
  rmm::device_buffer offsets_data(
    group_offsets.data().get(), group_offsets.size() * sizeof(cudf::size_type), stream, mr);

  auto offsets = std::make_unique<cudf::column>(
    cudf::data_type(cudf::type_to_id<cudf::size_type>()), num_groups + 1, std::move(offsets_data));

  return make_lists_column(num_groups,
                           std::move(offsets),
                           std::make_unique<cudf::column>(values, stream, mr),
                           0,
                           rmm::device_buffer{0, stream, mr},
                           stream,
                           mr);
}
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
