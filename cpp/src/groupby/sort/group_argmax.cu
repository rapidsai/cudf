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

#include <cudf/detail/gather.hpp>
#include <groupby/sort/group_single_pass_reduction_util.cuh>

#include <thrust/transform.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_argmax(column_view const& values,
                                     size_type num_groups,
                                     rmm::device_vector<size_type> const& group_labels,
                                     column_view const& key_sort_order,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
{
  auto indices = type_dispatcher(values.type(),
                                 reduce_functor<aggregation::ARGMAX>{},
                                 values,
                                 num_groups,
                                 group_labels,
                                 rmm::mr::get_default_resource(),
                                 stream);

  // The functor returns the index of maximum in the sorted values.
  // We need the index of maximum in the original unsorted values.
  // So use indices to gather the sort order used to sort `values`.
  // Gather map cannot be null so we make a view with the mask removed.
  // The values in data buffer of indices corresponding to null values was
  // initialized to ARGMAX_SENTINEL which is an out of bounds index value (-1)
  // and causes the gathered value to be null.
  column_view null_removed_indices(
    data_type(type_to_id<size_type>()),
    indices->size(),
    static_cast<void const*>(indices->view().template data<size_type>()));
  auto result_table =
    cudf::detail::gather(table_view({key_sort_order}),
                         null_removed_indices,
                         indices->nullable() ? cudf::detail::out_of_bounds_policy::IGNORE
                                             : cudf::detail::out_of_bounds_policy::NULLIFY,
                         cudf::detail::negative_index_policy::NOT_ALLOWED,
                         mr,
                         stream);

  return std::move(result_table->release()[0]);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
