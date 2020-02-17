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
#include <cudf/table/row_operators.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>
#include <cudf/aggregation.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {


std::unique_ptr<column> group_nunique(
    column_view const& values,
    rmm::device_vector<size_type> const& group_labels,
    size_type num_groups,
    rmm::device_vector<size_type> const& group_offsets,
    include_nulls _include_nulls,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
  CUDF_EXPECTS(num_groups >= 0, "number of groups cannot be negative");
  CUDF_EXPECTS(static_cast<size_t>(values.size()) == group_labels.size(),
               "Size of values column should be same as that of group labels");

  auto result = make_numeric_column(data_type(type_to_id<size_type>()),
                  num_groups, mask_state::UNALLOCATED, stream, mr);

  if (num_groups == 0) {
    return result;
  }

  auto values_view = column_device_view::create(values);
  if (values.has_nulls()) {

    auto comp = element_equality_comparator<true>{*values_view, *values_view};
    auto is_unique_iterator = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_type>(0),
        [v = *values_view, comp, _include_nulls,
         group_offsets = group_offsets.data().get(),
         group_labels =  group_labels.data().get()] 
         __device__(auto i) -> size_type {
          bool is_unique =
              (_include_nulls == include_nulls::YES || v.is_valid_nocheck(i)) &&
              (group_offsets[group_labels[i]] == i ||
              (not cudf::experimental::type_dispatcher(v.type(), comp, i, i - 1)));
          return static_cast<size_type>(is_unique);
        });

    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          group_labels.begin(),
                          group_labels.end(),
                          is_unique_iterator,
                          thrust::make_discard_iterator(),
                          result->mutable_view().begin<size_type>());
  } else {
    auto comp = element_equality_comparator<false>{*values_view, *values_view};
    auto is_unique_iterator = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_type>(0),
        [v = *values_view, comp, 
         group_offsets = group_offsets.data().get(),
         group_labels = group_labels.data().get()] 
         __device__(auto i) -> size_type {
          bool is_unique = group_offsets[group_labels[i]] == i ||
              (not cudf::experimental::type_dispatcher(v.type(), comp, i, i - 1));
          return static_cast<size_type>(is_unique);
        });
    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          group_labels.begin(),
                          group_labels.end(),
                          is_unique_iterator,
                          thrust::make_discard_iterator(),
                          result->mutable_view().begin<size_type>());
  }

  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
