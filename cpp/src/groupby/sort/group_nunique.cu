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

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/types.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {
struct nunique_functor {
  template <typename T>
  typename std::enable_if_t<cudf::is_equality_comparable<T, T>(), std::unique_ptr<column>>
  operator()(column_view const& values,
             rmm::device_vector<size_type> const& group_labels,
             size_type const num_groups,
             rmm::device_vector<size_type> const& group_offsets,
             null_policy null_handling,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream)
  {
    auto result = make_numeric_column(
      data_type(type_to_id<size_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);

    if (num_groups == 0) { return result; }

    auto values_view = column_device_view::create(values);
    if (values.has_nulls()) {
      auto equal              = element_equality_comparator<true>{*values_view, *values_view};
      auto is_unique_iterator = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_type>(0),
        [v = *values_view,
         equal,
         null_handling,
         group_offsets = group_offsets.data().get(),
         group_labels  = group_labels.data().get()] __device__(auto i) -> size_type {
          bool is_input_countable =
            (null_handling == null_policy::INCLUDE || v.is_valid_nocheck(i));
          bool is_unique = is_input_countable &&
                           (group_offsets[group_labels[i]] == i ||  // first element or
                            (not equal.operator()<T>(i, i - 1)));   // new unique value in sorted
          return static_cast<size_type>(is_unique);
        });

      thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                            group_labels.begin(),
                            group_labels.end(),
                            is_unique_iterator,
                            thrust::make_discard_iterator(),
                            result->mutable_view().begin<size_type>());
    } else {
      auto equal              = element_equality_comparator<false>{*values_view, *values_view};
      auto is_unique_iterator = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_type>(0),
        [v = *values_view,
         equal,
         group_offsets = group_offsets.data().get(),
         group_labels  = group_labels.data().get()] __device__(auto i) -> size_type {
          bool is_unique = group_offsets[group_labels[i]] == i ||  // first element or
                           (not equal.operator()<T>(i, i - 1));    // new unique value in sorted
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

  template <typename T>
  typename std::enable_if_t<!cudf::is_equality_comparable<T, T>(), std::unique_ptr<column>>
  operator()(column_view const& values,
             rmm::device_vector<size_type> const& group_labels,
             size_type const num_groups,
             rmm::device_vector<size_type> const& group_offsets,
             null_policy null_handling,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream)
  {
    CUDF_FAIL("list_view group_nunique not supported yet");
  }
};
}  // namespace
std::unique_ptr<column> group_nunique(column_view const& values,
                                      rmm::device_vector<size_type> const& group_labels,
                                      size_type const num_groups,
                                      rmm::device_vector<size_type> const& group_offsets,
                                      null_policy null_handling,
                                      rmm::mr::device_memory_resource* mr,
                                      cudaStream_t stream)
{
  CUDF_EXPECTS(num_groups >= 0, "number of groups cannot be negative");
  CUDF_EXPECTS(static_cast<size_t>(values.size()) == group_labels.size(),
               "Size of values column should be same as that of group labels");

  return type_dispatcher(values.type(),
                         nunique_functor{},
                         values,
                         group_labels,
                         num_groups,
                         group_offsets,
                         null_handling,
                         mr,
                         stream);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
