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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>

#include <thrust/gather.h>

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {

std::unique_ptr<column>
group_nth_element(column_view const &values,
                  column_view const &group_sizes,
                  rmm::device_vector<size_type> const &group_labels,
                  rmm::device_vector<size_type> const &group_offsets,
                  size_type num_groups, size_type n,
                  include_nulls _include_nulls,
                  rmm::mr::device_memory_resource *mr, cudaStream_t stream) {

  CUDF_EXPECTS(static_cast<size_t>(values.size()) == group_labels.size(),
               "Size of values column should be same as that of group labels");

  if (num_groups == 0) {
    return experimental::empty_like(values);
  }

  auto output = make_numeric_column(
      data_type{experimental::type_to_id<size_type>()}, num_groups,
      mask_state::UNALLOCATED, stream);
  mutable_column_view output_view = output->mutable_view();
  auto exec = rmm::exec_policy(stream)->on(stream);

  // include nulls (equivalent to pandas nth(dropna=None) but return nulls for n
  if (_include_nulls == include_nulls::YES || !values.has_nulls()) {
    // Returns index of nth value.
    thrust::transform(exec, group_sizes.begin<size_type>(),
                      group_sizes.end<size_type>(),
                      group_offsets.begin(), output_view.begin<size_type>(),
                      [n, out_of_bounds = values.size()] __device__(
                          auto group_size, auto group_offset) {
                        bool nth_within_group =
                            (n < 0) ? group_size >= (-n) : group_size > n;
                        if (nth_within_group)
                          return group_offset + ((n < 0) ? group_size + n : n);
                        else
                          return out_of_bounds;
                      });
  } else { // skip nulls (equivalent to pandas nth(dropna='any'))
    // Returns index of nth value.
    thrust::fill(exec, output->mutable_view().begin<size_type>(),
                 output->mutable_view().end<size_type>(),
                 values.size()); // for out of bounds
    auto values_view = column_device_view::create(values);
    auto bitmask_iterator = thrust::make_transform_iterator(
        experimental::detail::make_validity_iterator(*values_view),
        [] __device__(auto b) { return static_cast<size_type>(b); });
    rmm::device_vector<size_type> intra_group_index(values.size());
    // intra group index for valids only.
    thrust::exclusive_scan_by_key(exec, group_labels.begin(),
                                  group_labels.end(), bitmask_iterator,
                                  intra_group_index.begin());
    // gather the valid index == n
    thrust::scatter_if(
        exec, thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(0) + values.size(),
        group_labels.begin(),                         // map
        thrust::make_counting_iterator<size_type>(0), // stencil
        output->mutable_view().begin<size_type>(),
        [n, bitmask_iterator,
         intra_group_index =
             intra_group_index.begin()] __device__(auto i) -> bool {
          return (bitmask_iterator[i] && intra_group_index[i] == n);
        });
  }
  bool nullify_out_of_bounds = thrust::transform_reduce(
      exec, group_sizes.begin<size_type>(),
      group_sizes.end<size_type>(),
      [n] __device__(const size_type group_size) {
        bool nth_within_group = (n < 0) ? group_size >= (-n) : group_size > n;
        return !nth_within_group;
      },
      false, thrust::logical_or<bool>{});
  auto output_table =
      experimental::detail::gather(table_view{{values}}, output->view(), false,
                                   nullify_out_of_bounds, false, mr, stream);
  return std::make_unique<column>(std::move(output_table->get_column(0)));
}
} // namespace detail
} // namespace groupby
} // namespace experimental
} // namespace cudf
