/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

template <typename Nullate>
struct is_unique_iterator_fn {
  using comparator_type =
    typename cudf::experimental::row::equality::device_row_comparator<Nullate>;

  Nullate nulls;
  column_device_view const v;
  comparator_type equal;
  null_policy null_handling;
  size_type const* group_offsets;
  size_type const* group_labels;

  is_unique_iterator_fn(Nullate nulls,
                        column_device_view const& v,
                        comparator_type const& equal,
                        null_policy null_handling,
                        size_type const* group_offsets,
                        size_type const* group_labels)
    : nulls{nulls},
      v{v},
      equal{equal},
      null_handling{null_handling},
      group_offsets{group_offsets},
      group_labels{group_labels}
  {
  }

  __device__ size_type operator()(size_type i) const
  {
    auto const is_input_countable =
      !nulls || (null_handling == null_policy::INCLUDE || v.is_valid_nocheck(i));
    auto const is_unique =
      is_input_countable && (group_offsets[group_labels[i]] == i ||  // first element or
                             (not equal(i, i - 1)));                 // new unique value in sorted
    return static_cast<size_type>(is_unique);
  }
};
}  // namespace

std::unique_ptr<column> group_nunique(column_view const& values,
                                      cudf::device_span<size_type const> group_labels,
                                      size_type const num_groups,
                                      cudf::device_span<size_type const> group_offsets,
                                      null_policy null_handling,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(num_groups >= 0, "number of groups cannot be negative");
  CUDF_EXPECTS(static_cast<size_t>(values.size()) == group_labels.size(),
               "Size of values column should be same as that of group labels");

  auto result = make_numeric_column(
    data_type(type_to_id<size_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);

  if (num_groups == 0) { return result; }

  auto const values_view = table_view{{values}};
  auto const comparator  = cudf::experimental::row::equality::self_comparator{values_view, stream};
  auto const d_equal     = comparator.equal_to(
    cudf::nullate::DYNAMIC{cudf::has_nested_nulls(values_view)}, null_equality::EQUAL);

  auto const d_values_view = column_device_view::create(values, stream);
  auto const is_unique_iterator =
    thrust::make_transform_iterator(thrust::counting_iterator<cudf::size_type>(0),
                                    is_unique_iterator_fn{nullate::DYNAMIC{values.has_nulls()},
                                                          *d_values_view,
                                                          d_equal,
                                                          null_handling,
                                                          group_offsets.data(),
                                                          group_labels.data()});
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        group_labels.begin(),
                        group_labels.end(),
                        is_unique_iterator,
                        thrust::make_discard_iterator(),
                        result->mutable_view().begin<size_type>());

  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
