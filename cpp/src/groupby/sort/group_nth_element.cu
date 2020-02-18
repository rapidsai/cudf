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
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/types.hpp>

#include <thrust/gather.h>
#include "group_reductions.hpp"

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {

//TODO functor to implement both fixed width and string columns.
namespace {
template<bool include_nulls>
struct nth_element_functor {

  //include nulls (equivalent to pandas nth(dropna=None) but return nulls for n out of bounds)
  template <typename T>
  std::enable_if_t<is_fixed_width<T>() && include_nulls, std::unique_ptr<column>>
  operator()(column_view const &values,
             rmm::device_vector<size_type> const& group_labels,
             rmm::device_vector<size_type> const& group_offsets,
             size_type num_groups, size_type n,
             rmm::mr::device_memory_resource *mr, cudaStream_t stream)
  {
    using ResultType = cudf::experimental::detail::target_type_t<
        T, experimental::aggregation::Kind::NTH_ELEMENT>;
    std::unique_ptr<column> result = make_fixed_width_column(
                                data_type(type_to_id<ResultType>()), num_groups,
                                mask_state::UNALLOCATED, stream, mr);
    if (num_groups == 0) {
      return result;
    }
    //Returns a pair with index of nth value and a boolean denoting nth within group bounds.
    auto nth_offset_and_bounds = thrust::make_transform_iterator(
        thrust::make_zip_iterator(
          thrust::make_tuple(group_offsets.begin(), group_offsets.begin() + 1)),
        [n] __device__(auto g) ->thrust::pair<size_type, bool> {
          auto group_offset = thrust::get<0>(g);
          auto group_size = thrust::get<1>(g) - group_offset;
          bool nth_within_group = (n < 0) ? group_size >= (-n) : group_size > n;
          return thrust::pair<size_type, bool>{
            group_offset + ((n < 0) ? group_size + n : n), nth_within_group};
        });
    auto nth_offset = thrust::make_transform_iterator(
        nth_offset_and_bounds,
        [] __device__(auto of) { return of.first; });
    
    auto values_view = column_device_view::create(values, stream);
    auto values_element = thrust::make_transform_iterator(
      thrust::counting_iterator<size_type>(0),
      [col=*values_view] __device__(auto i) { return col.element<T>(i);});
    // copy nth if within group bounds to result
    thrust::gather_if(rmm::exec_policy(stream)->on(stream), 
                      nth_offset, nth_offset + num_groups,
                      nth_offset_and_bounds,
                      values_element,
                      result->mutable_view().data<ResultType>(),
                      []__device__(auto ofs) { return ofs.second; });
    // result nullmask = n within bounds and values.valid
    rmm::device_buffer result_bitmask;
    size_type result_null_count;
    std::tie(result_bitmask, result_null_count) =
        experimental::detail::valid_if(
            nth_offset_and_bounds, nth_offset_and_bounds + num_groups,
            [v=*values_view] __device__ (auto ofs) {
              auto nth_offset = ofs.first;
              auto nth_within_group = ofs.second;
              //return null if nth outside of group or value is null.
              return nth_within_group && v.is_valid(nth_offset);
            }, stream, mr);
    if(result_null_count)
      result->set_null_mask(std::move(result_bitmask), result_null_count);
    return result;
  }
  //skip nulls (equivalent to pandas nth(dropna='any'))
   template <typename T>
  std::enable_if_t<is_fixed_width<T>() && !include_nulls, std::unique_ptr<column>>
  operator()(column_view const &values,
             rmm::device_vector<size_type> const& group_labels,
             rmm::device_vector<size_type> const &group_offsets,
             size_type num_groups, size_type n,
             rmm::mr::device_memory_resource *mr, cudaStream_t stream)
  {
    using ResultType = cudf::experimental::detail::target_type_t<
        T, experimental::aggregation::Kind::NTH_ELEMENT>;
    std::unique_ptr<column> result = make_fixed_width_column(
                                data_type(type_to_id<ResultType>()), num_groups,
                                mask_state::UNALLOCATED, stream, mr);
    if (num_groups == 0) {
      return result;
    }
    auto group_sizes = group_count(values, group_labels, num_groups, mr, stream);

    auto values_view = column_device_view::create(values, stream);
    auto values_element = thrust::make_transform_iterator(
      thrust::counting_iterator<size_type>(0),
      [col=*values_view] __device__(auto i) { return col.element<T>(i);});
    auto bitmask_iterator = thrust::make_transform_iterator(
      experimental::detail::make_validity_iterator(*values_view),
      [] __device__ (auto b) { return static_cast<size_type>(b); });
    rmm::device_vector<size_type> intra_group_index(values.size());
    auto exec = rmm::exec_policy(stream)->on(stream);
    //intra group index for valids only.
    thrust::exclusive_scan_by_key(exec,
                          group_labels.begin(),
                          group_labels.end(),
                          bitmask_iterator,
                          intra_group_index.begin()
                          );
   //gather the valid index == n
   thrust::scatter_if(exec, values_element, values_element+values.size(), 
                 group_labels.begin(), //map
                 thrust::make_counting_iterator<size_type>(0),
                 result->mutable_view().begin<T>(),
                 [n, bitmask_iterator, intra_group_index=intra_group_index.begin()]
                 __device__ (auto i)  -> bool {
                 return (bitmask_iterator[i] && intra_group_index[i]==n);
                 });

   // null if n>=group_size.
   rmm::device_buffer result_bitmask;
   size_type result_null_count;

   std::tie(result_bitmask, result_null_count) = experimental::detail::valid_if(
       group_sizes->view().begin<size_type>(), group_sizes->view().end<size_type>(),
       [n] __device__(auto s) { return (n < 0) ? s >= (-n) : s > n; });
   if (result_null_count)
     result->set_null_mask(std::move(result_bitmask), result_null_count);
   return result;
  }

  template <typename T, typename... Args>
  std::enable_if_t<!is_fixed_width<T>(), std::unique_ptr<column> >
  operator()(Args&&... args) {
    //TODO for non-fixed types.
    CUDF_FAIL("Only fixed width types are supported in nth aggregation");
  }
};
} // namespace anonymous

std::unique_ptr<column> group_nth_element(
    column_view const& values,
    rmm::device_vector<size_type> const& group_labels,
    rmm::device_vector<size_type> const& group_offsets,
    size_type num_groups,
    size_type n,
    include_nulls _include_nulls,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
  
  CUDF_EXPECTS(static_cast<size_t>(values.size()) == group_labels.size(),
               "Size of values column should be same as that of group labels");

  if (_include_nulls == include_nulls::YES || !values.has_nulls()) {
    return type_dispatcher(values.type(), nth_element_functor<true>{}, 
                           values, group_labels, group_offsets, num_groups, n, 
                           mr, stream);
  } else {
  return type_dispatcher(values.type(), nth_element_functor<false>{}, 
                         values, group_labels, group_offsets, num_groups, n, 
                         mr, stream);
  }
}
}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
