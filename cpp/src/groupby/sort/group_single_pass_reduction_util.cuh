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

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/aggregation.hpp>
#include <cudf/types.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/iterator/discard_iterator.h>

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {

template <aggregation::Kind k>
struct reduce_functor {

  template <typename T>
  static constexpr bool is_supported(){
    if (cudf::is_numeric<T>())
      return true;
    else if (cudf::is_timestamp<T>() and 
              (k == aggregation::MIN or k == aggregation::MAX))
      return true;
    else
      return false;
  }

  template <typename T>
  std::enable_if_t<is_supported<T>(), std::unique_ptr<column> >
  operator()(column_view const& values,
             column_view const& group_sizes,
             rmm::device_vector<cudf::size_type> const& group_labels,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream)
  {
    CUDF_EXPECTS(static_cast<size_t>(values.size()) == group_labels.size(),
      "Size of values column should be same as that of group labels");

    using OpType = cudf::experimental::detail::corresponding_operator_t<k>;
    using ResultType = cudf::experimental::detail::target_type_t<T, k>;
    size_type num_groups = group_sizes.size();

    auto result = make_fixed_width_column(data_type(type_to_id<ResultType>()), 
                                          num_groups,
                                          values.nullable() 
                                            ? mask_state::UNINITIALIZED
                                            : mask_state::UNALLOCATED,
                                          stream, mr);
    auto op = OpType{};

    if (values.size() == 0) {
      return result;
    }

    if (values.nullable()) {
      T default_value = OpType::template identity<T>();
      auto device_values = column_device_view::create(values);
      auto val_it = cudf::experimental::detail::make_null_replacement_iterator(
                        *device_values, default_value);

      // Without this transform, thrust throws a runtime error
      auto it = thrust::make_transform_iterator(val_it,
                          [] __device__ (auto i) -> ResultType { return i; });
      
      thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
        // Input keys
          group_labels.begin(), group_labels.end(),
        // Input values
          it,
        // Output keys
          thrust::make_discard_iterator(),
        // Output values
          result->mutable_view().begin<ResultType>(),
        // comparator and operation
          thrust::equal_to<size_type>(), op);

      auto result_view = mutable_column_device_view::create(*result);
      auto group_size_view = column_device_view::create(group_sizes);

      thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator(0), group_sizes.size(),
        [d_result=*result_view, d_group_sizes=*group_size_view]
        __device__ (size_type i){
          size_type group_size = d_group_sizes.element<size_type>(i);
          if (group_size == 0)
            d_result.set_null(i);
          else
            d_result.set_valid(i);
        });
    } else {
      auto it = thrust::make_transform_iterator(values.data<T>(),
                          [] __device__ (auto i) -> ResultType { return i; });
      thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                            group_labels.begin(), group_labels.end(),
                            it, thrust::make_discard_iterator(),
                            result->mutable_view().begin<ResultType>(),
                            thrust::equal_to<size_type>(), op);
    }
    
    return result;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not is_supported<T>(), std::unique_ptr<column> >
  operator()(Args&&... args) {
    CUDF_FAIL("Only numeric types are supported in variance");
  }
};

}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
