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
#include <cudf/table/table_device_view.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/detail/valid_if.cuh>
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

    rmm::device_buffer result_bitmask;
    size_type result_null_count;
    std::tie(result_bitmask, result_null_count) = 
      experimental::detail::valid_if(
        group_sizes.begin<size_type>(), group_sizes.end<size_type>(),
        [] __device__ (auto s) { return s > 0; });

    std::unique_ptr<column> result;
    if (result_null_count > 0) {
      result = make_fixed_width_column(data_type(type_to_id<ResultType>()), 
                                       num_groups,
                                       std::move(result_bitmask),
                                       result_null_count,
                                       stream, mr);
    } else {
      result = make_fixed_width_column(data_type(type_to_id<ResultType>()), 
                                       num_groups,
                                       mask_state::UNALLOCATED,
                                       stream, mr);
    }

    if (values.size() == 0) {
      return result;
    }

    thrust::fill(rmm::exec_policy(stream)->on(stream),
      result->mutable_view().begin<ResultType>(), 
      result->mutable_view().end<ResultType>(),
      OpType::template identity<ResultType>());

    auto resultview = mutable_column_device_view::create(result->mutable_view());
    auto valuesview = column_device_view::create(values);

    thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator(0), values.size(),
      [
        d_values = *valuesview,
        d_result = *resultview,
        dest_indices = group_labels.data().get()
      ] __device__ (auto i) {
        if (d_values.is_valid(i))
          switch (k) {
          case aggregation::Kind::SUM:
            atomicAdd(d_result.data<ResultType>() + dest_indices[i],
                      static_cast<ResultType>(d_values.element<T>(i)));
            break;
          case aggregation::Kind::MIN:
            atomicMin(d_result.data<ResultType>() + dest_indices[i],
                      static_cast<ResultType>(d_values.element<T>(i)));
            break;
          case aggregation::Kind::MAX:
            atomicMax(d_result.data<ResultType>() + dest_indices[i],
                      static_cast<ResultType>(d_values.element<T>(i)));
            break;
          default:
            break;
          }
      });
    
    return result;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not is_supported<T>(), std::unique_ptr<column> >
  operator()(Args&&... args) {
    CUDF_FAIL("Unsupported type-agg combination");
  }
};

}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
