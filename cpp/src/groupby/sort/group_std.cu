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

#include "group_reductions.hpp"

#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {

namespace {

template <typename ResultType, typename T>
struct var_transform
{
  column_device_view d_values;
  column_device_view d_means;
  column_device_view d_group_sizes;
  size_type const* d_group_labels;
  size_type ddof;

  __device__ ResultType operator() (size_type i) {
    if (d_values.is_null(i))
      return 0.0;
    
    ResultType x = d_values.element<T>(i);
    size_type group_idx = d_group_labels[i];
    size_type group_size = d_group_sizes.element<size_type>(group_idx);
    
    // prevent divide by zero error
    if (group_size == 0 or group_size - ddof <= 0)
      return 0.0;

    ResultType mean = d_means.element<ResultType>(group_idx);
    return (x - mean) * (x - mean) / (group_size - ddof);
  }
};


struct var_functor {
  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, std::unique_ptr<column> >
  operator()(
    column_view const& values,
    column_view const& group_means,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_labels,
    size_type ddof,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
  {
    // Running this in debug build causes a runtime error:
    // `reduce_by_key failed on 2nd step: invalid device function`
    #if !defined(__CUDACC_DEBUG__)
    using ResultType = experimental::detail::target_type_t<
                        T, experimental::aggregation::Kind::VARIANCE>;
    size_type const* d_group_labels = group_labels.data().get();
    auto values_view = column_device_view::create(values);
    auto means_view = column_device_view::create(group_means);
    auto group_size_view = column_device_view::create(group_sizes);

    std::unique_ptr<column> result =
      make_numeric_column(data_type(type_to_id<ResultType>()),
        group_sizes.size(), mask_state::UNINITIALIZED, stream, mr);

    auto d_values = *values_view;
    auto d_means = *means_view;
    auto d_group_sizes = *group_size_view;

    auto values_it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), 
      var_transform<ResultType, T>{d_values, d_means, d_group_sizes, 
                                   d_group_labels, ddof}
    );

    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          group_labels.begin(), group_labels.end(), values_it, 
                          thrust::make_discard_iterator(),
                          result->mutable_view().data<ResultType>());

    // set nulls
    auto result_view = mutable_column_device_view::create(*result);

    thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator(0), group_sizes.size(),
      [d_result=*result_view, d_group_sizes=*group_size_view, ddof]
      __device__ (size_type i){
        size_type group_size = d_group_sizes.element<size_type>(i);
        if (group_size == 0 or group_size - ddof <= 0)
          d_result.set_null(i);
        else
          d_result.set_valid(i);
      });

    return result;
    #else
    CUDF_FAIL("Groupby std/var supported in debug build");
    #endif
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic<T>::value, std::unique_ptr<column> >
  operator()(Args&&... args) {
    CUDF_FAIL("Only numeric types are supported in std/variance");
  }
};

} // namespace anonymous

std::unique_ptr<column> group_var(
    column_view const& values,
    column_view const& group_means,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_labels,
    size_type ddof,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
  return type_dispatcher(values.type(), var_functor{},
    values, group_means, group_sizes, group_labels, ddof, mr, stream);
}

} // namespace detail
} // namespace groupby
} // namespace experimental
} // namespace cudf
