/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

template <typename ResultType, typename Iterator>
struct var_transform {
  // column_device_view d_values;
  Iterator values_iter;
  ResultType const* d_means;
  size_type const* d_group_sizes;
  size_type const* d_group_labels;
  size_type ddof;

  __device__ ResultType operator()(size_type i)
  {
    // if (d_values.is_null(i)) return 0.0;
    if (!thrust::get<1>(values_iter[i])) return 0.0;

    // ResultType x         = d_values.element<T>(i);
    ResultType x = static_cast<ResultType>(thrust::get<0>(values_iter[i]));

    size_type group_idx  = d_group_labels[i];
    size_type group_size = d_group_sizes[group_idx];  //.element<size_type>(group_idx);

    // prevent divide by zero error
    if (group_size == 0 or group_size - ddof <= 0) return 0.0;

    ResultType mean = d_means[group_idx];  //.element<ResultType>(group_idx);
    return (x - mean) * (x - mean) / (group_size - ddof);
  }
};

template <typename ResultType, typename Iterator>
void reduce_by_key_fn(Iterator values_iter,
                      rmm::device_vector<size_type> const& group_labels,
                      ResultType const* d_means,
                      size_type const* d_group_sizes,
                      size_type ddof,
                      ResultType* d_result,
                      cudaStream_t stream)
{
  auto var_iter = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    var_transform<ResultType, decltype(values_iter)>{
      values_iter, d_means, d_group_sizes, group_labels.data().get(), ddof});

  thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                        group_labels.begin(),
                        group_labels.end(),
                        var_iter,
                        thrust::make_discard_iterator(),
                        d_result);
}

struct var_functor {
  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, std::unique_ptr<column>> operator()(
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
    using ResultType = cudf::detail::target_type_t<T, aggregation::Kind::VARIANCE>;

    std::unique_ptr<column> result = make_numeric_column(data_type(type_to_id<ResultType>()),
                                                         group_sizes.size(),
                                                         mask_state::UNINITIALIZED,
                                                         stream,
                                                         mr);

    auto values_view = column_device_view::create(values, stream);
    auto d_values    = *values_view;

    auto d_group_labels = group_labels.data().get();
    auto d_means        = group_means.data<ResultType>();
    auto d_group_sizes  = group_sizes.data<size_type>();
    auto d_result       = result->mutable_view().data<ResultType>();

    if (!cudf::is_dictionary(values.type())) {
      if (values.has_nulls()) {
        auto values_iter = d_values.pair_begin<T, true>();
        reduce_by_key_fn(values_iter, group_labels, d_means, d_group_sizes, ddof, d_result, stream);
      } else {
        auto values_iter = d_values.pair_begin<T, false>();
        reduce_by_key_fn(values_iter, group_labels, d_means, d_group_sizes, ddof, d_result, stream);
      }
    } else {  // dictionary column type uses special pair iterator
      if (values.has_nulls()) {
        auto values_iter =
          cudf::dictionary::detail::make_dictionary_pair_iterator<T, true>(*values_view);
        reduce_by_key_fn(values_iter, group_labels, d_means, d_group_sizes, ddof, d_result, stream);
      } else {
        auto values_iter =
          cudf::dictionary::detail::make_dictionary_pair_iterator<T, false>(*values_view);
        reduce_by_key_fn(values_iter, group_labels, d_means, d_group_sizes, ddof, d_result, stream);
      }
    }

    // set nulls
    auto result_view = mutable_column_device_view::create(*result, stream);
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator(0),
                       group_sizes.size(),
                       [d_result = *result_view, d_group_sizes, ddof] __device__(size_type i) {
                         size_type group_size = d_group_sizes[i];
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
  std::enable_if_t<!std::is_arithmetic<T>::value, std::unique_ptr<column>> operator()(
    Args&&... args)
  {
    CUDF_FAIL("Only numeric types are supported in std/variance");
  }
};

}  // namespace

std::unique_ptr<column> group_var(column_view const& values,
                                  column_view const& group_means,
                                  column_view const& group_sizes,
                                  rmm::device_vector<size_type> const& group_labels,
                                  size_type ddof,
                                  rmm::mr::device_memory_resource* mr,
                                  cudaStream_t stream)
{
  auto values_type = cudf::is_dictionary(values.type())
                       ? dictionary_column_view(values).keys().type()
                       : values.type();

  return type_dispatcher(
    values_type, var_functor{}, values, group_means, group_sizes, group_labels, ddof, mr, stream);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
