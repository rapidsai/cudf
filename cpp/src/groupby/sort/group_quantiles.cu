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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>

#include <groupby/sort/group_reductions.hpp>
#include <quantiles/quantiles_util.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/for_each.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

template <typename ResultType, typename Iterator>
struct calculate_quantile_fn {
  Iterator values_iter;
  column_device_view d_group_size;
  mutable_column_device_view d_result;
  size_type const* d_group_offset;
  double const* d_quantiles;
  size_type num_quantiles;
  interpolation interpolation;

  __device__ void operator()(size_type i)
  {
    size_type segment_size = d_group_size.element<size_type>(i);

    auto d_itr = values_iter + d_group_offset[i];
    thrust::transform(thrust::seq,
                      d_quantiles,
                      d_quantiles + num_quantiles,
                      d_result.data<ResultType>() + i * num_quantiles,
                      [d_itr, segment_size, interpolation = interpolation](auto q) {
                        return cudf::detail::select_quantile_data<ResultType>(
                          d_itr, segment_size, q, interpolation);
                      });

    size_type offset = i * num_quantiles;
    thrust::for_each_n(thrust::seq,
                       thrust::make_counting_iterator(0),
                       num_quantiles,
                       [d_result = d_result, segment_size, offset](size_type j) {
                         if (segment_size == 0)
                           d_result.set_null(offset + j);
                         else
                           d_result.set_valid(offset + j);
                       });
  }
};

struct quantiles_functor {
  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, std::unique_ptr<column>> operator()(
    column_view const& values,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_offsets,
    size_type const num_groups,
    rmm::device_vector<double> const& quantile,
    interpolation interpolation,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    using ResultType = cudf::detail::target_type_t<T, aggregation::QUANTILE>;

    auto result = make_numeric_column(data_type(type_to_id<ResultType>()),
                                      group_sizes.size() * quantile.size(),
                                      mask_state::UNINITIALIZED,
                                      stream,
                                      mr);
    // TODO (dm): Support for no-materialize index indirection values
    // TODO (dm): Future optimization: add column order to aggregation request
    //            so that sorting isn't required. Then add support for pre-sorted

    // prepare args to be used by lambda below
    auto values_view     = column_device_view::create(values, stream);
    auto group_size_view = column_device_view::create(group_sizes, stream);
    auto result_view     = mutable_column_device_view::create(result->mutable_view(), stream);

    // For each group, calculate quantile
    if (!cudf::is_dictionary(values.type())) {
      auto values_iter = values_view->begin<T>();
      thrust::for_each_n(rmm::exec_policy(stream),
                         thrust::make_counting_iterator(0),
                         num_groups,
                         calculate_quantile_fn<ResultType, decltype(values_iter)>{
                           values_iter,
                           *group_size_view,
                           *result_view,
                           group_offsets.data().get(),
                           quantile.data().get(),
                           static_cast<size_type>(quantile.size()),
                           interpolation});
    } else {
      auto values_iter = cudf::dictionary::detail::make_dictionary_iterator<T>(*values_view);
      thrust::for_each_n(rmm::exec_policy(stream),
                         thrust::make_counting_iterator(0),
                         num_groups,
                         calculate_quantile_fn<ResultType, decltype(values_iter)>{
                           values_iter,
                           *group_size_view,
                           *result_view,
                           group_offsets.data().get(),
                           quantile.data().get(),
                           static_cast<size_type>(quantile.size()),
                           interpolation});
    }

    return result;
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic<T>::value, std::unique_ptr<column>> operator()(
    Args&&... args)
  {
    CUDF_FAIL("Only arithmetic types are supported in quantiles");
  }
};

}  // namespace

// TODO: add optional check for is_sorted. Use context.flag_sorted
std::unique_ptr<column> group_quantiles(column_view const& values,
                                        column_view const& group_sizes,
                                        rmm::device_vector<size_type> const& group_offsets,
                                        size_type const num_groups,
                                        std::vector<double> const& quantiles,
                                        interpolation interp,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  rmm::device_vector<double> dv_quantiles(quantiles);

  auto values_type = cudf::is_dictionary(values.type())
                       ? dictionary_column_view(values).keys().type()
                       : values.type();

  return type_dispatcher(values_type,
                         quantiles_functor{},
                         values,
                         group_sizes,
                         group_offsets,
                         num_groups,
                         dv_quantiles,
                         interp,
                         stream,
                         mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
