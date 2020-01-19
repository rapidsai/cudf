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

#include <quantiles/quantiles_util.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/for_each.h>

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {

namespace {

struct quantiles_functor {

  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, std::unique_ptr<column> >
  operator()(column_view const& values,
             column_view const& group_sizes,
             rmm::device_vector<size_type> const& group_offsets,
             rmm::device_vector<double> const& quantile,
             interpolation interpolation, rmm::mr::device_memory_resource* mr,
             cudaStream_t stream = 0)
  {
    using ResultType = experimental::detail::target_type_t<T, aggregation::QUANTILE>;

    auto result = make_numeric_column(data_type(type_to_id<ResultType>()), 
                                      group_sizes.size() * quantile.size(),
                                      mask_state::UNINITIALIZED, stream, mr);
    // TODO (dm): Support for no-materialize index indirection values
    // TODO (dm): Future optimization: add column order to aggregation request
    //            so that sorting isn't required. Then add support for pre-sorted

    // prepare args to be used by lambda below
    auto values_view = column_device_view::create(values);
    auto group_size_view = column_device_view::create(group_sizes);
    auto result_view = mutable_column_device_view::create(result->mutable_view());

    // For each group, calculate quantile
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator(0),
      group_offsets.size(),
      [
        d_values = *values_view,
        d_group_size = *group_size_view,
        d_result = *result_view,
        d_group_offset = group_offsets.data().get(),
        d_quantiles = quantile.data().get(),
        num_quantiles = quantile.size(),
        interpolation
      ] __device__ (size_type i) {
        size_type segment_size = d_group_size.element<size_type>(i);

        auto selector = [i, &d_values, d_group_offset] (size_type j) {
          return d_values.element<T>(d_group_offset[i] + j);
        };
        
        thrust::transform(thrust::seq, d_quantiles, d_quantiles + num_quantiles,
                          d_result.data<ResultType>() + i * num_quantiles,
                          [selector, segment_size, interpolation] (auto q) { 
                            return experimental::detail::select_quantile<ResultType>(
                              selector, segment_size, q, interpolation); 
                          });

        for (size_t j = 0; j < num_quantiles; j++) {
          size_type group_size = d_group_size.element<size_type>(i);
          if (group_size == 0)
            d_result.set_null(i * num_quantiles + j);
          else
            d_result.set_valid(i * num_quantiles + j);
        }

        // DEAR REVIEWER, would you prefer I do this instead of the loop above?
        // thrust::for_each_n(thrust::seq,
        //   thrust::make_counting_iterator(0), num_quantiles,
        //   [&d_result, &d_group_size, num_quantiles, i] (size_type j) {
        //     size_type group_size = d_group_size.element<size_type>(i);
        //     if (group_size == 0)
        //       d_result.set_null(i * num_quantiles + j);
        //     else
        //       d_result.set_valid(i * num_quantiles + j);
        //   });
      }
    );

    return result;
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic<T>::value, std::unique_ptr<column> >
  operator()(Args&&... args) {
    CUDF_FAIL("Only arithmetic types are supported in quantiles");
  }
};

} // namespace anonymous


// TODO: add optional check for is_sorted. Use context.flag_sorted
std::unique_ptr<column> group_quantiles(
    column_view const& values,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_offsets,
    std::vector<double> const& quantiles,
    interpolation interp,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
  rmm::device_vector<double> dv_quantiles(quantiles);

  return type_dispatcher(values.type(), quantiles_functor{},
                         values, group_sizes, group_offsets,
                         dv_quantiles, interp, mr, stream);
}

}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
