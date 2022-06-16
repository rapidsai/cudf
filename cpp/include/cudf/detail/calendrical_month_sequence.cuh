/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/datetime_ops.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
struct calendrical_month_sequence_functor {
  template <typename T>
  std::enable_if_t<cudf::is_timestamp_t<T>::value, std::unique_ptr<cudf::column>> operator()(
    size_type n,
    scalar const& input,
    size_type months,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
  {
    // Return empty column if n = 0
    if (n == 0) return cudf::make_empty_column(input.type());

    auto const device_input =
      get_scalar_device_view(static_cast<cudf::scalar_type_t<T>&>(const_cast<scalar&>(input)));
    auto output_column_type = cudf::data_type{cudf::type_to_id<T>()};
    auto output             = cudf::make_fixed_width_column(
      output_column_type, n, cudf::mask_state::UNALLOCATED, stream, mr);

    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(n),
                      output->mutable_view().begin<T>(),
                      [initial = device_input, months] __device__(size_type i) {
                        return datetime::detail::add_calendrical_months_with_scale_back(
                          initial.value(), cuda::std::chrono::months{i * months});
                      });

    return output;
  }

  template <typename T, typename... Args>
  std::enable_if_t<!cudf::is_timestamp_t<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUDF_FAIL("Cannot make a date_range of a non-datetime type");
  }
};

}  // namespace detail
}  // namespace cudf
