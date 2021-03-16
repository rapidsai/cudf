/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/discard_iterator.h>

namespace cudf {
namespace groupby {
namespace detail {
template <aggregation::Kind K>
struct reduce_functor {
  template <typename T>
  static constexpr bool is_supported()
  {
    if (K == aggregation::SUM)
      return cudf::is_numeric<T>() || cudf::is_duration<T>() || cudf::is_fixed_point<T>();
    else if (K == aggregation::MIN or K == aggregation::MAX)
      return cudf::is_fixed_width<T>() and is_relationally_comparable<T, T>();
    else if (K == aggregation::ARGMIN or K == aggregation::ARGMAX)
      return is_relationally_comparable<T, T>();
    else
      return false;
  }

  template <typename T>
  std::enable_if_t<is_supported<T>(), std::unique_ptr<column>> operator()(
    column_view const& values,
    size_type num_groups,
    cudf::device_span<size_type const> group_labels,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    using DeviceType = device_storage_type_t<T>;
    using OpType     = cudf::detail::corresponding_operator_t<K>;
    using ResultType = cudf::detail::target_type_t<T, K>;

    auto result_type = is_fixed_point<T>()
                         ? data_type{type_to_id<ResultType>(), values.type().scale()}
                         : data_type{type_to_id<ResultType>()};

    std::unique_ptr<column> result =
      make_fixed_width_column(result_type,
                              num_groups,
                              values.has_nulls() ? mask_state::ALL_NULL : mask_state::UNALLOCATED,
                              stream,
                              mr);

    if (values.is_empty()) { return result; }

    auto result_table = mutable_table_view({*result});
    cudf::detail::initialize_with_identity(result_table, {K}, stream);

    auto resultview = mutable_column_device_view::create(result->mutable_view(), stream);
    auto valuesview = column_device_view::create(values, stream);

    if (!cudf::is_dictionary(values.type())) {
      thrust::for_each_n(rmm::exec_policy(stream),
                         thrust::make_counting_iterator(0),
                         values.size(),
                         [d_values     = *valuesview,
                          d_result     = *resultview,
                          dest_indices = group_labels.data()] __device__(auto i) {
                           cudf::detail::update_target_element<DeviceType, K, true, true>{}(
                             d_result, dest_indices[i], d_values, i);
                         });
    } else {
      thrust::for_each_n(rmm::exec_policy(stream),
                         thrust::make_counting_iterator(0),
                         values.size(),
                         [d_values     = *valuesview,
                          d_result     = *resultview,
                          dest_indices = group_labels.data()] __device__(auto i) {
                           cudf::detail::update_target_element<dictionary32, K, true, true>{}(
                             d_result, dest_indices[i], d_values, i);
                         });
    }

    return result;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not is_supported<T>(), std::unique_ptr<column>> operator()(Args&&... args)
  {
    CUDF_FAIL("Unsupported type-agg combination");
  }
};

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
