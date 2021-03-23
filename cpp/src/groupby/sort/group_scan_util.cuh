/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf/detail/null_mask.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/scan.h>

namespace cudf {
namespace groupby {
namespace detail {
template <aggregation::Kind K>
struct scan_functor {
  template <typename T>
  static constexpr bool is_supported()
  {
    if (K == aggregation::SUM)
      return cudf::is_numeric<T>() || cudf::is_duration<T>() || cudf::is_fixed_point<T>();
    else if (K == aggregation::MIN or K == aggregation::MAX)
      return cudf::is_fixed_width<T>() and is_relationally_comparable<T, T>();
    else
      return false;
  }

  template <typename T>
  std::enable_if_t<is_supported<T>(), std::unique_ptr<column>> operator()(
    column_view const& values,
    size_type num_groups,
    cudf::device_span<cudf::size_type const> group_labels,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    using DeviceType       = device_storage_type_t<T>;
    using OpType           = cudf::detail::corresponding_operator_t<K>;
    using ResultType       = cudf::detail::target_type_t<T, K>;
    using ResultDeviceType = device_storage_type_t<ResultType>;

    auto result_type = is_fixed_point<T>()
                         ? data_type{type_to_id<ResultType>(), values.type().scale()}
                         : data_type{type_to_id<ResultType>()};

    std::unique_ptr<column> result =
      make_fixed_width_column(result_type, values.size(), mask_state::UNALLOCATED, stream, mr);

    if (values.is_empty()) { return result; }

    auto result_table = mutable_table_view({*result});
    cudf::detail::initialize_with_identity(result_table, {K}, stream);

    auto result_view = mutable_column_device_view::create(result->mutable_view(), stream);
    auto values_view = column_device_view::create(values, stream);

    if (values.has_nulls()) {
      auto input = thrust::make_transform_iterator(
        make_null_replacement_iterator(*values_view, OpType::template identity<DeviceType>()),
        thrust::identity<ResultDeviceType>{});
      thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                    group_labels.begin(),
                                    group_labels.end(),
                                    input,
                                    result_view->begin<ResultDeviceType>(),
                                    thrust::equal_to<size_type>{},
                                    OpType{});
      result->set_null_mask(cudf::detail::copy_bitmask(values, stream));
    } else {
      auto input = thrust::make_transform_iterator(values_view->begin<DeviceType>(),
                                                   thrust::identity<ResultDeviceType>{});
      thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                    group_labels.begin(),
                                    group_labels.end(),
                                    input,
                                    result_view->begin<ResultDeviceType>(),
                                    thrust::equal_to<size_type>{},
                                    OpType{});
    }
    return result;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not is_supported<T>(), std::unique_ptr<column>> operator()(Args&&... args)
  {
    CUDF_FAIL("Unsupported groupby scan type-agg combination");
  }
};

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
