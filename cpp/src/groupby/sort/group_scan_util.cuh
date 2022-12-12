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

#include <reductions/struct_minmax_util.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

namespace cudf {
namespace groupby {
namespace detail {
// Error case when no other overload or specialization is available
template <aggregation::Kind K, typename T, typename Enable = void>
struct group_scan_functor {
  template <typename... Args>
  static std::unique_ptr<column> invoke(Args&&...)
  {
    CUDF_FAIL("Unsupported groupby scan type-agg combination.");
  }
};

template <aggregation::Kind K>
struct group_scan_dispatcher {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& values,
                                     size_type num_groups,
                                     cudf::device_span<cudf::size_type const> group_labels,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return group_scan_functor<K, T>::invoke(values, num_groups, group_labels, stream, mr);
  }
};

/**
 * @brief Check if the given aggregation K with data type T is supported in groupby scan.
 */
template <aggregation::Kind K, typename T>
static constexpr bool is_group_scan_supported()
{
  if (K == aggregation::SUM)
    return cudf::is_numeric<T>() || cudf::is_duration<T>() || cudf::is_fixed_point<T>();
  else if (K == aggregation::MIN or K == aggregation::MAX)
    return not cudf::is_dictionary<T>() and
           (is_relationally_comparable<T, T>() or std::is_same_v<T, cudf::struct_view>);
  else
    return false;
}

template <aggregation::Kind K, typename T>
struct group_scan_functor<K, T, std::enable_if_t<is_group_scan_supported<K, T>()>> {
  static std::unique_ptr<column> invoke(column_view const& values,
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

    // Perform segmented scan.
    auto const do_scan = [&](auto const& inp_iter, auto const& out_iter, auto const& binop) {
      thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                    group_labels.begin(),
                                    group_labels.end(),
                                    inp_iter,
                                    out_iter,
                                    thrust::equal_to{},
                                    binop);
    };

    if (values.has_nulls()) {
      auto input = thrust::make_transform_iterator(
        make_null_replacement_iterator(*values_view, OpType::template identity<DeviceType>()),
        thrust::identity<ResultDeviceType>{});
      do_scan(input, result_view->begin<ResultDeviceType>(), OpType{});
      result->set_null_mask(cudf::detail::copy_bitmask(values, stream));
    } else {
      auto input = thrust::make_transform_iterator(values_view->begin<DeviceType>(),
                                                   thrust::identity<ResultDeviceType>{});
      do_scan(input, result_view->begin<ResultDeviceType>(), OpType{});
    }
    return result;
  }
};

template <aggregation::Kind K>
struct group_scan_functor<K,
                          cudf::string_view,
                          std::enable_if_t<is_group_scan_supported<K, cudf::string_view>()>> {
  static std::unique_ptr<column> invoke(column_view const& values,
                                        size_type num_groups,
                                        cudf::device_span<cudf::size_type const> group_labels,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
  {
    using OpType = cudf::detail::corresponding_operator_t<K>;

    if (values.is_empty()) { return cudf::make_empty_column(cudf::type_id::STRING); }

    // create an empty output vector we can fill with string_view instances
    auto results_vector = rmm::device_uvector<string_view>(values.size(), stream);

    auto values_view = column_device_view::create(values, stream);

    // Perform segmented scan.
    auto const do_scan = [&](auto const& inp_iter, auto const& out_iter, auto const& binop) {
      thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                    group_labels.begin(),
                                    group_labels.end(),
                                    inp_iter,
                                    out_iter,
                                    thrust::equal_to{},
                                    binop);
    };

    if (values.has_nulls()) {
      auto input = make_null_replacement_iterator(
        *values_view, OpType::template identity<string_view>(), values.has_nulls());
      do_scan(input, results_vector.begin(), OpType{});
    } else {
      do_scan(values_view->begin<string_view>(), results_vector.begin(), OpType{});
    }

    // turn the string_view vector into a strings column
    auto results = make_strings_column(results_vector, string_view{}, stream, mr);
    if (values.has_nulls())
      results->set_null_mask(cudf::detail::copy_bitmask(values, stream), values.null_count());
    return results;
  }
};

template <aggregation::Kind K>
struct group_scan_functor<K,
                          cudf::struct_view,
                          std::enable_if_t<is_group_scan_supported<K, cudf::struct_view>()>> {
  static std::unique_ptr<column> invoke(column_view const& values,
                                        size_type num_groups,
                                        cudf::device_span<cudf::size_type const> group_labels,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
  {
    if (values.is_empty()) { return cudf::empty_like(values); }

    // Create a gather map containing indices of the prefix min/max elements within each group.
    auto gather_map = rmm::device_uvector<size_type>(values.size(), stream);

    auto const binop_generator =
      cudf::reduction::detail::comparison_binop_generator::create<K>(values, stream);
    thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                  group_labels.begin(),
                                  group_labels.end(),
                                  thrust::make_counting_iterator<size_type>(0),
                                  gather_map.begin(),
                                  thrust::equal_to{},
                                  binop_generator.binop());

    //
    // Gather the children elements of the prefix min/max struct elements first.
    //
    // Typically, we should use `get_sliced_child` for each child column to properly handle the
    // input if it is a sliced view. However, since the input to this function is just generated
    // from groupby internal APIs which is never a sliced view, we just use `child_begin` and
    // `child_end` iterators for simplicity.
    auto scanned_children =
      cudf::detail::gather(
        table_view(std::vector<column_view>{values.child_begin(), values.child_end()}),
        gather_map,
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::detail::negative_index_policy::NOT_ALLOWED,
        stream,
        mr)
        ->release();

    // After gathering the children elements, we need to push down nulls from the root structs
    // column to them.
    if (values.has_nulls()) {
      for (std::unique_ptr<column>& child : scanned_children) {
        child = structs::detail::superimpose_nulls(
          values.null_mask(), values.null_count(), std::move(child), stream, mr);
      }
    }

    return make_structs_column(values.size(),
                               std::move(scanned_children),
                               values.null_count(),
                               cudf::detail::copy_bitmask(values, stream, mr),
                               stream,
                               mr);
  }
};

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
