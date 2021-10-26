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
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

namespace cudf {
namespace groupby {
namespace detail {

/**
 * @brief ArgMin binary operator with index values into input column.
 *
 * @tparam T Type of the underlying column. Must support '<' operator.
 */
template <typename T>
struct ArgMin {
  column_device_view const d_col;
  CUDA_DEVICE_CALLABLE auto operator()(size_type const& lhs, size_type const& rhs) const
  {
    // The extra bounds checking is due to issue github.com/rapidsai/cudf/9156 and
    // github.com/NVIDIA/thrust/issues/1525
    // where invalid random values may be passed here by thrust::reduce_by_key
    if (lhs < 0 || lhs >= d_col.size() || d_col.is_null(lhs)) { return rhs; }
    if (rhs < 0 || rhs >= d_col.size() || d_col.is_null(rhs)) { return lhs; }
    return d_col.element<T>(lhs) < d_col.element<T>(rhs) ? lhs : rhs;
  }
};

/**
 * @brief ArgMax binary operator with index values into input column.
 *
 * @tparam T Type of the underlying column. Must support '<' operator.
 */
template <typename T>
struct ArgMax {
  column_device_view const d_col;
  CUDA_DEVICE_CALLABLE auto operator()(size_type const& lhs, size_type const& rhs) const
  {
    // The extra bounds checking is due to issue github.com/rapidsai/cudf/9156 and
    // github.com/NVIDIA/thrust/issues/1525
    // where invalid random values may be passed here by thrust::reduce_by_key
    if (lhs < 0 || lhs >= d_col.size() || d_col.is_null(lhs)) { return rhs; }
    if (rhs < 0 || rhs >= d_col.size() || d_col.is_null(rhs)) { return lhs; }
    return d_col.element<T>(rhs) < d_col.element<T>(lhs) ? lhs : rhs;
  }
};

/**
 * @brief Value accessor for column which supports dictionary column too.
 *
 * @tparam T Type of the underlying column. For dictionary column, type of the key column.
 */
template <typename T>
struct value_accessor {
  column_device_view const col;
  bool const is_dict;
  value_accessor(column_device_view const& col) : col(col), is_dict(cudf::is_dictionary(col.type()))
  {
  }

  __device__ T value(size_type i) const
  {
    if (is_dict) {
      auto keys = col.child(dictionary_column_view::keys_column_index);
      return keys.element<T>(static_cast<size_type>(col.element<dictionary32>(i)));
    } else {
      return col.element<T>(i);
    }
  }
  __device__ auto operator()(size_type i) const { return value(i); }
};

/**
 * @brief Null replaced value accessor for column which supports dictionary column too.
 * For null value, returns null `init` value
 *
 * @tparam T Type of the underlying column. For dictionary column, type of the key column.
 */
template <typename T>
struct null_replaced_value_accessor : value_accessor<T> {
  using super_t = value_accessor<T>;
  bool const has_nulls;
  T const init;
  null_replaced_value_accessor(column_device_view const& col, T const& init, bool const has_nulls)
    : super_t(col), init(init), has_nulls(has_nulls)
  {
  }
  __device__ T operator()(size_type i) const
  {
    return has_nulls && super_t::col.is_null_nocheck(i) ? init : super_t::value(i);
  }
};

template <aggregation::Kind K>
struct reduce_functor {
  template <typename T>
  static constexpr bool is_natively_supported()
  {
    switch (K) {
      case aggregation::SUM:
        return cudf::is_numeric<T>() || cudf::is_duration<T>() || cudf::is_fixed_point<T>();
      case aggregation::PRODUCT: return cudf::detail::is_product_supported<T>();
      case aggregation::MIN:
      case aggregation::MAX:
        return cudf::is_fixed_width<T>() and is_relationally_comparable<T, T>();
      case aggregation::ARGMIN:
      case aggregation::ARGMAX: return is_relationally_comparable<T, T>();
      default: return false;
    }
  }

  template <typename T>
  std::enable_if_t<is_natively_supported<T>(), std::unique_ptr<column>> operator()(
    column_view const& values,
    size_type num_groups,
    cudf::device_span<size_type const> group_labels,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    using DeviceType  = device_storage_type_t<T>;
    using OpType      = cudf::detail::corresponding_operator_t<K>;
    using ResultType  = cudf::detail::target_type_t<T, K>;
    using ResultDType = device_storage_type_t<ResultType>;

    auto result_type = is_fixed_point<ResultType>()
                         ? data_type{type_to_id<ResultType>(), values.type().scale()}
                         : data_type{type_to_id<ResultType>()};

    std::unique_ptr<column> result =
      make_fixed_width_column(result_type, num_groups, mask_state::UNALLOCATED, stream, mr);

    if (values.is_empty()) { return result; }

    auto resultview = mutable_column_device_view::create(result->mutable_view(), stream);
    auto valuesview = column_device_view::create(values, stream);
    if constexpr (K == aggregation::ARGMAX || K == aggregation::ARGMIN) {
      using OpType =
        std::conditional_t<(K == aggregation::ARGMAX), ArgMax<DeviceType>, ArgMin<DeviceType>>;
      thrust::reduce_by_key(rmm::exec_policy(stream),
                            group_labels.data(),
                            group_labels.data() + group_labels.size(),
                            thrust::make_counting_iterator<ResultType>(0),
                            thrust::make_discard_iterator(),
                            resultview->begin<ResultType>(),
                            thrust::equal_to<size_type>{},
                            OpType{*valuesview});
    } else {
      auto init  = OpType::template identity<DeviceType>();
      auto begin = cudf::detail::make_counting_transform_iterator(
        0, null_replaced_value_accessor{*valuesview, init, values.has_nulls()});
      thrust::reduce_by_key(rmm::exec_policy(stream),
                            group_labels.data(),
                            group_labels.data() + group_labels.size(),
                            begin,
                            thrust::make_discard_iterator(),
                            resultview->begin<ResultDType>(),
                            thrust::equal_to<size_type>{},
                            OpType{});
    }

    if (values.has_nulls()) {
      rmm::device_uvector<bool> validity(num_groups, stream);
      thrust::reduce_by_key(rmm::exec_policy(stream),
                            group_labels.data(),
                            group_labels.data() + group_labels.size(),
                            cudf::detail::make_validity_iterator(*valuesview),
                            thrust::make_discard_iterator(),
                            validity.begin(),
                            thrust::equal_to<size_type>{},
                            thrust::logical_or<bool>{});
      auto [null_mask, null_count] = cudf::detail::valid_if(
        validity.begin(), validity.end(), thrust::identity<bool>{}, stream, mr);
      result->set_null_mask(std::move(null_mask));
      result->set_null_count(null_count);
    }
    return result;
  }

  template <typename T>
  std::enable_if_t<not is_natively_supported<T>() and std::is_same_v<T, struct_view> and
                     (K == aggregation::ARGMIN or K == aggregation::ARGMAX),
                   std::unique_ptr<column>>
  operator()(column_view const& values,
             size_type num_groups,
             cudf::device_span<size_type const> group_labels,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr)
  {
    using ResultType = cudf::detail::target_type_t<T, K>;
    auto result      = make_fixed_width_column(
      data_type{type_to_id<ResultType>()}, num_groups, mask_state::UNALLOCATED, stream, mr);

    if (values.is_empty()) { return result; }

    // The comparison and null orders for finding arg_min/arg_max for the min/max elements.
    auto const comp_order      = K == aggregation::ARGMIN ? order::ASCENDING : order::DESCENDING;
    auto const null_precedence = K == aggregation::ARGMIN ? null_order::AFTER : null_order::BEFORE;

    auto const flattened_values =
      structs::detail::flatten_nested_columns(table_view{{values}},
                                              {comp_order},
                                              {null_precedence},
                                              structs::detail::column_nullability::MATCH_INCOMING);
    auto const values_ptr = table_device_view::create(flattened_values, stream);

    // Perform reduction to find arg_min/arg_max.
    auto const do_reduction = [&](auto const& inp_iter, auto const& out_iter, auto const& comp) {
      thrust::reduce_by_key(rmm::exec_policy(stream),
                            group_labels.data(),
                            group_labels.data() + group_labels.size(),
                            inp_iter,
                            thrust::make_discard_iterator(),
                            out_iter,
                            thrust::equal_to<size_type>{},
                            comp);
    };

    auto const count_iter   = thrust::make_counting_iterator<ResultType>(0);
    auto const result_begin = result->mutable_view().template begin<ResultType>();
    if (!values.has_nulls()) {
      auto const comp = row_lexicographic_comparator<false>(*values_ptr,
                                                            *values_ptr,
                                                            flattened_values.orders().data(),
                                                            flattened_values.null_orders().data());
      do_reduction(count_iter, result_begin, comp);
    } else {
      auto const comp = row_lexicographic_comparator<true>(*values_ptr,
                                                           *values_ptr,
                                                           flattened_values.orders().data(),
                                                           flattened_values.null_orders().data());
      do_reduction(count_iter, result_begin, comp);

      // Generate bitmask for the output from the input.
      auto const values_ptr = column_device_view::create(values, stream);
      auto validity         = rmm::device_uvector<bool>(num_groups, stream);
      do_reduction(cudf::detail::make_validity_iterator(*values_ptr),
                   validity.begin(),
                   thrust::logical_or<bool>{});

      auto [null_mask, null_count] = cudf::detail::valid_if(
        validity.begin(), validity.end(), thrust::identity<bool>{}, stream, mr);
      result->set_null_mask(std::move(null_mask));
      result->set_null_count(null_count);
    }

    return result;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not is_natively_supported<T>() and
                     (not std::is_same_v<T, struct_view> or
                      (K != aggregation::ARGMIN or K != aggregation::ARGMAX)),
                   std::unique_ptr<column>>
  operator()(Args&&... args)
  {
    CUDF_FAIL("Unsupported type-agg combination");
  }
};

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
