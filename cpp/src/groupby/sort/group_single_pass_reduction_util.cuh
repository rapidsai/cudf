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
#include <cudf/types.hpp>
#include <cudf/utilities/output_writer_iterator.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

namespace cudf {
namespace groupby {
namespace detail {

// ArgMin binary operator with tuple of (value, index)
template <typename T>
struct ArgMin {
  CUDA_HOST_DEVICE_CALLABLE auto operator()(thrust::tuple<T, size_type> const& lhs,
                                            thrust::tuple<T, size_type> const& rhs) const
  {
    if (thrust::get<1>(lhs) == cudf::detail::ARGMIN_SENTINEL)
      return rhs;
    else if (thrust::get<1>(rhs) == cudf::detail::ARGMIN_SENTINEL)
      return lhs;
    else
      return thrust::get<0>(lhs) < thrust::get<0>(rhs) ? lhs : rhs;
  }
};

// ArgMax binary operator with tuple of (value, index)
template <typename T>
struct ArgMax {
  CUDA_HOST_DEVICE_CALLABLE auto operator()(thrust::tuple<T, size_type> const& lhs,
                                            thrust::tuple<T, size_type> const& rhs) const
  {
    if (thrust::get<1>(lhs) == cudf::detail::ARGMIN_SENTINEL)
      return rhs;
    else if (thrust::get<1>(rhs) == cudf::detail::ARGMIN_SENTINEL)
      return lhs;
    else
      return thrust::get<0>(lhs) > thrust::get<0>(rhs) ? lhs : rhs;
  }
};

/**
 * @brief Functor to store the index of tuple to column.
 *
 */
struct tuple_index_to_column {
  mutable_column_device_view d_result;
  template <typename T>
  __device__ void operator()(size_type i, thrust::tuple<T, size_type> const& rhs)
  {
    d_result.element<size_type>(i) = thrust::get<1>(rhs);
  }
};

/**
 * @brief Functor to store the boolean value to null mask.
 */
struct bool_to_nullmask {
  mutable_column_device_view d_result;
  __device__ void operator()(size_type i, bool rhs)
  {
    if (rhs) {
      d_result.set_valid(i);
    } else {
      d_result.set_null(i);
    }
  }
};

/**
 * @brief Returns index for non-null element, and SENTINEL for null element in a column.
 *
 */
struct null_as_sentinel {
  column_device_view const col;
  size_type const SENTINEL;
  __device__ size_type operator()(size_type i) const { return col.is_null(i) ? SENTINEL : i; }
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
  CUDA_HOST_DEVICE_CALLABLE auto operator()(size_type i) const { return value(i); }
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
  static constexpr bool is_supported()
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
  std::enable_if_t<is_supported<T>(), std::unique_ptr<column>> operator()(
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
      make_fixed_width_column(result_type,
                              num_groups,
                              values.has_nulls() ? mask_state::ALL_NULL : mask_state::UNALLOCATED,
                              stream,
                              mr);

    if (values.is_empty()) { return result; }

    auto resultview = mutable_column_device_view::create(result->mutable_view(), stream);
    auto valuesview = column_device_view::create(values, stream);
    if constexpr (K == aggregation::ARGMAX || K == aggregation::ARGMIN) {
      constexpr auto SENTINEL =
        (K == aggregation::ARGMAX ? cudf::detail::ARGMAX_SENTINEL : cudf::detail::ARGMIN_SENTINEL);
      auto idx_begin =
        cudf::detail::make_counting_transform_iterator(0, null_as_sentinel{*valuesview, SENTINEL});
      // dictionary keys are sorted, so dictionary32 index comparison is enough.
      auto column_begin = valuesview->begin<DeviceType>();
      auto begin        = thrust::make_zip_iterator(thrust::make_tuple(column_begin, idx_begin));
      auto result_begin = make_output_writer_iterator(thrust::make_counting_iterator<size_type>(0),
                                                      tuple_index_to_column{*resultview});
      using OpType =
        std::conditional_t<(K == aggregation::ARGMAX), ArgMax<DeviceType>, ArgMin<DeviceType>>;
      thrust::reduce_by_key(rmm::exec_policy(stream),
                            group_labels.data(),
                            group_labels.data() + group_labels.size(),
                            begin,
                            thrust::make_discard_iterator(),
                            result_begin,
                            thrust::equal_to<size_type>{},
                            OpType{});
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
      auto result_valid = make_output_writer_iterator(thrust::make_counting_iterator<size_type>(0),
                                                      bool_to_nullmask{*resultview});
      thrust::reduce_by_key(rmm::exec_policy(stream),
                            group_labels.data(),
                            group_labels.data() + group_labels.size(),
                            cudf::detail::make_validity_iterator(*valuesview),
                            thrust::make_discard_iterator(),
                            result_valid,
                            thrust::equal_to<size_type>{},
                            thrust::logical_or<bool>{});
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
