/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "common_sort_impl.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cudf {
namespace detail {

/**
 * @brief Sort indices of a single column.
 *
 * This API offers fast sorting for primitive types. It cannot handle nested types and will not
 * consider `NaN` as equivalent to other `NaN`.
 *
 * @tparam method Whether to use stable sort
 * @param input Column to sort. The column data is not modified.
 * @param column_order Ascending or descending sort order
 * @param null_precedence How null rows are to be ordered
 * @param stable True if sort should be stable
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Sorted indices for the input column.
 */
template <sort_method method>
std::unique_ptr<column> sorted_order(column_view const& input,
                                     order column_order,
                                     null_order null_precedence,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @brief Comparator functor needed for single column sort.
 *
 * @tparam Column element type.
 */
template <typename T>
struct simple_comparator {
  __device__ bool operator()(size_type lhs, size_type rhs)
  {
    if (has_nulls) {
      bool lhs_null{d_column.is_null(lhs)};
      bool rhs_null{d_column.is_null(rhs)};
      if (lhs_null || rhs_null) {
        if (!ascending) thrust::swap(lhs_null, rhs_null);
        return (null_precedence == cudf::null_order::BEFORE ? !rhs_null : !lhs_null);
      }
    }
    return relational_compare(d_column.element<T>(lhs), d_column.element<T>(rhs)) ==
           (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
  }
  column_device_view const d_column;
  bool has_nulls;
  bool ascending;
  null_order null_precedence{};
};

template <sort_method method>
struct column_sorted_order_fn {
  /**
   * @brief Compile time check for allowing faster sort.
   *
   * Faster sort is defined for fixed-width types where only
   * the primitive comparators thrust::greater or thrust::less
   * are needed.
   *
   * Floating point is removed here for special handling of NaNs
   * which require the row-comparator.
   */
  template <typename T>
  static constexpr bool is_faster_sort_supported()
  {
    return cudf::is_fixed_width<T>() && !cudf::is_floating_point<T>();
  }

  /**
   * @brief Sorts fixed-width columns using faster thrust sort.
   *
   * Should not be called if `input.has_nulls()==true`
   *
   * @param input Column to sort
   * @param indices Output sorted indices
   * @param ascending True if sort order is ascending
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  template <typename T>
  void faster_sort(column_view const& input,
                   mutable_column_view& indices,
                   bool ascending,
                   rmm::cuda_stream_view stream)
  {
    // A thrust sort on a column of primitive types will use a radix sort.
    // For other fixed-width types, thrust will use merge-sort.
    // But this also requires making a copy of the input data.
    auto temp_col = column(input, stream);
    auto d_col    = temp_col.mutable_view();

    auto const do_sort = [&](auto const comp) {
      // Compiling `thrust::*sort*` APIs is expensive.
      // Thus, we should optimize that by using constexpr condition to only compile what we need.
      if constexpr (method == sort_method::STABLE) {
        thrust::stable_sort_by_key(rmm::exec_policy(stream),
                                   d_col.begin<T>(),
                                   d_col.end<T>(),
                                   indices.begin<size_type>(),
                                   comp);
      } else {
        thrust::sort_by_key(rmm::exec_policy(stream),
                            d_col.begin<T>(),
                            d_col.end<T>(),
                            indices.begin<size_type>(),
                            comp);
      }
    };

    if (ascending) {
      do_sort(thrust::less<T>{});
    } else {
      do_sort(thrust::greater<T>{});
    }
  }

  /**
   * @brief Sorts a single column with a relationally comparable type.
   *
   * This is used when a comparator is required.
   *
   * @param input Column to sort
   * @param indices Output sorted indices
   * @param ascending True if sort order is ascending
   * @param null_precedence How null rows are to be ordered
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  template <typename T>
  void sorted_order(column_view const& input,
                    mutable_column_view& indices,
                    bool ascending,
                    null_order null_precedence,
                    rmm::cuda_stream_view stream)
  {
    auto keys = column_device_view::create(input, stream);
    auto comp = simple_comparator<T>{*keys, input.has_nulls(), ascending, null_precedence};
    // Compiling `thrust::*sort*` APIs is expensive.
    // Thus, we should optimize that by using constexpr condition to only compile what we need.
    if constexpr (method == sort_method::STABLE) {
      thrust::stable_sort(
        rmm::exec_policy(stream), indices.begin<size_type>(), indices.end<size_type>(), comp);
    } else {
      thrust::sort(
        rmm::exec_policy(stream), indices.begin<size_type>(), indices.end<size_type>(), comp);
    }
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_relationally_comparable<T, T>())>
  void operator()(column_view const& input,
                  mutable_column_view& indices,
                  bool ascending,
                  null_order null_precedence,
                  rmm::cuda_stream_view stream)
  {
    if constexpr (is_faster_sort_supported<T>()) {
      if (input.has_nulls()) {
        sorted_order<T>(input, indices, ascending, null_precedence, stream);
      } else {
        faster_sort<T>(input, indices, ascending, stream);
      }
    } else {
      sorted_order<T>(input, indices, ascending, null_precedence, stream);
    }
  }

  template <typename T, CUDF_ENABLE_IF(not cudf::is_relationally_comparable<T, T>())>
  void operator()(column_view const&, mutable_column_view&, bool, null_order, rmm::cuda_stream_view)
  {
    CUDF_FAIL("Column type must be relationally comparable");
  }
};

}  // namespace detail
}  // namespace cudf
