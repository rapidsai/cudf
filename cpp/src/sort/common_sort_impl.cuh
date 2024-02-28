/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sort.h>

namespace cudf {
namespace detail {

/**
 * @brief The enum specifying which sorting method to use (stable or unstable).
 */
enum class sort_method : bool { STABLE, UNSTABLE };

/**
 * @brief Functor performs a fast-path, in-place sort on eligible columns
 *
 * @tparam method Whether to use a stable or unstable sort.
 */
template <sort_method method>
struct inplace_column_sort_fn {
  /**
   * @brief Check if fast-path, in-place sort is available for the given column
   *
   * @param column to check
   * @return true if fast-path sort is available, false otherwise.
   */
  static bool is_usable(column_view const& column)
  {
    return !column.has_nulls() && cudf::is_fixed_width(column.type()) &&
           !cudf::is_floating_point(column.type());
  }
  /**
   * @brief Check if fast-path, in-place sort is available for the given table
   *
   * @param table to check
   * @return true if fast-path sort is available, false otherwise.
   */
  static bool is_usable(table_view const& table)
  {
    return table.num_columns() == 1 && is_usable(table.column(0));
  }

  /**
   * @brief Fast-path sort a column in place
   *
   * Precondition, is_usable(column) returned true
   *
   * @tparam T column data type.
   * @param col Column to sort, modified in place.
   * @param order Ascending or descending sort order.
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   */
  template <typename T, std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
  void operator()(mutable_column_view& col, order order, rmm::cuda_stream_view stream) const
  {
    auto const do_sort = [&](auto const cmp) {
      if constexpr (method == sort_method::STABLE) {
        thrust::stable_sort(rmm::exec_policy(stream), col.begin<T>(), col.end<T>(), cmp);
      } else {
        thrust::sort(rmm::exec_policy(stream), col.begin<T>(), col.end<T>(), cmp);
      }
    };
    if (order == order::ASCENDING) {
      do_sort(thrust::less<T>());
    } else {
      do_sort(thrust::greater<T>());
    }
  }

  template <typename T, std::enable_if_t<!cudf::is_fixed_width<T>()>* = nullptr>
  void operator()(mutable_column_view&, order, rmm::cuda_stream_view) const
  {
    CUDF_FAIL("Column type must be relationally comparable and fixed-width");
  }
};

}  // namespace detail
}  // namespace cudf
