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

#include <sort/sort_impl.cuh>

#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cudf {
namespace detail {

template <bool stable>
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
    if (ascending) {
      if constexpr (stable) {
        thrust::stable_sort_by_key(rmm::exec_policy(stream),
                                   d_col.begin<T>(),
                                   d_col.end<T>(),
                                   indices.begin<size_type>(),
                                   thrust::less<T>());
      } else {
        thrust::sort_by_key(rmm::exec_policy(stream),
                            d_col.begin<T>(),
                            d_col.end<T>(),
                            indices.begin<size_type>(),
                            thrust::less<T>());
      }
    } else {
      if constexpr (stable) {
        thrust::stable_sort_by_key(rmm::exec_policy(stream),
                                   d_col.begin<T>(),
                                   d_col.end<T>(),
                                   indices.begin<size_type>(),
                                   thrust::greater<T>());
      } else {
        thrust::sort_by_key(rmm::exec_policy(stream),
                            d_col.begin<T>(),
                            d_col.end<T>(),
                            indices.begin<size_type>(),
                            thrust::greater<T>());
      }
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
    if constexpr (stable) {
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
