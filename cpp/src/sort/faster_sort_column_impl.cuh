/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sort.h>

namespace cudf {
namespace detail {

/**
 * @brief Sort indices of a single column.
 *
 * This API offers fast sorting for most primitive types.
 *
 * @tparam method Whether to use stable sort
 * @param input Column to sort. The column data is not modified.
 * @param ascending Sort order
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Sorted indices for the input column.
 */
template <sort_method method>
void faster_sorted_order(column_view const& input,
                         mutable_column_view& indices,
                         bool ascending,
                         rmm::cuda_stream_view stream);

template <sort_method method>
struct faster_sorted_order_fn {
  /**
   * @brief Compile time check for allowing faster sort.
   *
   * Faster sort is defined for fixed-width types where only
   * the primitive comparators cuda::std::greater or cuda::std::less
   * are needed.
   *
   * Floating point is removed here for special handling of NaNs
   * which require the row-comparator.
   */
  template <typename T>
  static constexpr bool is_supported()
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
  void faster_sort(mutable_column_view& input,
                   mutable_column_view& indices,
                   bool ascending,
                   rmm::cuda_stream_view stream)
  {
    // A thrust sort on a column of most primitive types will use a radix sort.
    // For other fixed-width types, thrust may use merge-sort.
    // The API sorts inplace so it requires making a copy of the input data
    // and creating the input indices sequence.

    auto const do_sort = [&](auto const comp) {
      // Compiling `thrust::*sort*` APIs is expensive.
      // Thus, we should optimize that by using constexpr condition to only compile what we need.
      if constexpr (method == sort_method::STABLE) {
        thrust::stable_sort_by_key(rmm::exec_policy(stream),
                                   input.begin<T>(),
                                   input.end<T>(),
                                   indices.begin<size_type>(),
                                   comp);
      } else {
        thrust::sort_by_key(rmm::exec_policy(stream),
                            input.begin<T>(),
                            input.end<T>(),
                            indices.begin<size_type>(),
                            comp);
      }
    };

    if (ascending) {
      do_sort(cuda::std::less<T>{});
    } else {
      do_sort(cuda::std::greater<T>{});
    }
  }

  template <typename T, CUDF_ENABLE_IF(is_supported<T>())>
  void operator()(mutable_column_view& input,
                  mutable_column_view& indices,
                  bool ascending,
                  rmm::cuda_stream_view stream)
  {
    faster_sort<T>(input, indices, ascending, stream);
  }

  template <typename T, CUDF_ENABLE_IF(not is_supported<T>())>
  void operator()(mutable_column_view&, mutable_column_view&, bool, rmm::cuda_stream_view)
  {
    CUDF_UNREACHABLE("invalid type for faster sort");
  }
};

}  // namespace detail
}  // namespace cudf
