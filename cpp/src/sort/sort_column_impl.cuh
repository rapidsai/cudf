/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sort.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_merge_sort.cuh>

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
        if (!ascending) { cuda::std::swap(lhs_null, rhs_null); }
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
    auto keys      = column_device_view::create(input, stream);
    auto comp      = simple_comparator<T>{*keys, input.has_nulls(), ascending, null_precedence};
    auto in_keys   = thrust::make_counting_iterator<cudf::size_type>(0);
    auto out_keys  = indices.begin<size_type>();
    auto tmp_bytes = std::size_t{0};
    if constexpr (method == sort_method::STABLE) {
      cub::DeviceMergeSort::StableSortKeysCopy(
        nullptr, tmp_bytes, in_keys, out_keys, indices.size(), comp, stream.value());
      auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
      cub::DeviceMergeSort::StableSortKeysCopy(
        tmp_stg.data(), tmp_bytes, in_keys, out_keys, indices.size(), comp, stream.value());
    } else {
      cub::DeviceMergeSort::SortKeysCopy(
        nullptr, tmp_bytes, in_keys, out_keys, indices.size(), comp, stream.value());
      auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
      cub::DeviceMergeSort::SortKeysCopy(
        tmp_stg.data(), tmp_bytes, in_keys, out_keys, indices.size(), comp, stream.value());
    }
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_relationally_comparable<T, T>())>
  void operator()(column_view const& input,
                  mutable_column_view& indices,
                  bool ascending,
                  null_order null_precedence,
                  rmm::cuda_stream_view stream)
  {
    sorted_order<T>(input, indices, ascending, null_precedence, stream);
  }

  template <typename T, CUDF_ENABLE_IF(not cudf::is_relationally_comparable<T, T>())>
  void operator()(column_view const&, mutable_column_view&, bool, null_order, rmm::cuda_stream_view)
  {
    CUDF_FAIL("Column type must be relationally comparable");
  }
};

}  // namespace detail
}  // namespace cudf
