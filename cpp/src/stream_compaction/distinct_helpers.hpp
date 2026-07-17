/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include <cstdint>
#include <limits>

namespace cudf::detail {

struct distinct_physical_equality {
  nan_equality compare_nans;

  template <typename Element>
  __device__ constexpr bool operator()(Element const lhs, Element const rhs) const noexcept
  {
    if constexpr (cuda::std::is_floating_point_v<Element>) {
      return lhs == rhs || (compare_nans == nan_equality::ALL_EQUAL && cuda::std::isnan(lhs) &&
                            cuda::std::isnan(rhs));
    } else {
      return lhs == rhs;
    }
  }
};

/**
 * @brief Return the value used to initialize or mark reduction results.
 *
 * @param keep A value of `duplicate_keep_option` type
 * @return The reduction value
 */
auto constexpr reduction_init_value(duplicate_keep_option keep)
{
  switch (keep) {
    case duplicate_keep_option::KEEP_FIRST: return std::numeric_limits<size_type>::max();
    case duplicate_keep_option::KEEP_LAST: return std::numeric_limits<size_type>::min();
    case duplicate_keep_option::KEEP_NONE: return size_type{0};
    case duplicate_keep_option::KEEP_ANY: return CUDF_SIZE_TYPE_SENTINEL;
    default: CUDF_UNREACHABLE("Invalid duplicate keep option");
  }
}

void initialize_reduction_results(size_type* results,
                                  size_type num_rows,
                                  duplicate_keep_option keep,
                                  rmm::cuda_stream_view stream);

size_type copy_reduction_results(size_type const* results,
                                 size_type num_rows,
                                 size_type* output,
                                 duplicate_keep_option keep,
                                 rmm::cuda_stream_view stream);

template <typename RowEqual>
using distinct_set_t =
  cuco::static_set<size_type,
                   cuco::extent<int64_t>,
                   cuda::thread_scope_device,
                   RowEqual,
                   cuco::linear_probing<
                     1,
                     cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                                cudf::nullate::DYNAMIC>>,
                   rmm::mr::polymorphic_allocator<char>,
                   cuco::storage<1>>;

/**
 * @brief Perform a reduction on groups of rows that are compared equal and returns output indices
 * of the occurrences of the distinct elements based on `keep` parameter.
 *
 * This is essentially a reduce-by-key operation with keys are non-contiguous rows and are compared
 * equal. A hash set is used to find groups of equal rows.
 *
 * Depending on the `keep` parameter, the reduction operation for each row group is:
 * - If `keep == KEEP_ANY`: retain the row inserted into the set.
 * - If `keep == KEEP_FIRST`: min of row indices in the group.
 * - If `keep == KEEP_LAST`: max of row indices in the group.
 * - If `keep == KEEP_NONE`: count of equivalent rows (group size).
 *
 * Except for `KEEP_ANY`, the result array is initialized with `reduction_init_value()`, then each
 * row group writes its reduction at the index of an unspecified row in the group. For `KEEP_ANY`,
 * each row stores either its index or the sentinel value, depending on whether it was inserted.
 *
 * @tparam RowEqual The type of row equality comparator
 *
 * @param set The auxiliary set to perform reduction
 * @param num_rows The number of all input rows
 * @param keep The parameter to determine what type of reduction to perform
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return A device_uvector containing the output indices
 */
template <typename RowEqual>
rmm::device_uvector<size_type> reduce_by_row(distinct_set_t<RowEqual>& set,
                                             size_type num_rows,
                                             duplicate_keep_option keep,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);
}  // namespace cudf::detail
