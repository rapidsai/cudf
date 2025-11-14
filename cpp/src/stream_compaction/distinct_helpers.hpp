/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cuco/static_set.cuh>
#include <cuda/functional>
#include <cuda/std/iterator>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf::detail {

/**
 * @brief Return the reduction identity used to initialize results of `hash_reduce_by_row`.
 *
 * @param keep A value of `duplicate_keep_option` type, must not be `KEEP_ANY`.
 * @return The initial reduction value.
 */
auto constexpr reduction_init_value(duplicate_keep_option keep)
{
  switch (keep) {
    case duplicate_keep_option::KEEP_FIRST: return std::numeric_limits<size_type>::max();
    case duplicate_keep_option::KEEP_LAST: return std::numeric_limits<size_type>::min();
    case duplicate_keep_option::KEEP_NONE: return size_type{0};
    default: CUDF_UNREACHABLE("This function should not be called with KEEP_ANY");
  }
}

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
 * - If `keep == KEEP_ANY` : order does not matter.
 * - If `keep == KEEP_FIRST`: min of row indices in the group.
 * - If `keep == KEEP_LAST`: max of row indices in the group.
 * - If `keep == KEEP_NONE`: count of equivalent rows (group size).
 *
 * Note that this function is not needed when `keep == KEEP_NONE`.
 *
 * At the beginning of the operation, the entire output array is filled with a value given by
 * the `reduction_init_value()` function. Then, the reduction result for each row group is written
 * into the output array at the index of an unspecified row in the group.
 *
 * @tparam RowEqual The type of row equality comparator
 *
 * @param set The auxiliary set to perform reduction
 * @param set_size The number of elements in set
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
