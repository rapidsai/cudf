/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <utility>

namespace cudf::detail {

constexpr int DEFAULT_JOIN_BLOCK_SIZE = 128;

// Convenient alias for a pair of unique pointers to device uvectors.
using VectorPair = std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                             std::unique_ptr<rmm::device_uvector<size_type>>>;

/**
 * @brief Computes the trivial left join operation for the case when the
 * right table is empty.
 *
 * In this case all the valid indices of the left table
 * are returned with their corresponding right indices being set to
 * `JoinNoMatch`, i.e. `cuda::std::numeric_limits<size_type>::min()`.
 *
 * @param left Table of left columns to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the result
 *
 * @return Join output indices vector pair
 */
VectorPair get_trivial_left_join_indices(table_view const& left,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @brief Finalize a full-join result from a single probe-side `(left, right)` index pair.
 *
 * Takes ownership of `probe_indices`, resizes both vectors to `probe_indices.first->size() +
 * build_table_num_rows`, and appends the complement (unmatched build rows paired with
 * `JoinNoMatch`) into the tail. The vectors are then resized down to the true output length.
 *
 * Used by the non-partitioned full-join paths (hash/mixed/conditional); consuming the caller's
 * buffers in-place avoids a redundant concat memcpy over the probe data.
 *
 * @param probe_indices Probe-side `(left, right)` index vectors (consumed).
 * @param probe_table_num_rows Number of rows in the probe table (0 → every build row is
 *                             unmatched, fast path).
 * @param build_table_num_rows Number of rows in the build table.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate working storage.
 *
 * @return `[left_indices, right_indices]` of the complete full-join output.
 */
VectorPair finalize_full_join(VectorPair&& probe_indices,
                              size_type probe_table_num_rows,
                              size_type build_table_num_rows,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

/**
 * @brief Finalize a full-join result from per-partition probe index spans.
 *
 * Concatenates every `(left_partials[i], right_partials[i])` pair into the head of the output
 * and appends the complement (unmatched build rows paired with `JoinNoMatch`) into the tail.
 * Internally delegates to the `VectorPair&&` overload, so the mark/compact path is shared.
 *
 * Used by `cudf::hash_join::full_join_finalize` for partitioned full joins where the partials
 * live in separate buffers and must be gathered.
 *
 * @param left_partials Per-partition probe-side (left) index spans.
 * @param right_partials Per-partition probe-side (right) index spans.
 * @param probe_table_num_rows Number of rows in the probe table.
 * @param build_table_num_rows Number of rows in the build table.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned vectors.
 *
 * @return `[left_indices, right_indices]` sized `sum(left_partials[i].size()) + num_unmatched`.
 */
VectorPair finalize_full_join(
  cudf::host_span<cudf::device_span<size_type const> const> left_partials,
  cudf::host_span<cudf::device_span<size_type const> const> right_partials,
  size_type probe_table_num_rows,
  size_type build_table_num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
