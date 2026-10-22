/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/buffer>
#include <cuda/stream>

#include <memory>

namespace cudf::groupby::detail {
namespace hash {
class grouped_keys;
struct grouped_rows;
}  // namespace hash

/**
 * @brief Helper class for computing grouped operations
 *
 * This class builds and memoizes HashCSR groups and provides
 * building blocks for aggregations. It can provide:
 * 1. On-demand grouping or sorting of a value column based on `keys`
 *   which is provided at construction
 * 2. Group offsets: starting offsets of all groups in grouped key table
 * 3. Group rows: original row indices and scheduling metadata for direct reductions
 *
 * A request that the memoized grouping cannot serve, grouped rows or their input order within
 * each group, rebuilds it. A rebuild may change the group order and invalidates all spans and
 * references returned before, so every operation requests the grouping it needs before taking
 * any of them. A stable grouping serves every request and is never rebuilt.
 */
struct groupby_helper {
  /**
   * @brief Construct a new helper object
   *
   * If `include_null_keys == NO`, then any row in `keys` containing a null
   * value will effectively be discarded. I.e., any values corresponding to
   * discarded rows in `keys` will not contribute to any aggregation.
   *
   * @param keys table to group by
   * @param include_null_keys Include rows in keys with nulls
   * @param keys_pre_sorted Indicate if the keys are already sorted. Enables
   *                        grouping by adjacent equality without hashing.
   */
  groupby_helper(table_view const& keys, null_policy include_null_keys, sorted keys_pre_sorted);

  ~groupby_helper();
  groupby_helper(groupby_helper const&)            = delete;
  groupby_helper& operator=(groupby_helper const&) = delete;
  groupby_helper(groupby_helper&&) noexcept;
  groupby_helper& operator=(groupby_helper&&) noexcept;

  /**
   * @brief Groups a column of values according to `keys` and sorts within each
   *  group.
   *
   * Groups the @p values where the groups are dictated by key table and each
   * group is sorted in ascending order, with NULL elements positioned at the
   * end of each group.
   *
   * @throw cudf::logic_error if `values.size() != keys.num_rows()`
   *
   * @param values The value column to group and sort
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned device memory
   * @return the sorted and grouped column
   */
  std::unique_ptr<column> sorted_values(column_view const& values,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr);

  /**
   * @brief Groups a column of values according to `keys`
   *
   * The values within each group maintain their original order.
   *
   * @throw cudf::logic_error if `values.size() != keys.num_rows()`
   *
   * @param values The value column to group
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned device memory
   * @return the grouped column
   */
  std::unique_ptr<column> grouped_values(column_view const& values,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

  /**
   * @brief Groups values without requiring their original order within each group.
   *
   * Groups have the same offsets and labels as `grouped_values`. The returned column owns its
   * gathered values, so subsequently requesting a stable grouped order does not change it.
   *
   * @param values The value column to group
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned device memory
   * @return the grouped column with unspecified order within each group
   */
  std::unique_ptr<column> unordered_grouped_values(column_view const& values,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr);

  /**
   * @brief Get a table of unique keys
   *
   * @return a new table in which each row is a unique row in the grouped key table.
   */
  std::unique_ptr<table> unique_keys(cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  /**
   * @brief Get a table of grouped keys
   *
   * @return a new table containing the grouped keys.
   */
  std::unique_ptr<table> grouped_keys(cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  /**
   * @brief Get the number of groups in `keys`
   */
  size_type num_groups(cuda::stream_ref stream) { return group_offsets(stream).size() - 1; }

  /**
   * @brief check if the groupby keys are presorted
   */
  bool is_presorted() { return _keys_pre_sorted == sorted::YES; }

  /**
   * @brief Check whether the cached grouped rows already retain input order within each group
   */
  [[nodiscard]] bool is_stable() const { return _stable; }

  /**
   * @brief Return the effective number of keys
   *
   * When include_null_keys = YES, returned value is same as `keys.num_rows()`
   * When include_null_keys = NO, returned value is the number of rows in `keys`
   *  in which no element is null
   */
  size_type num_keys(cuda::stream_ref stream);

  /**
   * @brief Get the grouped order of `keys`, retaining input order within each group.
   *
   * Gathering `keys` by these indices produces contiguous groups.
   *
   * When ignore_null_keys = true, the result will not include indices
   * for null keys.
   *
   * Computes and stores a stable grouped order on first invocation, and returns
   * the stored order on subsequent calls. Pass `keep_labels` when `group_labels`
   * will be requested too, so the labels sorted alongside the rows are kept.
   *
   * @return the grouped row indices for `keys`.
   */
  column_view grouped_order(bool keep_labels, cuda::stream_ref stream);

  /**
   * @brief Get grouped row indices without requiring input order within each group.
   */
  device_span<size_type const> unordered_grouped_order(cuda::stream_ref stream);

  /**
   * @brief Get cached row indices, offsets, and scheduling arrays for direct reductions.
   */
  hash::grouped_rows const& reduction_groups(cuda::stream_ref stream);

  /**
   * @brief Get each group's offset into the grouped order of `keys`.
   *
   * Computes and stores the group offsets on first invocation and returns
   * the stored group offsets on subsequent calls.
   * This returns a span of size `num_groups() + 1` such that the size of
   * group `i` is `group_offsets[i+1] - group_offsets[i]`
   *
   * @return span of offsets of the starting point of each group in the grouped
   * key table
   */
  device_span<size_type const> group_offsets(cuda::stream_ref stream);

  /**
   * @brief Get the group labels corresponding to the grouped order of `keys`.
   *
   * Each group is assigned a unique numerical "label" in
   * `[0, 1, 2, ... , num_groups() - 1, num_groups(stream))`.
   * For a row in grouped `keys`, its corresponding group label indicates which
   * group it belongs to.
   *
   * Computes and stores labels on first invocation and returns stored labels on
   * subsequent calls.
   *
   * @return span of group labels for each row in the grouped key column
   */
  device_span<size_type const> group_labels(cuda::stream_ref stream);

 private:
  /**
   * @brief Get the group label of every row of the ungrouped `keys`
   *
   * For an included row the label equals its label in `group_labels(stream)`; a
   * row excluded by `include_null_keys == NO` gets `num_groups(stream)`, which
   * sorts after every group.
   *
   * Computes and stores the labels on first invocation and returns the stored
   * labels on subsequent calls.
   *
   * @return span of group labels in the order of the ungrouped key table
   */
  device_span<size_type const> input_labels(cuda::stream_ref stream);

  /// Build the grouping unless the memoized one already has the requested grouped rows and order.
  /// A stable build keeps the group labels sorted alongside the rows when requested.
  void build_groups(bool need_grouped_rows,
                    bool stable_rows,
                    bool keep_labels,
                    cuda::stream_ref stream);

  std::unique_ptr<cuda::device_buffer<size_type>>
    _input_labels;   ///< Labels in input order; excluded rows get num_groups
  table_view _keys;  ///< Input grouping keys
  std::unique_ptr<hash::grouped_keys> _groups;                    ///< HashCSR grouping metadata
  std::unique_ptr<hash::grouped_rows> _reduction_groups;          ///< Cached reduction scheduling
  std::unique_ptr<cuda::device_buffer<size_type>> _group_labels;  ///< Labels in grouped order
  size_type _num_keys{};           ///< Number of grouped rows, set when they are built
  sorted _keys_pre_sorted;         ///< Whether _keys are pre-sorted
  null_policy _include_null_keys;  ///< Whether to use rows with nulls in _keys for grouping
  bool _stable{false};             ///< Whether rows within each group are in input order
};

}  // namespace cudf::groupby::detail
