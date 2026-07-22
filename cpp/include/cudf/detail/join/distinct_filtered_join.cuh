/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>

namespace cudf::detail::row::equality {
class preprocessed_table;
}

namespace cudf {

// Forward declaration
enum class join_kind : int32_t;

namespace detail {

/**
 * @brief Implementation of filtered join using set hash tables
 *
 * This class extends the base filtered_join to implement join operations
 * using set semantics, where duplicate keys are not allowed in the hash table.
 * This implementation is more memory efficient when the same filter table (right table)
 * is to be reused for multiple semi/anti join operations.
 */
class distinct_filtered_join : public filtered_join {
 private:
  /**
   * @brief Performs either a semi or anti join based on the specified kind
   *
   * @param left The left table to probe the hash table with
   * @param kind The kind of join to perform (SEMI or ANTI)
   * @param stream CUDA stream on which to perform operations
   * @param mr Memory resource for allocations
   * @return Device vector of indices representing the join result
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_anti_join(
    cudf::table_view const& left,
    join_kind kind,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  // Queries the hash table for every left row and writes the matches to contains_map.
  template <int32_t CGSize, typename Iterator, typename Ref>
  void query_right_table(cudf::table_view const& left,
                         Iterator left_iter,
                         Ref query_ref,
                         cudf::device_span<bool> contains_map,
                         rmm::cuda_stream_view stream);

  void query_right_table_primitive(
    cudf::table_view const& left,
    std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
    cudf::device_span<bool> contains_map,
    rmm::cuda_stream_view stream);

  void query_right_table_flat(
    cudf::table_view const& left,
    std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
    cudf::device_span<bool> contains_map,
    rmm::cuda_stream_view stream);

  void query_right_table_nested(
    cudf::table_view const& left,
    std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
    cudf::device_span<bool> contains_map,
    rmm::cuda_stream_view stream);

 public:
  /**
   * @brief Constructor for filtered join with set
   *
   * @param right The right table used to build the hash table
   * @param compare_nulls How null values should be compared
   * @param load_factor Target load factor for the hash table
   * @param stream CUDA stream on which to perform operations
   * @param mr Device memory resource used to allocate the internal hash table
   */
  distinct_filtered_join(cudf::table_view const& right,
                         cudf::null_equality compare_nulls,
                         double load_factor,
                         rmm::cuda_stream_view stream,
                         cuda::mr::any_resource<cuda::mr::device_accessible> mr);

  /**
   * @brief Implementation of semi join for set
   *
   * Returns indices of left table rows that have matching keys in the right table.
   *
   * @param left The left table to probe the hash table with
   * @param stream CUDA stream on which to perform operations
   * @param mr Memory resource for allocations
   * @return Device vector of indices representing the join result
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_join(
    cudf::table_view const& left,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) override;

  /**
   * @brief Implementation of anti join for set
   *
   * Returns indices of left table rows that do not have matching keys in the right table.
   *
   * @param left The left table to probe the hash table with
   * @param stream CUDA stream on which to perform operations
   * @param mr Memory resource for allocations
   * @return Device vector of indices representing the join result
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> anti_join(
    cudf::table_view const& left,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) override;
};

}  // namespace detail
}  // namespace cudf
