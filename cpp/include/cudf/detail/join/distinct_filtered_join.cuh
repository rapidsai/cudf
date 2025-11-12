/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf {
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
   * @param probe The table to probe the hash table with
   * @param kind The kind of join to perform (SEMI or ANTI)
   * @param stream CUDA stream on which to perform operations
   * @param mr Memory resource for allocations
   * @return Device vector of indices representing the join result
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_anti_join(
    cudf::table_view const& probe,
    join_kind kind,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  /**
   * @brief Core implementation for querying the hash table
   *
   * Performs the actual hash table query operation for both semi and anti joins
   * using set semantics.
   *
   * @tparam CGSize CUDA cooperative group size
   * @tparam Ref Reference type for the hash table
   * @param probe The table to probe the hash table with
   * @param preprocessed_probe Preprocessed probe table for row operators
   * @param kind The kind of join to perform
   * @param query_ref Reference to the hash table for querying
   * @param stream CUDA stream on which to perform operations
   * @param mr Memory resource for allocations
   * @return Device vector of indices representing the join result
   */
  template <int32_t CGSize, typename Ref>
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> query_build_table(
    cudf::table_view const& probe,
    std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_probe,
    join_kind kind,
    Ref query_ref,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

 public:
  /**
   * @brief Constructor for filtered join with set
   *
   * @param build The table to build the hash table from
   * @param compare_nulls How null values should be compared
   * @param load_factor Target load factor for the hash table
   * @param stream CUDA stream on which to perform operations
   */
  distinct_filtered_join(cudf::table_view const& build,
                         cudf::null_equality compare_nulls,
                         double load_factor,
                         rmm::cuda_stream_view stream);

  /**
   * @brief Implementation of semi join for set
   *
   * Returns indices of probe table rows that have matching keys in the build table.
   *
   * @param probe The table to probe the hash table with
   * @param stream CUDA stream on which to perform operations
   * @param mr Memory resource for allocations
   * @return Device vector of indices representing the join result
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_join(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) override;

  /**
   * @brief Implementation of anti join for set
   *
   * Returns indices of probe table rows that do not have matching keys in the build table.
   *
   * @param probe The table to probe the hash table with
   * @param stream CUDA stream on which to perform operations
   * @param mr Memory resource for allocations
   * @return Device vector of indices representing the join result
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> anti_join(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) override;
};

}  // namespace detail
}  // namespace cudf
