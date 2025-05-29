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

#include <cudf/hashing.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <utility>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 * @file
 */

namespace detail {
/**
 * @brief Forward declaration for our distinct hash join
 */
class distinct_hash_join;
}  // namespace detail

/**
 * @brief Distinct hash join that builds hash table in creation and probes results in subsequent
 * `*_join` member functions
 *
 * This class enables the distinct hash join scheme that builds hash table once, and probes as many
 * times as needed (possibly in parallel).
 *
 * @note Behavior is undefined if the build table contains duplicates.
 * @note All NaNs are considered as equal
 */
class distinct_hash_join {
 public:
  distinct_hash_join() = delete;
  ~distinct_hash_join();
  distinct_hash_join(distinct_hash_join const&)            = delete;
  distinct_hash_join(distinct_hash_join&&)                 = delete;
  distinct_hash_join& operator=(distinct_hash_join const&) = delete;
  distinct_hash_join& operator=(distinct_hash_join&&)      = delete;

  /**
   * @brief Constructs a distinct hash join object for subsequent probe calls
   *
   * @throw cudf::logic_error if the build table has no columns
   * @throw std::invalid_argument if load_factor is not greater than 0 and less than or equal to 1
   *
   * @param build The build table that contains distinct elements
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param load_factor The desired ratio of filled slots to total slots in the hash table, must be
   * in range (0,1]. For example, 0.5 indicates a target of 50% occupancy. Note that the actual
   * occupancy achieved may be slightly lower than the specified value.
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  distinct_hash_join(cudf::table_view const& build,
                     null_equality compare_nulls  = null_equality::EQUAL,
                     double load_factor           = 0.5,
                     rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * an inner join between two tables. @see cudf::inner_join().
   *
   * @param probe The probe table, from which the keys are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned indices' device memory.
   *
   * @return A pair of columns [`probe_indices`, `build_indices`] that can be used to
   * construct the result of performing an inner join between two tables
   * with `build` and `probe` as the join keys.
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(cudf::table_view const& probe,
             rmm::cuda_stream_view stream      = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns the build table indices that can be used to construct the result of performing
   * a left join between two tables.
   *
   * @note For a given row index `i` of the probe table, the resulting `build_indices[i]` contains
   * the row index of the matched row from the build table if there is a match. Otherwise, contains
   * `JoinNoneValue`.
   *
   * @param probe The probe table, from which the keys are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A `build_indices` column that can be used to construct the result of
   * performing a left join between two tables with `build` and `probe` as the join
   * keys.
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> left_join(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  using impl_type = cudf::detail::distinct_hash_join;  ///< Implementation type

  std::unique_ptr<impl_type> _impl;  ///< Distinct hash join implementation
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
