/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {

namespace detail {
class mark_join;
}  // namespace detail

/**
 * @brief Mark-based hash join for semi/anti join with left table reuse.
 *
 * Builds a hash table from the build (left) table using a multiset that allows
 * duplicate keys. The probe kernel atomically marks matching build entries via
 * CAS on the hash MSB, then a retrieve kernel collects marked (semi) or unmarked
 * (anti) entries.
 *
 * This class enables building the hash table once and probing multiple times.
 *
 * @note All NaNs are considered as equal
 */
class mark_join {
 public:
  mark_join() = delete;
  ~mark_join();
  mark_join(mark_join const&)            = delete;
  mark_join(mark_join&&)                 = delete;
  mark_join& operator=(mark_join const&) = delete;
  mark_join& operator=(mark_join&&)      = delete;

  /**
   * @brief Constructs a mark join object by building a hash table from the build table.
   *
   * @param build The build table (typically the left table)
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  mark_join(cudf::table_view const& build,
            cudf::null_equality compare_nulls = null_equality::EQUAL,
            rmm::cuda_stream_view stream      = cudf::get_default_stream());

  /**
   * @brief Constructs a mark join object with a specified load factor.
   *
   * @param build The build table (typically the left table)
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param load_factor Hash table load factor in range (0,1]
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  mark_join(cudf::table_view const& build,
            cudf::null_equality compare_nulls,
            double load_factor,
            rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Returns build row indices that have at least one match in the probe table.
   *
   * @param probe The probe table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned device memory
   * @return Device vector of build row indices
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> semi_join(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns build row indices that have no match in the probe table.
   *
   * @param probe The probe table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned device memory
   * @return Device vector of build row indices
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> anti_join(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  std::unique_ptr<cudf::detail::mark_join> _impl;
};

}  // namespace CUDF_EXPORT cudf
