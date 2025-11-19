/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/static_set.cuh>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace cudf::detail {

using cudf::detail::row::lhs_index_type;
using cudf::detail::row::rhs_index_type;

/**
 * @brief A custom comparator used for the build table insertion
 */
struct always_not_equal {
  __device__ constexpr bool operator()(
    cuco::pair<hash_value_type, rhs_index_type> const&,
    cuco::pair<hash_value_type, rhs_index_type> const&) const noexcept
  {
    // All build table keys are distinct thus `false` no matter what
    return false;
  }
};

/**
 * @brief A comparator adapter wrapping the two table comparator
 */
template <typename Equal>
struct comparator_adapter {
  comparator_adapter(Equal const& d_equal) : _d_equal{d_equal} {}

  __device__ constexpr auto operator()(
    cuco::pair<hash_value_type, lhs_index_type> const& lhs,
    cuco::pair<hash_value_type, rhs_index_type> const& rhs) const noexcept
  {
    if (lhs.first != rhs.first) { return false; }
    return _d_equal(lhs.second, rhs.second);
  }

 private:
  Equal _d_equal;
};

/**
 * @brief A comparator adapter wrapping the two table comparator
 */
template <typename Equal>
struct primitive_comparator_adapter {
  primitive_comparator_adapter(Equal const& d_equal) : _d_equal{d_equal} {}

  __device__ constexpr auto operator()(
    cuco::pair<hash_value_type, lhs_index_type> const& lhs,
    cuco::pair<hash_value_type, rhs_index_type> const& rhs) const noexcept
  {
    if (lhs.first != rhs.first) { return false; }
    return _d_equal(static_cast<size_type>(lhs.second), static_cast<size_type>(rhs.second));
  }

 private:
  Equal _d_equal;
};

/**
 * @brief Distinct hash join that builds hash table in creation and probes results in subsequent
 * `*_join` member functions.
 *
 * This class enables the distinct hash join scheme that builds hash table once, and probes as many
 * times as needed (possibly in parallel).
 */
class distinct_hash_join {
 public:
  distinct_hash_join()                                     = delete;
  ~distinct_hash_join()                                    = default;
  distinct_hash_join(distinct_hash_join const&)            = delete;
  distinct_hash_join(distinct_hash_join&&)                 = delete;
  distinct_hash_join& operator=(distinct_hash_join const&) = delete;
  distinct_hash_join& operator=(distinct_hash_join&&)      = delete;

  /**
   * @brief Hasher adapter used by distinct hash join
   */
  struct hasher {
    template <typename T>
    __device__ constexpr hash_value_type operator()(
      cuco::pair<hash_value_type, T> const& key) const noexcept
    {
      return key.first;
    }
  };

  /**
   * @brief Constructor that internally builds the hash table based on the given `build` table.
   *
   * @throw cudf::logic_error if the number of columns in `build` table is 0.
   *
   * @param build The build table, from which the hash table is built
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  distinct_hash_join(cudf::table_view const& build,
                     cudf::null_equality compare_nulls,
                     rmm::cuda_stream_view stream);

  /**
   * @copydoc distinct_hash_join(cudf::table_view const&, null_equality, rmm::cuda_stream_view)
   *
   * @param load_factor The hash table occupancy ratio in (0,1]. A value of 0.5 means 50% occupancy.
   */
  distinct_hash_join(cudf::table_view const& build,
                     cudf::null_equality compare_nulls,
                     double load_factor,
                     rmm::cuda_stream_view stream);

  /**
   * @copydoc cudf::distinct_hash_join::inner_join
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(cudf::table_view const& probe,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref mr) const;

  /**
   * @copydoc cudf::distinct_hash_join::left_join
   */
  std::unique_ptr<rmm::device_uvector<size_type>> left_join(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

 private:
  using probing_scheme_type = cuco::linear_probing<1, hasher>;
  using cuco_storage_type   = cuco::storage<1>;

  /// Hash table type
  using hash_table_type = cuco::static_set<cuco::pair<hash_value_type, rhs_index_type>,
                                           cuco::extent<std::size_t>,
                                           cuda::thread_scope_device,
                                           always_not_equal,
                                           probing_scheme_type,
                                           rmm::mr::polymorphic_allocator<char>,
                                           cuco_storage_type>;

  bool _has_nested_columns;  ///< True if nested columns are present in build and probe tables
  cudf::null_equality _nulls_equal;  ///< Whether to consider nulls as equal
  cudf::table_view _build;           ///< Input table to build the hash map
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table>
    _preprocessed_build;        ///< Input table preprocssed for row operators
  hash_table_type _hash_table;  ///< Hash table built on `_build`
};
}  // namespace cudf::detail
