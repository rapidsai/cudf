/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/bucket_storage.cuh>
#include <cuco/extent.cuh>
#include <cuco/static_set_ref.cuh>
#include <cuco/types.cuh>
#include <cuda/std/type_traits>

namespace cudf {
namespace detail {

using cudf::detail::row::lhs_index_type;
using cudf::detail::row::rhs_index_type;

/**
 * @brief Base class providing common functionality for filtered join operations.
 *
 * This abstract class implements the core components needed for hash-based semi
 * and anti join operations.
 */
class filtered_join {
 public:
  /**
   * @brief Properties of the build table used in the join operation
   */
  struct build_properties {
    bool has_nested_columns;  ///< True if the build table contains nested columns
  };

  /**
   * @brief Adapter for insertion operations in the hash table
   *
   * Returns result of self comparator passed
   */
  template <typename T>
  struct insertion_adapter {
    insertion_adapter(T const& _c) : _comparator{_c} {}
    __device__ constexpr bool operator()(
      cuco::pair<hash_value_type, lhs_index_type> const& lhs,
      cuco::pair<hash_value_type, lhs_index_type> const& rhs) const noexcept
    {
      if (lhs.first != rhs.first) { return false; }
      auto const lhs_index = static_cast<size_type>(lhs.second);
      auto const rhs_index = static_cast<size_type>(rhs.second);
      return _comparator(lhs_index, rhs_index);
    }

   private:
    T _comparator;
  };

  /**
   * @brief Adapter for extracting hash values from key-value pairs
   */
  struct hash_extract_fn {
    template <typename T>
    __device__ constexpr hash_value_type operator()(
      cuco::pair<hash_value_type, T> const& key) const noexcept
    {
      return key.first;
    }
  };

  /**
   * @brief Adapter for generating key-value pairs from indices
   *
   * @tparam T Index type
   * @tparam Hasher Hash function type
   */
  template <typename T, typename Hasher>
  struct key_pair_fn {
    CUDF_HOST_DEVICE constexpr key_pair_fn(Hasher const& hasher) : _hasher{hasher} {}

    __device__ __forceinline__ auto operator()(size_type i) const noexcept
    {
      return cuco::pair{_hasher(i), T{i}};
    }

   private:
    Hasher _hasher;
  };

  /**
   * @brief Adapter for comparing key-value pairs
   *
   * Compares hash values first for performance, then uses the provided equality comparator
   *
   * @tparam Equal Equality comparator type
   */
  template <typename Equal>
  struct comparator_adapter {
    comparator_adapter(Equal const& d_equal) : _d_equal{d_equal} {}

    __device__ constexpr auto operator()(
      cuco::pair<hash_value_type, rhs_index_type> const& rhs,
      cuco::pair<hash_value_type, lhs_index_type> const& lhs) const noexcept
    {
      if (lhs.first != rhs.first) { return false; }
      return _d_equal(lhs.second, rhs.second);
    }

   private:
    Equal _d_equal;
  };

  /**
   * @brief Constructor for filtered_join base class
   *
   * Initializes the hash table with the build table and prepares it for join operations.
   *
   * @param build The table to build the hash table from
   * @param compare_nulls How null values should be compared
   * @param load_factor Target load factor for the hash table
   * @param stream CUDA stream on which to perform operations
   */
  filtered_join(cudf::table_view const& build,
                cudf::null_equality compare_nulls,
                double load_factor,
                rmm::cuda_stream_view stream);

  /**
   * Virtual semi join function overridden in derived classes
   */
  virtual std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_join(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) = 0;

  /**
   * Virtual anti join function overridden in derived classes
   */
  virtual std::unique_ptr<rmm::device_uvector<cudf::size_type>> anti_join(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) = 0;

  /**
   * Virtual abstract base class destructor
   */
  virtual ~filtered_join() = default;

 protected:
  // Key type used in the hash table
  using key = cuco::pair<hash_value_type, lhs_index_type>;

  // Storage type for the hash table buckets
  using storage_type =
    cuco::bucket_storage<key,
                         1,  /// fixing bucket size to be 1 i.e each thread handles one slot
                         cuco::extent<std::size_t>,
                         rmm::mr::polymorphic_allocator<char>>;

  // Hasher for primitive row types
  using primitive_row_hasher =
    cudf::detail::row::primitive::row_hasher<cudf::hashing::detail::default_hash>;
  // Linear probing scheme with bucket size 1 for primitive types
  using primitive_probing_scheme = cuco::linear_probing<1, hash_extract_fn>;
  // Equality comparator for primitive rows
  using primitive_row_comparator = cudf::detail::row::primitive::row_equality_comparator;

  // Hasher for complex row types with compile-time null handling
  using row_hasher =
    cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash, nullate::YES>;
  // Linear probing scheme with bucket size 4 for nested data structures
  using nested_probing_scheme = cuco::linear_probing<4, hash_extract_fn>;
  // Linear probing scheme with bucket size 1 for simple data
  using simple_probing_scheme = cuco::linear_probing<1, hash_extract_fn>;
  // Equality comparator for complex rows with null handling and NaN comparison
  using row_comparator = cudf::detail::row::equality::device_row_comparator<
    true,
    cudf::nullate::YES,
    cudf::detail::row::equality::nan_equal_physical_equality_comparator>;

  storage_type _bucket_storage;  ///< Storage for hash table buckets

  // Empty sentinel key used to mark empty slots in the hash table
  static constexpr auto empty_sentinel_key = cuco::empty_key{
    cuco::pair{std::numeric_limits<hash_value_type>::max(), lhs_index_type{cudf::JoinNoMatch}}};
  build_properties _build_props;           ///< Properties of the build table
  cudf::table_view _build;                 ///< input table to build the hash map
  cudf::null_equality const _nulls_equal;  ///< whether to consider nulls as equal
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table>
    _preprocessed_build;  ///< input table preprocssed for row operators

  /**
   * @brief Populates the hash table with the build table
   *
   * @tparam CGSize CUDA cooperative group size
   * @tparam Ref Reference type for the hash table
   * @param insert_ref Reference to the hash table for insertion
   * @param stream CUDA stream on which to perform operations
   */
  template <int32_t CGSize, typename Ref>
  void insert_build_table(Ref const& insert_ref, rmm::cuda_stream_view stream);

 private:
  /**
   * @brief Calculates the required storage size for the hash table
   *
   * Computes the appropriate size for the bucket storage based on the input
   * table size and desired load factor.
   *
   * @param tbl Table for which to calculate storage
   * @param load_factor Target load factor for the hash table
   * @return Calculated bucket storage size
   */
  auto compute_bucket_storage_size(cudf::table_view tbl, double load_factor);
};

}  // namespace detail
}  // namespace cudf
