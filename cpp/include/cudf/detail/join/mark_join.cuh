/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/hash_functions.cuh>
#include <cuco/pair.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/types.cuh>
#include <cuda/std/limits>

#include <memory>

namespace cudf::detail {

static constexpr hash_value_type mark_bit_mask = hash_value_type{1}
                                                 << (sizeof(hash_value_type) * 8 - 1);

CUDF_HOST_DEVICE constexpr hash_value_type set_mark(hash_value_type value) noexcept
{
  return value | mark_bit_mask;
}

CUDF_HOST_DEVICE constexpr hash_value_type unset_mark(hash_value_type value) noexcept
{
  return value & ~mark_bit_mask;
}

CUDF_HOST_DEVICE constexpr bool is_marked(hash_value_type value) noexcept
{
  return (value & mark_bit_mask) != 0;
}

/**
 * @brief Implementation of filtered join using a mark-based multiset hash table.
 *
 * This class extends the base filtered_join to implement join operations using
 * multiset semantics, where duplicate keys are allowed in the hash table.
 * This is used when the build table is the **left** table (`set_as_build_table::LEFT`),
 * which may contain duplicate keys.
 *
 * Instead of the traditional two-pass retrieve + sort/dedup approach, this uses
 * a mark-based algorithm: the probe kernel atomically sets the MSB (mark bit) on
 * matching hash table entries via CAS, then a retrieve kernel collects marked (semi)
 * or unmarked (anti) entries. This provides implicit deduplication and eliminates
 * O(N log N) sort overhead.
 */
struct masked_hash_fn {
  template <typename T>
  __device__ constexpr hash_value_type operator()(
    cuco::pair<hash_value_type, T> const& key) const noexcept
  {
    return unset_mark(key.first);
  }
};

struct secondary_hash_fn {
  uint32_t _seed{0};

  CUDF_HOST_DEVICE secondary_hash_fn() = default;
  CUDF_HOST_DEVICE secondary_hash_fn(uint32_t seed) : _seed{seed} {}

  template <typename T>
  CUDF_HOST_DEVICE auto operator()(cuco::pair<hash_value_type, T> const& key) const noexcept
  {
    return cuco::xxhash_32<hash_value_type>{_seed}(unset_mark(key.first));
  }
};

template <typename T, typename Hasher>
struct masked_key_fn {
  CUDF_HOST_DEVICE constexpr masked_key_fn(Hasher const& hasher) : _hasher{hasher} {}

  __device__ __forceinline__ auto operator()(size_type i) const noexcept
  {
    return cuco::pair{unset_mark(_hasher(i)), T{i}};
  }

 private:
  Hasher _hasher;
};

template <typename Equal>
struct masked_comparator_fn {
  masked_comparator_fn(Equal const& d_equal) : _d_equal{d_equal} {}

  __device__ constexpr auto operator()(
    cuco::pair<hash_value_type, lhs_index_type> const& lhs,
    cuco::pair<hash_value_type, lhs_index_type> const& rhs) const noexcept
  {
    if (unset_mark(lhs.first) != unset_mark(rhs.first)) { return false; }
    return _d_equal(lhs.second, rhs.second);
  }

  __device__ constexpr auto operator()(
    cuco::pair<hash_value_type, rhs_index_type> const& probe,
    cuco::pair<hash_value_type, lhs_index_type> const& build) const noexcept
  {
    if (unset_mark(probe.first) != unset_mark(build.first)) { return false; }
    return _d_equal(build.second, probe.second);
  }

 private:
  Equal _d_equal;
};

using masked_probing_scheme = cuco::double_hashing<1, masked_hash_fn, secondary_hash_fn>;

static constexpr auto masked_empty_sentinel =
  cuco::empty_key{cuco::pair{unset_mark(cuda::std::numeric_limits<hash_value_type>::max()),
                             lhs_index_type{cudf::JoinNoMatch}}};

class mark_join : public filtered_join {
 private:
  cudf::size_type _num_build_inserted{0};  ///< Number of build rows inserted into hash table

  [[nodiscard]] cudf::size_type num_build_inserted() const { return _num_build_inserted; }

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
   * @brief Core implementation: mark probe + mark retrieve
   *
   * Performs the mark-based probe of the hash table followed by a retrieve to
   * collect result indices. The probe kernel walks the hash table for each
   * probe row and atomically marks matching build entries. The retrieve kernel
   * then iterates the hash table and collects marked (or unmarked for anti)
   * entries into the output buffer using coalesced shared-memory buffered writes.
   *
   * @tparam CGSize CUDA cooperative group size
   * @tparam ProbingScheme Type of probing scheme (mark-aware)
   * @tparam Comparator Type of equality comparator (mark-aware)
   * @param probe The table to probe the hash table with
   * @param preprocessed_probe Preprocessed probe table for row operators
   * @param kind The kind of join to perform
   * @param probing_scheme The probing scheme instance
   * @param comparator The equality comparator instance
   * @param stream CUDA stream on which to perform operations
   * @param mr Memory resource for allocations
   * @return Device vector of indices representing the join result
   */
  template <typename Comparator>
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_probe_and_retrieve(
    cudf::table_view const& probe,
    std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_probe,
    join_kind kind,
    Comparator comparator,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  /**
   * @brief Clears all mark bits from the hash table entries
   *
   * Must be called before each probe to ensure independent results when the
   * hash table is reused across multiple semi_join/anti_join calls.
   *
   * @param stream CUDA stream on which to perform operations
   */
  void clear_marks(rmm::cuda_stream_view stream);

 public:
  /**
   * @brief Constructor for mark-based filtered join
   *
   * @param build The table to build the hash table from
   * @param compare_nulls How null values should be compared
   * @param load_factor Target load factor for the hash table
   * @param stream CUDA stream on which to perform operations
   */
  mark_join(cudf::table_view const& build,
            cudf::null_equality compare_nulls,
            double load_factor,
            rmm::cuda_stream_view stream);

  /**
   * @brief Implementation of semi join
   *
   * Returns indices of build table rows that have matching keys in the probe table.
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
   * @brief Implementation of anti join
   *
   * Returns indices of build table rows that do not have matching keys in the probe table.
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

}  // namespace cudf::detail
