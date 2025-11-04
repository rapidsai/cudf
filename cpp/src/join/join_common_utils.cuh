/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "join_common_utils.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <memory>
#include <utility>

namespace cudf::detail {
template <typename Hasher>
struct pair_fn {
  CUDF_HOST_DEVICE pair_fn(Hasher hash) : _hash{std::move(hash)} {}

  __device__ cuco::pair<hash_value_type, size_type> operator()(size_type i) const noexcept
  {
    return cuco::pair{_hash(i), i};
  }

 private:
  Hasher _hash;
};

/**
 * @brief Device functor to determine if a row is valid.
 */
class row_is_valid {
 public:
  row_is_valid(bitmask_type const* row_bitmask) : _row_bitmask{row_bitmask} {}

  __device__ bool operator()(size_type const& i) const noexcept
  {
    return cudf::bit_is_set(_row_bitmask, i);
  }

 private:
  bitmask_type const* _row_bitmask;
};

/**
 * @brief Device functor to determine if two pairs are identical.
 *
 * This equality comparator is designed for use with cuco::static_multimap's
 * pair* APIs, which will compare equality based on comparing (key, value)
 * pairs. In the context of joins, these pairs are of the form
 * (row_hash, row_id). A hash probe hit indicates that hash of a probe row's hash is
 * equal to the hash of the hash of some row in the multimap, at which point we need an
 * equality comparator that will check whether the contents of the rows are
 * identical. This comparator does so by verifying key equality (i.e. that
 * probe_row_hash == build_row_hash) and then using a row_equality_comparator
 * to compare the contents of the row indices that are stored as the payload in
 * the hash map.
 *
 * @tparam Comparator The row comparator type to perform row equality comparison from row indices.
 */
template <typename DeviceComparator>
class pair_equality {
 public:
  pair_equality(DeviceComparator check_row_equality)
    : _check_row_equality{std::move(check_row_equality)}
  {
  }

  // The parameters are build/probe rather than left/right because the operator
  // is called by cuco's kernels with parameters in this order (note that this
  // is an implementation detail that we should eventually stop relying on by
  // defining operators with suitable heterogeneous typing). Rather than
  // converting to left/right semantics, we can operate directly on build/probe
  template <typename LhsPair, typename RhsPair>
  __device__ __forceinline__ bool operator()(LhsPair const& lhs, RhsPair const& rhs) const noexcept
  {
    using detail::row::lhs_index_type;
    using detail::row::rhs_index_type;

    return lhs.first == rhs.first and
           _check_row_equality(lhs_index_type{rhs.second}, rhs_index_type{lhs.second});
  }

 private:
  DeviceComparator _check_row_equality;
};

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
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
get_trivial_left_join_indices(table_view const& left,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

/**
 * @brief Builds the hash table based on the given `build_table`.
 *
 * @tparam MultimapType The type of the hash table
 *
 * @param build Table of columns used to build join hash.
 * @param preprocessed_build shared_ptr to cudf::detail::row::equality::preprocessed_table
 * for build
 * @param hash_table Build hash table.
 * @param has_nested_nulls Flag to denote if build or probe tables have nested nulls
 * @param nulls_equal Flag to denote nulls are equal or not.
 * @param bitmask Bitmask to denote whether a row is valid.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename HashTable>
void build_join_hash_table(
  cudf::table_view const& build,
  std::shared_ptr<detail::row::equality::preprocessed_table> const& preprocessed_build,
  HashTable& hash_table,
  bool has_nested_nulls,
  null_equality nulls_equal,
  [[maybe_unused]] bitmask_type const* bitmask,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(0 != build.num_columns(), "Selected build dataset is empty", std::invalid_argument);
  CUDF_EXPECTS(0 != build.num_rows(), "Build side table has no rows", std::invalid_argument);

  auto insert_rows = [&](auto const& build, auto const& d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});

    if (nulls_equal == cudf::null_equality::EQUAL or not nullable(build)) {
      hash_table.insert_async(iter, iter + build.num_rows(), stream.value());
    } else {
      auto const stencil = thrust::counting_iterator<size_type>{0};
      auto const pred    = row_is_valid{bitmask};

      // insert valid rows
      hash_table.insert_if_async(iter, iter + build.num_rows(), stencil, pred, stream.value());
    }
  };

  auto const nulls = nullate::DYNAMIC{has_nested_nulls};

  auto const row_hash = detail::row::hash::row_hasher{preprocessed_build};
  auto const d_hasher = row_hash.device_hasher(nulls);

  insert_rows(build, d_hasher);
}

// Convenient alias for a pair of unique pointers to device uvectors.
using VectorPair = std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                             std::unique_ptr<rmm::device_uvector<size_type>>>;

/**
 * @brief Takes two pairs of vectors and returns a single pair where the first
 * element is a vector made from concatenating the first elements of both input
 * pairs and the second element is a vector made from concatenating the second
 * elements of both input pairs.
 *
 * This function's primary use is for computing the indices of a full join by
 * first performing a left join, then separately getting the complementary
 * right join indices, then finally calling this function to concatenate the
 * results. In this case, each input VectorPair contains the left and right
 * indices from a join.
 *
 * Note that this is a destructive operation, in that at least one of a or b
 * will be invalidated (by a move) by this operation. Calling code should
 * assume that neither input VectorPair is valid after this function executes.
 *
 * @param a The first pair of vectors.
 * @param b The second pair of vectors.
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return A pair of vectors containing the concatenated output.
 */
VectorPair concatenate_vector_pairs(VectorPair& a, VectorPair& b, rmm::cuda_stream_view stream);

/**
 * @brief  Creates a table containing the complement of left join indices.
 *
 * This table has two columns. The first one is filled with `JoinNoMatch`
 * and the second one contains values from 0 to right_table_row_count - 1
 * excluding those found in the right_indices column.
 *
 * @param right_indices Vector of indices
 * @param left_table_row_count Number of rows of left table
 * @param right_table_row_count Number of rows of right table
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned vectors.
 *
 * @return Pair of vectors containing the left join indices complement
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
get_left_join_indices_complement(std::unique_ptr<rmm::device_uvector<size_type>>& right_indices,
                                 size_type left_table_row_count,
                                 size_type right_table_row_count,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @brief Device functor to determine if an index is contained in a range.
 */
template <typename T>
struct valid_range {
  T start, stop;
  __host__ __device__ valid_range(T const begin, T const end) : start(begin), stop(end) {}

  __host__ __device__ __forceinline__ bool operator()(T const index)
  {
    return ((index >= start) && (index < stop));
  }
};

}  // namespace cudf::detail
