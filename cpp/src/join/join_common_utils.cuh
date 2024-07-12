/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "join_common_utils.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace detail {
/**
 * @brief Remaps a hash value to a new value if it is equal to the specified sentinel value.
 *
 * @param hash The hash value to potentially remap
 * @param sentinel The reserved value
 */
template <typename H, typename S>
constexpr auto remap_sentinel_hash(H hash, S sentinel)
{
  // Arbitrarily choose hash - 1
  return (hash == sentinel) ? (hash - 1) : hash;
}

/**
 * @brief Device functor to create a pair of {hash_value, row_index} for a given row.
 *
 * @tparam T Type of row index, must be convertible to `size_type`.
 * @tparam Hasher The type of internal hasher to compute row hash.
 */
template <typename Hasher, typename T = size_type>
class make_pair_function {
 public:
  CUDF_HOST_DEVICE make_pair_function(Hasher const& hash, hash_value_type const empty_key_sentinel)
    : _hash{hash}, _empty_key_sentinel{empty_key_sentinel}
  {
  }

  __device__ __forceinline__ auto operator()(size_type i) const noexcept
  {
    // Compute the hash value of row `i`
    auto row_hash_value = remap_sentinel_hash(_hash(i), _empty_key_sentinel);
    return cuco::make_pair(row_hash_value, T{i});
  }

 private:
  Hasher _hash;
  hash_value_type const _empty_key_sentinel;
};

/**
 * @brief Device functor to determine if a row is valid.
 */
class row_is_valid {
 public:
  row_is_valid(bitmask_type const* row_bitmask) : _row_bitmask{row_bitmask} {}

  __device__ __inline__ bool operator()(size_type const& i) const noexcept
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
    using experimental::row::lhs_index_type;
    using experimental::row::rhs_index_type;

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
 * JoinNoneValue, i.e. -1.
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
 * @param preprocessed_build shared_ptr to cudf::experimental::row::equality::preprocessed_table for
 *                           build
 * @param hash_table Build hash table.
 * @param has_nulls Flag to denote if build or probe tables have nested nulls
 * @param nulls_equal Flag to denote nulls are equal or not.
 * @param bitmask Bitmask to denote whether a row is valid.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 */
template <typename MultimapType>
void build_join_hash_table(
  cudf::table_view const& build,
  std::shared_ptr<experimental::row::equality::preprocessed_table> const& preprocessed_build,
  MultimapType& hash_table,
  bool has_nulls,
  null_equality nulls_equal,
  [[maybe_unused]] bitmask_type const* bitmask,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(0 != build.num_columns(), "Selected build dataset is empty");
  CUDF_EXPECTS(0 != build.num_rows(), "Build side table has no rows");

  std::unordered_set<cudf::type_id> build_column_types;
  for (auto col : build) {
    build_column_types.insert(col.type().id());
  }

  auto const row_hash   = experimental::row::hash::row_hasher{preprocessed_build};
  auto const hash_build = row_hash.device_hasher(build_column_types, nullate::DYNAMIC{has_nulls});

  auto const empty_key_sentinel = hash_table.get_empty_key_sentinel();

  size_type const build_table_num_rows{build.num_rows()};
  if (nulls_equal == cudf::null_equality::EQUAL or (not nullable(build))) {
    std::visit(
      [&](auto&& hasher) {
        make_pair_function pair_func{hasher, empty_key_sentinel};
        auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_func);
        hash_table.insert(iter, iter + build_table_num_rows, stream.value());
      },
      hash_build);
  } else {
    thrust::counting_iterator<size_type> stencil(0);
    row_is_valid pred{bitmask};

    std::visit(
      [&](auto&& hasher) {
        make_pair_function pair_func{hasher, empty_key_sentinel};
        auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_func);
        // insert valid rows
        hash_table.insert_if(iter, iter + build_table_num_rows, stencil, pred, stream.value());
      },
      hash_build);
  }
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
 * This table has two columns. The first one is filled with JoinNoneValue(-1)
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

}  // namespace detail

}  // namespace cudf
