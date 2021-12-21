/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <join/join_common_utils.hpp>

#include <cudf/detail/utilities/cuda.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

namespace cudf {
namespace detail {

/**
 * @brief Device functor to determine if a row is valid.
 */
class row_is_valid {
 public:
  row_is_valid(bitmask_type const* row_bitmask) : _row_bitmask{row_bitmask} {}

  __device__ __inline__ bool operator()(const size_type& i) const noexcept
  {
    return cudf::bit_is_set(_row_bitmask, i);
  }

 private:
  bitmask_type const* _row_bitmask;
};

/**
 * @brief Device functor to determine if two pairs are identical.
 */
class pair_equality {
 public:
  pair_equality(table_device_view lhs,
                table_device_view rhs,
                nullate::DYNAMIC has_nulls,
                null_equality nulls_are_equal = null_equality::EQUAL)
    : _check_row_equality{has_nulls, lhs, rhs, nulls_are_equal}
  {
  }

  __device__ __forceinline__ bool operator()(const pair_type& lhs,
                                             const pair_type& rhs) const noexcept
  {
    return lhs.first == rhs.first and _check_row_equality(rhs.second, lhs.second);
  }

 private:
  row_equality _check_row_equality;
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
get_trivial_left_join_indices(
  table_view const& left,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
                                 rmm::mr::device_memory_resource* mr);

/**
 * @brief Device functor to determine if an index is contained in a range.
 */
template <typename T>
struct valid_range {
  T start, stop;
  __host__ __device__ valid_range(const T begin, const T end) : start(begin), stop(end) {}

  __host__ __device__ __forceinline__ bool operator()(const T index)
  {
    return ((index >= start) && (index < stop));
  }
};

/**
 * @brief Adds a pair of indices to the shared memory cache
 *
 * @param[in] first The first index in the pair
 * @param[in] second The second index in the pair
 * @param[in,out] current_idx_shared Pointer to shared index that determines
 * where in the shared memory cache the pair will be written
 * @param[in] warp_id The ID of the warp of the calling the thread
 * @param[out] joined_shared_l Pointer to the shared memory cache for left indices
 * @param[out] joined_shared_r Pointer to the shared memory cache for right indices
 */
__inline__ __device__ void add_pair_to_cache(const size_type first,
                                             const size_type second,
                                             size_type* current_idx_shared,
                                             const int warp_id,
                                             size_type* joined_shared_l,
                                             size_type* joined_shared_r)
{
  size_type my_current_idx{atomicAdd(current_idx_shared + warp_id, size_type(1))};

  // its guaranteed to fit into the shared cache
  joined_shared_l[my_current_idx] = first;
  joined_shared_r[my_current_idx] = second;
}

template <int num_warps, cudf::size_type output_cache_size>
__device__ void flush_output_cache(const unsigned int activemask,
                                   const cudf::size_type max_size,
                                   const int warp_id,
                                   const int lane_id,
                                   cudf::size_type* current_idx,
                                   cudf::size_type current_idx_shared[num_warps],
                                   size_type join_shared_l[num_warps][output_cache_size],
                                   size_type join_shared_r[num_warps][output_cache_size],
                                   size_type* join_output_l,
                                   size_type* join_output_r)
{
  // count how many active threads participating here which could be less than warp_size
  int num_threads               = __popc(activemask);
  cudf::size_type output_offset = 0;

  if (0 == lane_id) { output_offset = atomicAdd(current_idx, current_idx_shared[warp_id]); }

  // No warp sync is necessary here because we are assuming that ShuffleIndex
  // is internally using post-CUDA 9.0 synchronization-safe primitives
  // (__shfl_sync instead of __shfl). __shfl is technically not guaranteed to
  // be safe by the compiler because it is not required by the standard to
  // converge divergent branches before executing.
  output_offset = cub::ShuffleIndex<detail::warp_size>(output_offset, 0, activemask);

  for (int shared_out_idx = lane_id; shared_out_idx < current_idx_shared[warp_id];
       shared_out_idx += num_threads) {
    cudf::size_type thread_offset = output_offset + shared_out_idx;
    if (thread_offset < max_size) {
      join_output_l[thread_offset] = join_shared_l[warp_id][shared_out_idx];
      join_output_r[thread_offset] = join_shared_r[warp_id][shared_out_idx];
    }
  }
}

}  // namespace detail

}  // namespace cudf
