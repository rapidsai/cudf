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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>

namespace cudf {
namespace detail {

/**
 * @brief Computes the trivial left join operation for the case when the
 * right table is empty. In this case all the valid indices of the left table
 * are returned with their corresponding right indices being set to
 * JoinNoneValue, i.e. -1.
 *
 * @param left Table of left columns to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the result
 *
 * @return Join output indices vector pair
 */
inline std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                 std::unique_ptr<rmm::device_uvector<size_type>>>
get_trivial_left_join_indices(
  table_view const& left,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);
  thrust::sequence(rmm::exec_policy(stream), left_indices->begin(), left_indices->end(), 0);
  auto right_indices =
    std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);
  thrust::fill(
    rmm::exec_policy(stream), right_indices->begin(), right_indices->end(), JoinNoneValue);
  return std::make_pair(std::move(left_indices), std::move(right_indices));
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
 * @param a The first pair of vectors.
 * @param b The second pair of vectors.
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return New pair of vectors containing the concatenated output.
 */
inline VectorPair concatenate_vector_pairs(VectorPair& a,
                                           VectorPair& b,
                                           rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS((a.first->size() == a.second->size()),
               "Mismatch between sizes of vectors in vector pair");
  CUDF_EXPECTS((b.first->size() == b.second->size()),
               "Mismatch between sizes of vectors in vector pair");
  if (a.first->is_empty()) {
    return std::move(b);
  } else if (b.first->is_empty()) {
    return std::move(a);
  }
  auto original_size = a.first->size();
  a.first->resize(a.first->size() + b.first->size(), stream);
  a.second->resize(a.second->size() + b.second->size(), stream);
  thrust::copy(
    rmm::exec_policy(stream), b.first->begin(), b.first->end(), a.first->begin() + original_size);
  thrust::copy(rmm::exec_policy(stream),
               b.second->begin(),
               b.second->end(),
               a.second->begin() + original_size);
  return std::move(a);
}

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
 * @brief  Creates a table containing the complement of left join indices.
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
inline std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                 std::unique_ptr<rmm::device_uvector<size_type>>>
get_left_join_indices_complement(std::unique_ptr<rmm::device_uvector<size_type>>& right_indices,
                                 size_type left_table_row_count,
                                 size_type right_table_row_count,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  // Get array of indices that do not appear in right_indices

  // Vector allocated for unmatched result
  auto right_indices_complement =
    std::make_unique<rmm::device_uvector<size_type>>(right_table_row_count, stream);

  // If left table is empty in a full join call then all rows of the right table
  // should be represented in the joined indices. This is an optimization since
  // if left table is empty and full join is called all the elements in
  // right_indices will be JoinNoneValue, i.e. -1. This if path should
  // produce exactly the same result as the else path but will be faster.
  if (left_table_row_count == 0) {
    thrust::sequence(rmm::exec_policy(stream),
                     right_indices_complement->begin(),
                     right_indices_complement->end(),
                     0);
  } else {
    // Assume all the indices in invalid_index_map are invalid
    auto invalid_index_map =
      std::make_unique<rmm::device_uvector<size_type>>(right_table_row_count, stream);
    thrust::uninitialized_fill(
      rmm::exec_policy(stream), invalid_index_map->begin(), invalid_index_map->end(), int32_t{1});

    // Functor to check for index validity since left joins can create invalid indices
    valid_range<size_type> valid(0, right_table_row_count);

    // invalid_index_map[index_ptr[i]] = 0 for i = 0 to right_table_row_count
    // Thus specifying that those locations are valid
    thrust::scatter_if(rmm::exec_policy(stream),
                       thrust::make_constant_iterator(0),
                       thrust::make_constant_iterator(0) + right_indices->size(),
                       right_indices->begin(),      // Index locations
                       right_indices->begin(),      // Stencil - Check if index location is valid
                       invalid_index_map->begin(),  // Output indices
                       valid);                      // Stencil Predicate
    size_type begin_counter = static_cast<size_type>(0);
    size_type end_counter   = static_cast<size_type>(right_table_row_count);

    // Create list of indices that have been marked as invalid
    size_type indices_count = thrust::copy_if(rmm::exec_policy(stream),
                                              thrust::make_counting_iterator(begin_counter),
                                              thrust::make_counting_iterator(end_counter),
                                              invalid_index_map->begin(),
                                              right_indices_complement->begin(),
                                              thrust::identity<size_type>()) -
                              right_indices_complement->begin();
    right_indices_complement->resize(indices_count, stream);
  }

  auto left_invalid_indices =
    std::make_unique<rmm::device_uvector<size_type>>(right_indices_complement->size(), stream);
  thrust::fill(rmm::exec_policy(stream),
               left_invalid_indices->begin(),
               left_invalid_indices->end(),
               JoinNoneValue);

  return std::make_pair(std::move(left_invalid_indices), std::move(right_indices_complement));
}

}  // namespace detail

}  // namespace cudf
