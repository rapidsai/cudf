/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"

#include <cudf/join/join.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/functional>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/uninitialized_fill.h>

namespace cudf {
namespace detail {

bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type)
{
  // If there is nothing to join, then send empty table with all columns
  if (left.is_empty() || right.is_empty()) { return true; }

  // If left join and the left table is empty, return immediately
  if ((join_kind::LEFT_JOIN == join_type) && (0 == left.num_rows())) { return true; }

  // If Inner Join and either table is empty, return immediately
  if ((join_kind::INNER_JOIN == join_type) && ((0 == left.num_rows()) || (0 == right.num_rows()))) {
    return true;
  }

  // If left semi join (contains) and right table is empty,
  // return immediately
  if ((join_kind::LEFT_SEMI_JOIN == join_type) && (0 == right.num_rows())) { return true; }

  // If left semi- or anti- join, and the left table is empty, return immediately
  if ((join_kind::LEFT_SEMI_JOIN == join_type || join_kind::LEFT_ANTI_JOIN == join_type) &&
      (0 == left.num_rows())) {
    return true;
  }

  return false;
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
get_trivial_left_join_indices(table_view const& left,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);
  thrust::sequence(rmm::exec_policy(stream), left_indices->begin(), left_indices->end(), 0);
  auto right_indices =
    std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), right_indices->begin(), right_indices->end(), cudf::JoinNoMatch);
  return std::pair(std::move(left_indices), std::move(right_indices));
}

VectorPair concatenate_vector_pairs(VectorPair& a, VectorPair& b, rmm::cuda_stream_view stream)
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

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
get_left_join_indices_complement(std::unique_ptr<rmm::device_uvector<size_type>>& right_indices,
                                 size_type left_table_row_count,
                                 size_type right_table_row_count,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  // Get array of indices that do not appear in right_indices

  // Vector allocated for unmatched result
  auto right_indices_complement =
    std::make_unique<rmm::device_uvector<size_type>>(right_table_row_count, stream);

  // If left table is empty in a full join call then all rows of the right table
  // should be represented in the joined indices. This is an optimization since
  // if left table is empty and full join is called all the elements in
  // right_indices will be cudf::JoinNoMatch, i.e. `cuda::std::numeric_limits<size_type>::min()`.
  // This if path should produce exactly the same result as the else path but will be faster.
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
                                              cuda::std::identity{}) -
                              right_indices_complement->begin();
    right_indices_complement->resize(indices_count, stream);
  }

  auto left_invalid_indices =
    std::make_unique<rmm::device_uvector<size_type>>(right_indices_complement->size(), stream);
  thrust::uninitialized_fill(rmm::exec_policy(stream),
                             left_invalid_indices->begin(),
                             left_invalid_indices->end(),
                             cudf::JoinNoMatch);

  return std::pair(std::move(left_invalid_indices), std::move(right_indices_complement));
}

}  // namespace detail
}  // namespace cudf
