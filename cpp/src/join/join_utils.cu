/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"

#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/std/functional>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/uninitialized_fill.h>

#include <memory>

namespace cudf {
namespace detail {

VectorPair get_trivial_left_join_indices(table_view const& left,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);
  thrust::sequence(rmm::exec_policy_nosync(stream), left_indices->begin(), left_indices->end(), 0);
  auto right_indices =
    std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);
  thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                             right_indices->begin(),
                             right_indices->end(),
                             cudf::JoinNoMatch);
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
  thrust::copy(rmm::exec_policy_nosync(stream),
               b.first->begin(),
               b.first->end(),
               a.first->begin() + original_size);
  thrust::copy(rmm::exec_policy_nosync(stream),
               b.second->begin(),
               b.second->end(),
               a.second->begin() + original_size);
  return std::move(a);
}

VectorPair get_left_join_indices_complement(
  std::unique_ptr<rmm::device_uvector<size_type>>& right_indices,
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
    thrust::sequence(rmm::exec_policy_nosync(stream),
                     right_indices_complement->begin(),
                     right_indices_complement->end(),
                     0);
  } else {
    // Assume all the indices in invalid_index_map are invalid
    auto invalid_index_map =
      std::make_unique<rmm::device_uvector<size_type>>(right_table_row_count, stream);
    thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                               invalid_index_map->begin(),
                               invalid_index_map->end(),
                               int32_t{1});

    // Functor to check for index validity since left joins can create invalid indices
    valid_range<size_type> valid(0, right_table_row_count);

    // invalid_index_map[index_ptr[i]] = 0 for i = 0 to right_table_row_count
    // Thus specifying that those locations are valid
    thrust::scatter_if(rmm::exec_policy_nosync(stream),
                       thrust::make_constant_iterator(0),
                       thrust::make_constant_iterator(0) + right_indices->size(),
                       right_indices->begin(),      // Index locations
                       right_indices->begin(),      // Stencil - Check if index location is valid
                       invalid_index_map->begin(),  // Output indices
                       valid);                      // Stencil Predicate
    size_type begin_counter = static_cast<size_type>(0);
    size_type end_counter   = static_cast<size_type>(right_table_row_count);

    // Create list of indices that have been marked as invalid
    size_type indices_count = thrust::copy_if(rmm::exec_policy_nosync(stream),
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
  thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                             left_invalid_indices->begin(),
                             left_invalid_indices->end(),
                             cudf::JoinNoMatch);

  return std::pair(std::move(left_invalid_indices), std::move(right_indices_complement));
}

}  // namespace detail
}  // namespace cudf
