/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>

namespace cudf {

namespace detail {

template <typename size_type>
struct ValidRange {
    size_type start, stop;
    __host__ __device__
    ValidRange(
            const size_type begin,
            const size_type end) :
        start(begin), stop(end) {}

    __host__ __device__ __forceinline__
    bool operator()(const size_type index)
    {
        return ((index >= start) && (index < stop));
    }
};

/* --------------------------------------------------------------------------*/
/**
* @Synopsis  Creates a column of indices containing values from 0 to
* right_table_row_count - 1 is created exluding those in the right_indices
* column.
*
* @Param right_indices Column of indices
* @Param left_table_row_count Number of rows of left table
* @Param right_table_row_count Number of rows of right table
* @param mr Optional, the memory resource that will be used for allocating
* the device memory for the new column
* @param stream Optional, stream on which all memory allocations and copies
* will be performed
*
* @Returns  Column containing the indices that are missing from right_indices
*/
/* ----------------------------------------------------------------------------*/
template <typename index_type>
std::unique_ptr<column>
create_missing_indices(
    column_view const& right_indices,
    const size_type left_table_row_count,
    const size_type right_table_row_count,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) {

  //Vector allocated for unmatched result
  rmm::device_buffer unmatched_indices{
    sizeof(index_type)*right_table_row_count, stream, mr};

  thrust::device_ptr<index_type> unmatched_indices_ptr(
      static_cast<index_type*>(unmatched_indices.data()));

  size_type indices_count = right_table_row_count;
  //If left table is empty in a full join call then all rows of the right table
  //should be represented in the joined indices. This is an optimization since
  //if left table is empty and full join is called all the elements in
  //right_indices will be JoinNoneValue, i.e. -1. This if path should
  //produce exactly the same result as the else path but will be faster.
  if (left_table_row_count == 0) {
    thrust::sequence(
        rmm::exec_policy(stream)->on(stream),
        unmatched_indices_ptr,
        unmatched_indices_ptr + right_table_row_count,
        0);
  } else {
    //Assume all the indices in invalid_index_map are invalid
    rmm::device_vector<index_type> invalid_index_map(right_table_row_count, 1);
    //Functor to check for index validity since left joins can create invalid indices
    ValidRange<size_type> valid_range(0, right_table_row_count);

    //invalid_index_map[index_ptr[i]] = 0 for i = 0 to right_table_row_count
    //Thus specifying that those locations are valid
    thrust::scatter_if(
        rmm::exec_policy(stream)->on(stream),
        thrust::make_constant_iterator(0),
        thrust::make_constant_iterator(0) + right_indices.size(),
        right_indices.begin<index_type>(),//Index locations
        right_indices.begin<index_type>(),//Stencil - Check if index location is valid
        invalid_index_map.begin(),//Output indices
        valid_range);//Stencil Predicate
    size_type begin_counter = static_cast<size_type>(0);
    size_type end_counter = static_cast<size_type>(right_table_row_count);

    //Create list of indices that have been marked as invalid
    indices_count = thrust::copy_if(
        rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator(begin_counter),
        thrust::make_counting_iterator(end_counter),
        invalid_index_map.begin(),
        unmatched_indices_ptr,
        thrust::identity<index_type>()) -
      unmatched_indices_ptr;
  }
  return std::make_unique<column>(
      integer_type<index_type>(),
      indices_count,
      std::move(unmatched_indices));
}


template <typename index_type>
std::unique_ptr<experimental::table>
get_join_indices_complement(
    column_view const& right_indices,
    size_type left_table_row_count,
    size_type right_table_row_count,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) {

  //Get array of indices that do not appear in r_index_ptr
  auto right_indices_complement = create_missing_indices<index_type>(
      right_indices, left_table_row_count, right_table_row_count, mr, stream);

  auto left_invalid_indices = experimental::allocate_like(right_indices_complement->view());
  thrust::device_ptr<index_type> inv_index_ptr(
      (left_invalid_indices->mutable_view()).template begin<index_type>());

  //Copy JoinNoneValue to inv_index_ptr to denote that a match does not exist on the left
  thrust::fill(
      rmm::exec_policy(stream)->on(stream),
      inv_index_ptr,
      inv_index_ptr + left_invalid_indices->size(),
      JoinNoneValue);

  std::vector<std::unique_ptr<column>> cols;
  cols.emplace_back(std::move(left_invalid_indices));
  cols.emplace_back(std::move(right_indices_complement));
  return std::make_unique<experimental::table>(std::move(cols));
}

}//namespace detail

}//namespace cudf
