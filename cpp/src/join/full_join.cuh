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

#include <join/join_common_utils.hpp>
#include <cudf/copying.hpp>

namespace cudf {

namespace experimental {

namespace detail {

template <typename T>
struct valid_range {
    T start, stop;
    __host__ __device__
    valid_range(
            const T begin,
            const T end) :
        start(begin), stop(end) {}

    __host__ __device__ __forceinline__
    bool operator()(const T index)
    {
        return ((index >= start) && (index < stop));
    }
};


/* --------------------------------------------------------------------------*/
/**
* @Synopsis  Creates a table containing the complement of left join indices.
* This table has two columns. The first one is filled with JoinNoneValue(-1)
* and the second one contains values from 0 to right_table_row_count - 1
* excluding those found in the right_indices column.
* 
* @Param right_indices Vector of indices
* @Param left_table_row_count Number of rows of left table
* @Param right_table_row_count Number of rows of right table
* @param stream Optional, stream on which all memory allocations and copies
* will be performed
*
* @Returns  Pair of vectors containing the left join indices complement
*/
/* ----------------------------------------------------------------------------*/
template <typename index_type>
std::pair<rmm::device_vector<size_type>,
rmm::device_vector<size_type>>
get_left_join_indices_complement(
    rmm::device_vector<size_type>& right_indices,
    size_type left_table_row_count,
    size_type right_table_row_count,
    cudaStream_t stream) {

  //Get array of indices that do not appear in right_indices

  //Vector allocated for unmatched result
  rmm::device_vector<size_type> right_indices_complement(right_table_row_count);

  //If left table is empty in a full join call then all rows of the right table
  //should be represented in the joined indices. This is an optimization since
  //if left table is empty and full join is called all the elements in
  //right_indices will be JoinNoneValue, i.e. -1. This if path should
  //produce exactly the same result as the else path but will be faster.
  if (left_table_row_count == 0) {
    thrust::sequence(
        rmm::exec_policy(stream)->on(stream),
        right_indices_complement.begin(),
        right_indices_complement.end(),
        0);
  } else {
    //Assume all the indices in invalid_index_map are invalid
    rmm::device_vector<index_type> invalid_index_map(right_table_row_count, 1);
    //Functor to check for index validity since left joins can create invalid indices
    valid_range<size_type> valid(0, right_table_row_count);

    //invalid_index_map[index_ptr[i]] = 0 for i = 0 to right_table_row_count
    //Thus specifying that those locations are valid
    thrust::scatter_if(
        rmm::exec_policy(stream)->on(stream),
        thrust::make_constant_iterator(0),
        thrust::make_constant_iterator(0) + right_indices.size(),
        right_indices.begin(),//Index locations
        right_indices.begin(),//Stencil - Check if index location is valid
        invalid_index_map.begin(),//Output indices
        valid);//Stencil Predicate
    size_type begin_counter = static_cast<size_type>(0);
    size_type end_counter = static_cast<size_type>(right_table_row_count);

    //Create list of indices that have been marked as invalid
    size_type indices_count = thrust::copy_if(
        rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator(begin_counter),
        thrust::make_counting_iterator(end_counter),
        invalid_index_map.begin(),
        right_indices_complement.begin(),
        thrust::identity<index_type>()) -
      right_indices_complement.begin();
    right_indices_complement.resize(indices_count);
  }

  rmm::device_vector<size_type> left_invalid_indices(
      right_indices_complement.size(), JoinNoneValue);

  return std::make_pair(std::move(left_invalid_indices), std::move(right_indices_complement));
}

}//namespace detail

} //namespace experimental

}//namespace cudf
