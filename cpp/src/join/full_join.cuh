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
* @Synopsis  Creates a buffer of indices missing from an array of indices given
* a maximum index value
*
* A buffer of indices containing values from 0 to max_index_value - 1 is created
* provided that they do not appear in the range index_ptr to index_ptr + index_size
*
* @Param index_ptr Array of indices
* @Param max_index_value The maximum value an index can have in index_ptr
* @Param index_size Number of left and right indices
* @tparam index_type The type of data associated with index_ptr
* @tparam size_type The data type used for size calculations
*
* @Returns  rmm::device_vector containing the indices that are missing from index_ptr
*/
/* ----------------------------------------------------------------------------*/
template <typename index_type, typename size_type>
size_type
create_missing_indices(
    index_type const * const index_ptr,
    index_type * const unmatched_indices,
    const size_type max_index_value,
    const size_type index_size,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr) {
  //Assume all the indices in invalid_index_map are invalid
  rmm::device_buffer invalid_index_map{sizeof(index_type)*max_index_value, stream, mr};
  //Functor to check for index validity since left joins can create invalid indices
  ValidRange<size_type> valid_range(0, max_index_value);

  thrust::device_ptr<index_type> invalid_index_map_ptr(
      static_cast<index_type*>(invalid_index_map.data()));
  //invalid_index_map[index_ptr[i]] = 0 for i = 0 to max_index_value
  //Thus specifying that those locations are valid
  thrust::scatter_if(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_constant_iterator(0),
      thrust::make_constant_iterator(0) + index_size,
      index_ptr,//Index locations
      index_ptr,//Stencil - Check if index location is valid
      invalid_index_map_ptr,//Output indices
      valid_range);//Stencil Predicate
  size_type begin_counter = static_cast<size_type>(0);
  size_type end_counter = static_cast<size_type>(max_index_value);

  thrust::device_ptr<index_type> unmatched_indices_ptr(
      static_cast<index_type*>(unmatched_indices));
  //Create list of indices that have been marked as invalid
  size_type compacted_size = thrust::copy_if(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator(begin_counter),
      thrust::make_counting_iterator(end_counter),
      invalid_index_map_ptr,
      unmatched_indices_ptr,
      thrust::identity<index_type>()) -
    unmatched_indices_ptr;
  return compacted_size;
}


template <typename index_type, typename B = rmm::device_buffer>
std::unique_ptr<experimental::table>
get_full_join_indices_table(
    B&& left_indices,
    B&& right_indices,
    size_type join_size,
    size_type max_index_value,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr) {
  rmm::device_buffer l{std::forward<B>(left_indices)};
  rmm::device_buffer r{std::forward<B>(right_indices)};

  l.resize(sizeof(index_type)*join_size, stream);
  r.resize(sizeof(index_type)*join_size, stream);

  //Vector allocated for unmatched result
  rmm::device_buffer unmatched_indices{sizeof(index_type)*max_index_value, stream, mr};

  //Get array of indices that do not appear in r_index_ptr
  size_type unmatched_index_size = create_missing_indices(
      static_cast<index_type const *>(r.data()),
      static_cast<index_type *>(unmatched_indices.data()),
      max_index_value, join_size, stream, mr);
  unmatched_indices.resize(sizeof(index_type)*unmatched_index_size, stream);
  CUDA_CHECK_LAST();

  size_type full_join_size = unmatched_index_size + join_size;

  l.resize(sizeof(index_type)*full_join_size, stream);
  r.resize(sizeof(index_type)*full_join_size, stream);

  thrust::device_ptr<index_type> l_index_ptr(static_cast<index_type*>(l.data()));
  thrust::device_ptr<index_type> r_index_ptr(static_cast<index_type*>(r.data()));
  //Copy JoinNoneValue to l_index_ptr to denote that a match does not exist on the left
  thrust::fill(
      rmm::exec_policy(stream)->on(stream),
      l_index_ptr + join_size,
      l_index_ptr + full_join_size,
      JoinNoneValue);

  thrust::device_ptr<index_type> i_index_ptr(static_cast<index_type*>(unmatched_indices.data()));
  //Copy unmatched indices to the r_index_ptr
  thrust::copy(
      rmm::exec_policy(stream)->on(stream),
      i_index_ptr,
      i_index_ptr + unmatched_index_size,
      r_index_ptr + join_size);
  return get_indices_table<index_type>(
      std::move(l), std::move(r),
      full_join_size, stream, mr);
}

}//namespace detail

}//namespace cudf
