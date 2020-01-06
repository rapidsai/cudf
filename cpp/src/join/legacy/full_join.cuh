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
#ifndef FULL_JOIN_CUH
#define FULL_JOIN_CUH

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <rmm/thrust_rmm_allocator.h>

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

/** ---------------------------------------------------------------------------*
* @file full_join.cuh
* @brief Implementation of full join.
*
* The functions in this file are used to append indices to the output of a left
* join call to create the output of a full join. The highest level function in
* this file is append_full_join_indices.
* ---------------------------------------------------------------------------**/

/* --------------------------------------------------------------------------*/
/**
* @Synopsis  Creates a vector of indices missing from an array of indices given
* a maximum index value
*
* A vector of indices containing values from 0 to max_index_value - 1 is created
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
rmm::device_vector<index_type>
create_missing_indices(
        index_type const * const index_ptr,
        const size_type max_index_value,
        const size_type index_size,
        cudaStream_t stream) {
    //Assume all the indices in invalid_index_map are invalid
    rmm::device_vector<index_type> invalid_index_map(max_index_value, 1);
    //Vector allocated for unmatched result
    rmm::device_vector<index_type> unmatched_indices(max_index_value);
    //Functor to check for index validity since left joins can create invalid indices
    ValidRange<size_type> valid_range(0, max_index_value);

    //invalid_index_map[index_ptr[i]] = 0 for i = 0 to max_index_value
    //Thus specifying that those locations are valid
    thrust::scatter_if(
            rmm::exec_policy(stream)->on(stream),
            thrust::make_constant_iterator(0),
            thrust::make_constant_iterator(0) + index_size,
            index_ptr,//Index locations
            index_ptr,//Stencil - Check if index location is valid
            invalid_index_map.begin(),//Output indices
            valid_range);//Stencil Predicate
    size_type begin_counter = static_cast<size_type>(0);
    size_type end_counter = static_cast<size_type>(invalid_index_map.size());
    //Create list of indices that have been marked as invalid
    size_type compacted_size = thrust::copy_if(
            rmm::exec_policy(stream)->on(stream),
            thrust::make_counting_iterator(begin_counter),
            thrust::make_counting_iterator(end_counter),
            invalid_index_map.begin(),
            unmatched_indices.begin(),
            thrust::identity<index_type>()) -
        unmatched_indices.begin();
    unmatched_indices.resize(compacted_size);
    return unmatched_indices;
}

/* --------------------------------------------------------------------------*/
/**
* @Synopsis  Expands a buffer's size if the requested size is greater that its
* current size.
*
* @Param buffer Address of the buffer to expand
* @Param buffer_capacity Memory allocated for buffer
* @Param buffer_size Number of elements in the buffer
* @Param expand_size Amount of extra elements to be pushed into the buffer
* @tparam data_type The type of data associated with the buffer
* @tparam size_type The data type used for size calculations
*
* @Returns  cudaSuccess upon successful completion of buffer expansion. Otherwise returns
* the appropriate CUDA error code
*/
/* ----------------------------------------------------------------------------*/
template <typename data_type, typename size_type>
gdf_error expand_buffer(
        data_type ** buffer,
        size_type * const buffer_capacity,
        const size_type buffer_size,
        const size_type expand_size,
        cudaStream_t stream) {
    size_type requested_size = buffer_size + expand_size;
    //No need to proceed if the buffer can contain requested additional elements
    if (*buffer_capacity >= requested_size) {
        return GDF_SUCCESS;
    }
    data_type * new_buffer{nullptr};
    data_type * old_buffer = *buffer;
    RMM_TRY( RMM_ALLOC((void**)&new_buffer, requested_size*sizeof(data_type), stream) );
    CUDA_TRY( cudaMemcpy(new_buffer, old_buffer, buffer_size*sizeof(data_type), cudaMemcpyDeviceToDevice) );
    RMM_TRY( RMM_FREE(old_buffer, stream) );
    *buffer = new_buffer;
    *buffer_capacity = requested_size;

    return GDF_SUCCESS;
}

/* --------------------------------------------------------------------------*/
/**
* @Synopsis  Adds indices that are missing in r_index_ptr at the ends and places
* JoinNoneValue to the corresponding l_index_ptr.
*
* @Param l_index_ptr Address of the left indices
* @Param r_index_ptr Address of the right indices
* @Param index_capacity Amount of memory allocated for left and right indices
* @Param index_size Number of left and right indices
* @Param max_index_value The maximum value an index can have in r_index_ptr
* @tparam index_type The type of data associated with index_ptr
* @tparam size_type The data type used for size calculations
*
* @Returns  cudaSuccess upon successful completion of append call. Otherwise returns
* the appropriate CUDA error code
*/
/* ----------------------------------------------------------------------------*/
template <typename index_type, typename size_type>
gdf_error append_full_join_indices(
        index_type ** l_index_ptr,
        index_type ** r_index_ptr,
        size_type &index_capacity,
        size_type &index_size,
        const size_type max_index_value,
        cudaStream_t stream) {
    gdf_error err;
    //Get array of indices that do not appear in r_index_ptr
    rmm::device_vector<index_type> unmatched_indices =
        create_missing_indices(
                *r_index_ptr, max_index_value, index_size, stream);
    CHECK_CUDA(stream);

    //Expand l_index_ptr and r_index_ptr if necessary
    size_type mismatch_index_size = unmatched_indices.size();
    size_type l_index_capacity = index_capacity;
    size_type r_index_capacity = index_capacity;
    err = expand_buffer(l_index_ptr, &l_index_capacity, index_size, mismatch_index_size, stream);
    if (GDF_SUCCESS != err) return err;
    err = expand_buffer(r_index_ptr, &r_index_capacity, index_size, mismatch_index_size, stream);
    if (GDF_SUCCESS != err) return err;

    //Copy JoinNoneValue to l_index_ptr to denote that a match does not exist on the left
    thrust::fill(
            rmm::exec_policy(stream)->on(stream),
            *l_index_ptr + index_size,
            *l_index_ptr + index_size + mismatch_index_size,
            JoinNoneValue);

    //Copy unmatched indices to the r_index_ptr
    thrust::copy(
            rmm::exec_policy(stream)->on(stream),
            unmatched_indices.begin(),
            unmatched_indices.begin() + mismatch_index_size,
            *r_index_ptr + index_size);
    index_capacity = l_index_capacity;
    index_size = index_size + mismatch_index_size;

    CHECK_CUDA(stream);
    return GDF_SUCCESS;
}

#endif
