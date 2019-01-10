/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#ifndef JOIN_COMPUTE_API_H
#define JOIN_COMPUTE_API_H

#include <cuda_runtime.h>
#include <future>

#include "join_kernels.cuh"

#include "dataframe/cudf_table.cuh"
#include "rmm/rmm.h"
#include "utilities/error_utils.h"

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

// TODO for Arrow integration:
//   1) replace mgpu::context_t with a new CudaComputeContext class (see the design doc)
//   2) replace cudaError_t with arrow::Status
//   3) replace input iterators & input counts with arrow::Datum
//   3) replace output iterators & output counts with arrow::ArrayData

constexpr int64_t DEFAULT_HASH_TABLE_OCCUPANCY = 50;
constexpr int DEFAULT_CUDA_BLOCK_SIZE = 128;
constexpr int DEFAULT_CUDA_CACHE_SIZE = 128;

/* --------------------------------------------------------------------------*/
/**
* @Synopsis  Creates a vector of indices that do not appear in index_ptr
*
* @Param index_ptr Array of indices
* @Param max_index_value The maximum value an index can have in index_ptr
* @Param index_size Number of left and right indices
* @tparam index_type The type of data associated with index_ptr
* @tparam size_type The data type used for size calculations
*
* @Returns  thrust::device_vector containing the indices that are missing from index_ptr
*/
/* ----------------------------------------------------------------------------*/
template <typename index_type, typename size_type>
thrust::device_vector<index_type>
create_missing_indices(
        index_type const * const index_ptr,
        const size_type max_index_value,
		const size_type index_size) {
	//Assume all the indices in invalid_index_map are invalid
	thrust::device_vector<index_type> invalid_index_map(max_index_value, 1);
	//Vector allocated for unmatched result
	thrust::device_vector<index_type> unmatched_indices(max_index_value);
	//Functor to check for index validity since left joins can create invalid indices
	ValidRange<size_type> valid_range(0, max_index_value);

	//invalid_index_map[index_ptr[i]] = 0 for i = 0 to max_index_value
	//Thus specifying that those locations are valid
	thrust::scatter_if(
			thrust::device,
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
			thrust::device,
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
* @Synopsis  Expands a buffer's size
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
        const size_type expand_size) {
    size_type requested_size = buffer_size + expand_size;
    //No need to proceed if the buffer can contain requested additional elements
    if (*buffer_capacity >= requested_size) {
        return GDF_SUCCESS;
    }
    data_type * new_buffer{nullptr};
    data_type * old_buffer = *buffer;
    RMM_TRY( RMM_ALLOC((void**)&new_buffer, requested_size*sizeof(data_type), 0) );
    CUDA_TRY( cudaMemcpy(new_buffer, old_buffer, buffer_size*sizeof(data_type), cudaMemcpyDeviceToDevice) );
    RMM_TRY( RMM_FREE(old_buffer, 0) );
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
        size_type * const index_capacity,
        size_type * const index_size,
        const size_type max_index_value) {
    gdf_error err;
    //Get array of indices that do not appear in r_index_ptr
    thrust::device_vector<index_type> unmatched_indices =
        create_missing_indices(
                *r_index_ptr, max_index_value, *index_size);
    CUDA_CHECK_LAST()

    //Expand l_index_ptr and r_index_ptr if necessary
    size_type mismatch_index_size = unmatched_indices.size();
    size_type l_index_capacity = *index_capacity;
    size_type r_index_capacity = *index_capacity;
    err = expand_buffer(l_index_ptr, &l_index_capacity, *index_size, mismatch_index_size);
    if (GDF_SUCCESS != err) return err;
    err = expand_buffer(r_index_ptr, &r_index_capacity, *index_size, mismatch_index_size);
    if (GDF_SUCCESS != err) return err;

    //Copy JoinNoneValue to l_index_ptr to denote that a match does not exist on the left
    thrust::fill(
            thrust::device,
            *l_index_ptr + *index_size,
            *l_index_ptr + *index_size + mismatch_index_size,
            JoinNoneValue);

    //Copy unmatched indices to the r_index_ptr
    thrust::copy(thrust::device,
            unmatched_indices.begin(),
            unmatched_indices.begin() + mismatch_index_size,
            *r_index_ptr + *index_size);
    *index_capacity = l_index_capacity;
    *index_size = *index_size + mismatch_index_size;

    CUDA_CHECK_LAST()
	return GDF_SUCCESS;
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Gives an estimate of the size of the join output produced when
 * joining two tables together. If the two tables are of relatively equal size,
 * then the returned output size will be the exact output size. However, if the
 * probe table is significantly larger than the build table, then we attempt
 * to estimate the output size by using only a subset of the rows in the probe table.
 * 
 * @Param build_table The right hand table
 * @Param probe_table The left hand table
 * @Param hash_table A hash table built on the build table that maps the index
 * of every row to the hash value of that row.
 * 
 * @Returns An estimate of the size of the output of the join operation
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type,
          typename multimap_type,
          typename size_type>
gdf_error estimate_join_output_size(gdf_table<size_type> const & build_table,
                                    gdf_table<size_type> const & probe_table,
                                    multimap_type const & hash_table,
                                    size_type * join_output_size_estimate)
{
  const size_type build_table_num_rows{build_table.get_column_length()};
  const size_type probe_table_num_rows{probe_table.get_column_length()};
  
  // If the probe table is significantly larger (5x) than the build table, 
  // then we attempt to only use a subset of the probe table rows to compute an
  // estimate of the join output size.
  size_type probe_to_build_ratio{0};
  if(build_table_num_rows > 0) {
    probe_to_build_ratio = static_cast<size_type>(std::ceil(static_cast<float>(probe_table_num_rows)/build_table_num_rows));
  }
  else {
    // If the build table is empty, we know exactly how large the output
    // will be for the different types of joins and can return immediately
    switch(join_type)
    {
      case JoinType::INNER_JOIN:
        {
          // Inner join with an empty table will have no output
          *join_output_size_estimate = 0;
          break;
        }
      case JoinType::LEFT_JOIN:
        {
          // Left join with an empty table will have an output of NULL rows
          // equal to the number of rows in the probe table
          *join_output_size_estimate = probe_table_num_rows;
          break;
        }
      default:
        return GDF_UNSUPPORTED_JOIN_TYPE;
    }
    return GDF_SUCCESS;
  }

  size_type sample_probe_num_rows{probe_table_num_rows};
  constexpr size_type MAX_RATIO{5};
  if(probe_to_build_ratio > MAX_RATIO)
  {
    sample_probe_num_rows = build_table_num_rows;
  }

  // Allocate storage for the counter used to get the size of the join output
  size_type * d_size_estimate{nullptr};
  size_type h_size_estimate{0};

  CUDA_TRY(cudaMallocHost(&d_size_estimate, sizeof(size_type)));
  *d_size_estimate = 0;

  CUDA_TRY( cudaGetLastError() );

  // Continue probing with a subset of the probe table until either:
  // a non-zero output size estimate is found OR
  // all of the rows in the probe table have been sampled
  do{

    sample_probe_num_rows = std::min(sample_probe_num_rows, probe_table_num_rows);

    *d_size_estimate = 0;

    constexpr int block_size{DEFAULT_CUDA_BLOCK_SIZE};
    const size_type probe_grid_size{(sample_probe_num_rows + block_size -1)/block_size};
    
    // Probe the hash table without actually building the output to simply
    // find what the size of the output will be.
    compute_join_output_size<join_type,
                             multimap_type,
                             size_type,
                             block_size,
                             DEFAULT_CUDA_CACHE_SIZE>
    <<<probe_grid_size, block_size>>>(&hash_table,
                                      build_table,
                                      probe_table,
                                      sample_probe_num_rows,
                                      d_size_estimate);

    // Device sync is required to ensure d_size_estimate is updated
    CUDA_TRY( cudaDeviceSynchronize() );
    
    	
    // Increase the estimated output size by a factor of the ratio between the
    // probe and build tables
    h_size_estimate = *d_size_estimate * probe_to_build_ratio;

    // If the size estimate is non-zero, then we have a valid estimate and can break
    // If sample_probe_num_rows >= probe_table_num_rows, then we've sampled the entire
    // probe table, in which case the estimate is exact and we can break 
    if((h_size_estimate > 0) 
       || (sample_probe_num_rows >= probe_table_num_rows))
    {
      break;
    }

    // If the size estimate is zero, then double the number of sampled rows in the probe
    // table. Reduce the ratio of the number of probe rows sampled to the 
    // number of rows in the build table by the same factor
    if(0 == h_size_estimate)
    {
      constexpr size_type GROW_RATIO{2};
      sample_probe_num_rows *= GROW_RATIO;
      probe_to_build_ratio = static_cast<size_type>(std::ceil(static_cast<float>(probe_to_build_ratio)/GROW_RATIO));
    }

  } while(true);

  CUDA_TRY( cudaFreeHost(d_size_estimate) );

  *join_output_size_estimate = h_size_estimate;

  return GDF_SUCCESS;
}

/**---------------------------------------------------------------------------*
 * @brief Computes the number of entries required in a hash table to satisfy
 * inserting a specified number of keys to achieve the specified hash table
 * occupancy.
 *
 * @param num_keys_to_insert The number of keys that will be inserted
 * @param desired_occupancy The desired occupancy percentage, e.g., 50 implies a
 * 50% occupancy
 * @return size_t The size of the hash table that will satisfy the desired
 * occupancy for the specified number of insertions
 *---------------------------------------------------------------------------**/
inline size_t compute_hash_table_size(
    gdf_size_type num_keys_to_insert,
    uint32_t desired_occupancy = DEFAULT_HASH_TABLE_OCCUPANCY) {
  assert(desired_occupancy != 0);
  assert(desired_occupancy <= 100);
  double const grow_factor{100.0 / desired_occupancy};

  // Calculate size of hash map based on the desired occupancy
  size_t hash_table_size{
      static_cast<size_t>(std::ceil(num_keys_to_insert * grow_factor))};

  return hash_table_size;
}

/* --------------------------------------------------------------------------*/
/**
* @Synopsis  Performs a hash-based join between two sets of gdf_tables.
*
* @Param joined_output The output of the join operation
* @Param left_table The left table to join
* @Param right_table The right table to join
* @Param flip_results Flag that indicates whether the left and right tables have been
* switched, indicating that the output indices should also be flipped
* @tparam join_type The type of join to be performed
* @tparam hash_value_type The data type to be used for the Keys in the hash table
* @tparam output_index_type The data type to be used for the output indices
* @tparam size_type The data type used for size calculations, e.g. size of hash table
*
* @Returns  cudaSuccess upon successful completion of the join. Otherwise returns
* the appropriate CUDA error code
*/
/* ----------------------------------------------------------------------------*/
template<JoinType join_type,
         typename output_index_type,
         typename size_type>
gdf_error compute_hash_join(
                            gdf_column * const output_l, 
                            gdf_column * const output_r,
                            gdf_table<size_type> const & left_table,
                            gdf_table<size_type> const & right_table,
                            bool flip_results = false)
{
  gdf_error gdf_error_code{GDF_SUCCESS};

  gdf_column_view(output_l, nullptr, nullptr, 0, N_GDF_TYPES);
  gdf_column_view(output_r, nullptr, nullptr, 0, N_GDF_TYPES);

  // The LEGACY allocator allocates the hash table array with normal cudaMalloc,
  // the non-legacy allocator uses managed memory
#ifdef HT_LEGACY_ALLOCATOR
  using multimap_type = concurrent_unordered_multimap<hash_value_type,
                                                      output_index_type,
                                                      size_type,
                                                      std::numeric_limits<hash_value_type>::max(),
                                                      std::numeric_limits<output_index_type>::max(),
                                                      default_hash<hash_value_type>,
                                                      equal_to<hash_value_type>,
                                                      legacy_allocator< thrust::pair<hash_value_type, output_index_type> > >;
#else
  using multimap_type = concurrent_unordered_multimap<hash_value_type,
                                                      output_index_type,
                                                      size_type,
                                                      std::numeric_limits<hash_value_type>::max(),
                                                      std::numeric_limits<output_index_type>::max()>;
#endif
  //If FULL_JOIN is selected then we process as LEFT_JOIN till we need to take care of unmatched indices
  constexpr JoinType base_join_type = (join_type == JoinType::FULL_JOIN)? JoinType::LEFT_JOIN : join_type;

  // Hash table will be built on the right table
  gdf_table<size_type> const & build_table{right_table};
  const size_type build_table_num_rows{build_table.get_column_length()};
  
  // Probe with the left table
  gdf_table<size_type> const & probe_table{left_table};
  const size_type probe_table_num_rows{probe_table.get_column_length()};

  // Hash table size must be at least 1 in order to have a valid allocation.
  // Even if the hash table will be empty, it still must be allocated for the
  // probing phase in the event of an outer join
  size_t const hash_table_size =
      std::max(compute_hash_table_size(build_table_num_rows), size_t{1});

  std::unique_ptr<multimap_type> hash_table(new multimap_type(hash_table_size));

  // FIXME: use GPU device id from the context?
  // but moderngpu only provides cudaDeviceProp
  // (although should be possible once we move to Arrow)
  hash_table->prefetch(0);

  CUDA_TRY( cudaDeviceSynchronize() );

  // Allocate a gdf_error for the device to hold error code returned from
  // the build kernel and intialize with GDF_SUCCESS
  // Use Page Locked memory to avoid overhead of memcpys
  gdf_error * d_gdf_error_code{nullptr};
  CUDA_TRY( cudaMallocHost(&d_gdf_error_code, sizeof(gdf_error)) );
  *d_gdf_error_code = GDF_SUCCESS;

  constexpr int block_size{DEFAULT_CUDA_BLOCK_SIZE};

  // build the hash table
  if(build_table_num_rows > 0)
  {
    const size_type build_grid_size{(build_table_num_rows + block_size - 1)/block_size};
    build_hash_table<<<build_grid_size, block_size>>>(hash_table.get(),
                                                      build_table,
                                                      build_table_num_rows,
                                                      d_gdf_error_code);
    
    // Device synch is required to ensure d_gdf_error_code 
    // has been written
    CUDA_TRY( cudaDeviceSynchronize() );
  }

  // Check error code from the kernel
  gdf_error_code = *d_gdf_error_code;
  if(GDF_SUCCESS != gdf_error_code){
    return gdf_error_code;
  }


  size_type estimated_join_output_size{0};
  gdf_error_code = estimate_join_output_size<base_join_type, multimap_type>(build_table, probe_table, *hash_table, &estimated_join_output_size);

  if(GDF_SUCCESS != gdf_error_code){
    return gdf_error_code;
  }

  // If the estimated output size is zero, return immediately
  if(0 == estimated_join_output_size){
    return GDF_SUCCESS;
  }

  // Because we are approximating the number of joined elements, our approximation 
  // might be incorrect and we might have underestimated the number of joined elements. 
  // As such we will need to de-allocate memory and re-allocate memory to ensure 
  // that the final output is correct.
  size_type h_actual_found{0};
  output_index_type *output_l_ptr{nullptr};
  output_index_type *output_r_ptr{nullptr};
  bool cont = true;

  // Allocate device global counter used by threads to determine output write location
  size_type *d_global_write_index{nullptr};
  RMM_TRY( RMM_ALLOC((void**)&d_global_write_index, sizeof(size_type), 0) ); // TODO non-default stream?
 
  // Because we only have an estimate of the output size, we may need to probe the
  // hash table multiple times until we've found an output buffer size that is large enough
  // to hold the output
  while(cont)
  {
    output_l_ptr = nullptr;
    output_r_ptr = nullptr;

    // Allocate temporary device buffer for join output
    RMM_TRY( RMM_ALLOC((void**)&output_l_ptr, estimated_join_output_size*sizeof(output_index_type), 0) );
    RMM_TRY( RMM_ALLOC((void**)&output_r_ptr, estimated_join_output_size*sizeof(output_index_type), 0) );
    CUDA_TRY( cudaMemsetAsync(d_global_write_index, 0, sizeof(size_type), 0) );

    const size_type probe_grid_size{(probe_table_num_rows + block_size -1)/block_size};
    
    // Do the probe of the hash table with the probe table and generate the output for the join
    probe_hash_table<base_join_type,
                     multimap_type,
                     hash_value_type,
                     size_type,
                     output_index_type,
                     block_size,
                     DEFAULT_CUDA_CACHE_SIZE>
    <<<probe_grid_size, block_size>>> (hash_table.get(),
                                       build_table,
                                       probe_table,
                                       probe_table.get_column_length(),
                                       output_l_ptr,
                                       output_r_ptr,
                                       d_global_write_index,
                                       estimated_join_output_size,
                                       flip_results);

    CUDA_TRY( cudaGetLastError() );

    CUDA_TRY( cudaMemcpy(&h_actual_found, d_global_write_index, sizeof(size_type), cudaMemcpyDeviceToHost));

    // The estimate was too small. Double the estimate and try again
    if(estimated_join_output_size < h_actual_found){
      cont = true;
      estimated_join_output_size *= 2;
      // Free the old buffers to prevent a memory leak on the new allocation
      RMM_TRY( RMM_FREE(output_l_ptr, 0) );
      RMM_TRY( RMM_FREE(output_r_ptr, 0) );
    }
    else
    {
      cont = false;
    }
  }

  // free memory used for the counters
  RMM_TRY( RMM_FREE(d_global_write_index, 0) );

  if (join_type == JoinType::FULL_JOIN) {
      append_full_join_indices(
              &output_l_ptr, &output_r_ptr,
              &estimated_join_output_size,
              &h_actual_found, build_table_num_rows);
  }

  // If the estimated join output size was larger than the actual output size,
  // then the buffers are larger than necessary. Allocate buffers of the actual
  // output size and copy the results to the buffers of the correct size
  // FIXME Is this really necessary? It's probably okay to have the buffers be oversized
  // and avoid the extra allocation/memcopy
  if (estimated_join_output_size > h_actual_found) {
      output_index_type *copy_output_l_ptr{nullptr};
      output_index_type *copy_output_r_ptr{nullptr};
      RMM_TRY( RMM_ALLOC((void**)&copy_output_l_ptr, h_actual_found*sizeof(output_index_type), 0) ); // TODO non-default stream?
      RMM_TRY( RMM_ALLOC((void**)&copy_output_r_ptr, h_actual_found*sizeof(output_index_type), 0) );
      CUDA_TRY( cudaMemcpy(copy_output_l_ptr, output_l_ptr, h_actual_found*sizeof(output_index_type), cudaMemcpyDeviceToDevice) );
      CUDA_TRY( cudaMemcpy(copy_output_r_ptr, output_r_ptr, h_actual_found*sizeof(output_index_type), cudaMemcpyDeviceToDevice) );
      RMM_TRY( RMM_FREE(output_l_ptr, 0) );
      RMM_TRY( RMM_FREE(output_r_ptr, 0) );
      output_l_ptr = copy_output_l_ptr;
      output_r_ptr = copy_output_r_ptr;
  }

  // Free the device error code
  CUDA_TRY( cudaFreeHost(d_gdf_error_code) );

  // Deduce the type of the output gdf_columns
  gdf_dtype dtype;
  switch(sizeof(output_index_type))
  {
    case 1 : dtype = GDF_INT8;  break;
    case 2 : dtype = GDF_INT16; break;
    case 4 : dtype = GDF_INT32; break;
    case 8 : dtype = GDF_INT64; break;
  }
  gdf_column_view(output_l, output_l_ptr, nullptr, h_actual_found, dtype);
  gdf_column_view(output_r, output_r_ptr, nullptr, h_actual_found, dtype);

  return gdf_error_code;
}
#endif
