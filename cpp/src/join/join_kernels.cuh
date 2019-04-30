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
#ifndef JOIN_KERNELS_CUH
#define JOIN_KERNELS_CUH

constexpr int JoinNoneValue = -1;

enum class JoinType {
  INNER_JOIN,
  LEFT_JOIN,
  FULL_JOIN
};

#include "cudf.h"
#include <table/device_table.cuh>
#include "hash/concurrent_unordered_multimap.cuh"
#include "hash/hash_functions.cuh"
#include "utilities/bit_util.cuh"

#include <cub/cub.cuh>

 constexpr int warp_size = 32;

/* --------------------------------------------------------------------------*/
/** 
* @brief  Builds a hash table from a device_table that maps the hash values of 
  each row to its respective row index.
* 
* @param[in,out] multi_map The hash table to be built to insert rows into
* @param[in] build_table The table to build the hash table on
* @param[in] build_table_num_rows The number of rows in the build table
* @tparam multimap_type The type of the hash table
* 
*/
/* ----------------------------------------------------------------------------*/
template<typename multimap_type>
__global__ void build_hash_table( multimap_type * const multi_map,
                                  device_table build_table,
                                  const gdf_size_type build_table_num_rows,
                                  gdf_error * gdf_error_code)
{
    gdf_size_type i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < build_table_num_rows) {
      // Compute the hash value of this row
      const hash_value_type row_hash_value{hash_row(build_table,i)};

      // Insert the (row hash value, row index) into the map
      // using the row hash value to determine the location in the
      // hash map where the new pair should be inserted
      const auto insert_location = multi_map->insert(
          thrust::make_pair(row_hash_value, i), true, row_hash_value);

      // If the insert failed, set the error code accordingly
      if (multi_map->end() == insert_location) {
        *gdf_error_code = GDF_HASH_TABLE_INSERT_FAILURE;
      }
      i += blockDim.x * gridDim.x;
    }
}

/* --------------------------------------------------------------------------*/
/** 
* @brief  Adds a pair of indices to the shared memory cache
* 
* @param[in] first The first index in the pair
* @param[in] second The second index in the pair
* @param[in,out] current_idx_shared Pointer to shared index that determines where in the shared
memory cache the pair will be written
* @param[in] warp_id The ID of the warp of the calling the thread
* @param[out] joined_shared_l Pointer to the shared memory cache for left indices
* @param[out] joined_shared_r Pointer to the shared memory cache for right indices
* 
*/
/* ----------------------------------------------------------------------------*/
template<typename size_type, typename output_index_type>
__inline__ __device__ void add_pair_to_cache(const output_index_type first, 
                                             const output_index_type second, 
                                             size_type *current_idx_shared, 
                                             const int warp_id, 
                                             output_index_type *joined_shared_l,
                                             output_index_type *joined_shared_r)
{
  size_type my_current_idx{atomicAdd(current_idx_shared + warp_id, size_type(1))};

  // its guaranteed to fit into the shared cache
  joined_shared_l[my_current_idx] = first;
  joined_shared_r[my_current_idx] = second;
}

/* --------------------------------------------------------------------------*/
/** 
* @brief  Computes the output size of joining the probe table to the build table
  by probing the hash map with the probe table and counting the number of matches.
* 
* @param[in] multi_map The hash table built on the build table
* @param[in] build_table The build table
* @param[in] probe_table The probe table
* @param[in] probe_table_num_rows The number of rows in the probe table
* @param[out] output_size The resulting output size
  @tparam join_type The type of join to be performed
  @tparam multimap_type The datatype of the hash table
  @tparam block_size The number of threads in a thread block for the kernel
  @tparam output_cache_size The size of the shared memory cache for caching the join output results
* 
*/
/* ----------------------------------------------------------------------------*/
template< JoinType join_type,
          typename multimap_type,
          int block_size,
          int output_cache_size>
__global__ void compute_join_output_size( multimap_type const * const multi_map,
                                          device_table build_table,
                                          device_table probe_table,
                                          const gdf_size_type probe_table_num_rows,
                                          gdf_size_type* output_size)
{

  __shared__ gdf_size_type block_counter;
  block_counter=0;
  __syncthreads();

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
  __syncwarp();
#endif

  gdf_size_type probe_row_index = threadIdx.x + blockIdx.x * blockDim.x;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
  const unsigned int activemask = __ballot_sync(0xffffffff, probe_row_index < probe_table_num_rows);
#endif
  if ( probe_row_index < probe_table_num_rows ) {
    const auto unused_key = multi_map->get_unused_key();
    const auto end = multi_map->end();
    auto found = end;

    // Search the hash map for the hash value of the probe row using the row's
    // hash value to determine the location where to search for the row in the hash map
    hash_value_type probe_row_hash_value{0};
    // Search the hash map for the hash value of the probe row
    probe_row_hash_value = hash_row(probe_table,probe_row_index);
    found = multi_map->find(probe_row_hash_value, true, probe_row_hash_value);

    // for left-joins we always need to add an output
    bool running = (join_type == JoinType::LEFT_JOIN) || (end != found); 
    bool found_match = false;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
    while ( __any_sync( activemask, running ) )
#else
      while ( __any( running ) )
#endif
      {
        if ( running )
        {
          // TODO Simplify this logic...

          // Left joins always have an entry in the output
          if (join_type == JoinType::LEFT_JOIN && (end == found)) {
            running = false;    
          }
          // Stop searching after encountering an empty hash table entry
          else if ( unused_key == found->first ) {
            running = false;
          }
          // First check that the hash values of the two rows match
          else if (found->first == probe_row_hash_value)
          {
            // If the hash values are equal, check that the rows are equal
            if( rows_equal(probe_table, probe_row_index, build_table, found->second) )
            {
              // If the rows are equal, then we have found a true match
              found_match = true;
              atomicAdd(&block_counter,1) ;
            }
            // Continue searching for matching rows until you hit an empty hash map entry
            ++found;
            // If you hit the end of the hash map, wrap around to the beginning
            if(end == found)
              found = multi_map->begin();
            // Next entry is empty, stop searching
            if(unused_key == found->first)
              running = false;
          }
          else 
          {
            // Continue searching for matching rows until you hit an empty hash table entry
            ++found;
            // If you hit the end of the hash map, wrap around to the beginning
            if(end == found)
              found = multi_map->begin();
            // Next entry is empty, stop searching
            if(unused_key == found->first)
              running = false;
          }

          if ((join_type == JoinType::LEFT_JOIN) && (!running) && (!found_match)) {
            atomicAdd(&block_counter,1);
          }
        }
      }
  }

  __syncthreads();

  // Add block counter to global counter
  if (threadIdx.x==0)
    atomicAdd(output_size, block_counter);
}


/* --------------------------------------------------------------------------*/
/** 
 * @brief  Probes the hash map with the probe table to find all matching rows 
 between the probe and hash table and generate the output for the desired Join operation.
 * 
 * @param[in] multi_map The hash table built from the build table
 * @param[in] build_table The build table
 * @param[in] probe_table The probe table
 * @param[in] probe_table_num_rows The length of the columns in the probe table
 * @param[out] join_output_l The left result of the join operation
 * @param[out] join_output_r The right result of the join operation
 * @param[in,out] current_idx A global counter used by threads to coordinate writes to the global output
 * @param[in] max_size The maximum size of the output
 * @param[in] offset An optional offset
 * @tparam join_type The type of join to be performed
 * @tparam multimap_type The type of the hash table
 * @tparam output_index_type The datatype used for the indices in the output arrays
 * @tparam block_size The number of threads per block for this kernel
 * @tparam output_cache_size The side of the shared memory buffer to cache join output results
 * 
 */
/* ----------------------------------------------------------------------------*/
template< JoinType join_type,
          typename multimap_type,
          typename key_type,
          typename output_index_type,
          gdf_size_type block_size,
          gdf_size_type output_cache_size>
__global__ void probe_hash_table( multimap_type const * const multi_map,
                                  device_table build_table,
                                  device_table probe_table,
                                  const gdf_size_type probe_table_num_rows,
                                  output_index_type * join_output_l,
                                  output_index_type * join_output_r,
                                  gdf_size_type* current_idx,
                                  const gdf_size_type max_size,
                                  bool flip_results,
                                  const output_index_type offset = 0)
{
  constexpr int num_warps = block_size/warp_size;
  __shared__ gdf_size_type current_idx_shared[num_warps];
  __shared__ output_index_type join_shared_l[num_warps][output_cache_size];
  __shared__ output_index_type join_shared_r[num_warps][output_cache_size];
  output_index_type *output_l = join_output_l, *output_r = join_output_r;

  if (flip_results) {
      output_l = join_output_r;
      output_r = join_output_l;
  }

  const int warp_id = threadIdx.x/warp_size;
  const int lane_id = threadIdx.x%warp_size;

  if ( 0 == lane_id )
  {
    current_idx_shared[warp_id] = 0;
  }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
  __syncwarp();
#endif

  output_index_type probe_row_index = threadIdx.x + blockIdx.x * blockDim.x;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
  const unsigned int activemask = __ballot_sync(0xffffffff, probe_row_index < probe_table_num_rows);
#endif
  if ( probe_row_index < probe_table_num_rows ) {

    const auto unused_key = multi_map->get_unused_key();
    const auto end = multi_map->end();  
    auto found = end;    

    // Search the hash map for the hash value of the probe row using the row's
    // hash value to determine the location where to search for the row in the hash map

    // Only probe the hash table if the probe row is valid
    hash_value_type probe_row_hash_value{0};
    // Search the hash map for the hash value of the probe row
    probe_row_hash_value = hash_row(probe_table,probe_row_index);
    found = multi_map->find(probe_row_hash_value, true, probe_row_hash_value);

    bool running = (join_type == JoinType::LEFT_JOIN) || (end != found);	// for left-joins we always need to add an output
    bool found_match = false;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
    while ( __any_sync( activemask, running ) )
#else
    while ( __any( running ) )
#endif
    {
      if ( running )
      {
        // TODO Simplify this logic...

        // Left joins always have an entry in the output
        if (join_type == JoinType::LEFT_JOIN && (end == found)) {
          running = false;    
        }
        // Stop searching after encountering an empty hash table entry
        else if ( unused_key == found->first ) {
          running = false;
        }
        // First check that the hash values of the two rows match
        else if (found->first == probe_row_hash_value)
        {
          // If the hash values are equal, check that the rows are equal
          if( rows_equal(probe_table, probe_row_index, build_table, found->second) )
          {

            // If the rows are equal, then we have found a true match
            found_match = true;
            const output_index_type probe_index{offset + probe_row_index};
            add_pair_to_cache(probe_index, found->second, current_idx_shared, warp_id, join_shared_l[warp_id], join_shared_r[warp_id]);
          }
          // Continue searching for matching rows until you hit an empty hash map entry
          ++found;
          // If you hit the end of the hash map, wrap around to the beginning
          if(end == found)
            found = multi_map->begin();
          // Next entry is empty, stop searching
          if(unused_key == found->first)
            running = false;
        }
        else 
        {
          // Continue searching for matching rows until you hit an empty hash table entry
          ++found;
          // If you hit the end of the hash map, wrap around to the beginning
          if(end == found)
            found = multi_map->begin();
          // Next entry is empty, stop searching
          if(unused_key == found->first)
            running = false;
        }

        // If performing a LEFT join and no match was found, insert a Null into the output
        if ((join_type == JoinType::LEFT_JOIN) && (!running) && (!found_match)) {
          const output_index_type probe_index{offset + probe_row_index};
          add_pair_to_cache(probe_index, static_cast<output_index_type>(JoinNoneValue), current_idx_shared, warp_id, join_shared_l[warp_id], join_shared_r[warp_id]);
        }
      }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
      __syncwarp(activemask);
#endif
      //flush output cache if next iteration does not fit
      if ( current_idx_shared[warp_id] + warp_size >= output_cache_size ) 
      {

        // count how many active threads participating here which could be less than warp_size
#if defined(CUDA_VERSION) && CUDA_VERSION < 9000
        const unsigned int activemask = __ballot(1);
#endif
        int num_threads = __popc(activemask);
        gdf_size_type output_offset = 0;

        if ( 0 == lane_id )
        {
          output_offset = atomicAdd( current_idx, current_idx_shared[warp_id] );
        }

        output_offset = cub::ShuffleIndex(output_offset, 0, warp_size, activemask);

        for ( int shared_out_idx = lane_id; shared_out_idx<current_idx_shared[warp_id]; shared_out_idx+=num_threads ) 
        {
          gdf_size_type thread_offset = output_offset + shared_out_idx;
          if (thread_offset < max_size) {
            output_l[thread_offset] = join_shared_l[warp_id][shared_out_idx];
            output_r[thread_offset] = join_shared_r[warp_id][shared_out_idx];
          }
        }
#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
        __syncwarp(activemask);
#endif
        if ( 0 == lane_id )
        {
          current_idx_shared[warp_id] = 0;
        }
#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
        __syncwarp(activemask);
#endif
      }
    }

    //final flush of output cache
    if ( current_idx_shared[warp_id] > 0 ) 
    {
      // count how many active threads participating here which could be less than warp_size
#if defined(CUDA_VERSION) && CUDA_VERSION < 9000
      const unsigned int activemask = __ballot(1);
#endif
      int num_threads = __popc(activemask);
      gdf_size_type output_offset = 0;
      if ( 0 == lane_id )
      {
        output_offset = atomicAdd( current_idx, current_idx_shared[warp_id] );
      }
        
      output_offset = cub::ShuffleIndex(output_offset, 0, warp_size, activemask);

      for ( int shared_out_idx = lane_id; shared_out_idx<current_idx_shared[warp_id]; shared_out_idx+=num_threads ) 
      {
        gdf_size_type thread_offset = output_offset + shared_out_idx;
        if (thread_offset < max_size) 
        {
          output_l[thread_offset] = join_shared_l[warp_id][shared_out_idx];
          output_r[thread_offset] = join_shared_r[warp_id][shared_out_idx];
        }
      }
    }
  }
}

/*
   // TODO This kernel still needs to be updated to work with an arbitrary number of columns
template<
    typename multimap_type,
    typename key_type,
    typename size_type,
    typename join_output_pair,
    int block_size>
__global__ void probe_hash_table_uniq_keys(
    multimap_type * multi_map,
    const key_type* probe_table,
    const size_type probe_table_num_rows,
    join_output_pair * const joined,
    size_type* const current_idx,
    const size_type offset)
{
    __shared__ int current_idx_shared;
    __shared__ size_type output_offset_shared;
    __shared__ join_output_pair joined_shared[block_size];
    if ( 0 == threadIdx.x ) {
        output_offset_shared = 0;
        current_idx_shared = 0;
    }
    
    __syncthreads();

    size_type i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < probe_table_num_rows ) {
        const auto end = multi_map->end();
        auto found = multi_map->find(probe_table[i]);
        if ( end != found ) {
            join_output_pair joined_val;
            joined_val.first = offset+i;
            joined_val.second = found->second;
            int my_current_idx = atomicAdd( &current_idx_shared, 1 );
            //its guranteed to fit into the shared cache
            joined_shared[my_current_idx] = joined_val;
        }
    }
    
    __syncthreads();
    
    if ( current_idx_shared > 0 ) {
        if ( 0 == threadIdx.x ) {
            output_offset_shared = atomicAdd( current_idx, current_idx_shared );
        }
        __syncthreads();
        
        if ( threadIdx.x < current_idx_shared ) {
            joined[output_offset_shared+threadIdx.x] = joined_shared[threadIdx.x];
        }
    }
}

*/
#endif
