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

constexpr int JoinNoneValue = -1;

enum class JoinType {
  INNER_JOIN,
  LEFT_JOIN,
};

#include "../../hashmap/concurrent_unordered_multimap.cuh"
#include <cub/cub.cuh>

constexpr int warp_size = 32;

template<typename multimap_type>
__global__ void build_hash_tbl(
    multimap_type * const multi_map,
    const typename multimap_type::key_type* const build_tbl,
    const typename multimap_type::size_type build_tbl_size)
{

    using mapped_type = typename multimap_type::mapped_type;
    
    const typename multimap_type::size_type i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < build_tbl_size ) {
      multi_map->insert( thrust::make_pair( build_tbl[i], (mapped_type) i ) );
    }
}

template<typename size_type,
	 typename joined_type>
__inline__ __device__ void add_pair_to_cache(const size_type first, const size_type second, int *current_idx_shared, const int warp_id, joined_type *joined_shared)
{
  joined_type joined_val;
  joined_val.first = first;
  joined_val.second = second;

  int my_current_idx = atomicAdd(current_idx_shared + warp_id, 1);

  // its guaranteed to fit into the shared cache
  joined_shared[my_current_idx] = joined_val;
}

template<
JoinType join_type,
		 typename multimap_type,
		 typename key_type,
		 typename key2_type,
		 typename key3_type,
		 typename size_type,
		 int block_size,
  int output_cache_size>
__global__ void probe_hash_tbl_count_common(
	multimap_type * multi_map,
	const key_type* probe_tbl,
	const size_type probe_tbl_size,
	const key2_type* probe_col2, const key2_type* build_col2,
	const key3_type* probe_col3, const key3_type* build_col3,
    size_type* globalCounterFound
    )
{
 
    typedef typename multimap_type::key_equal key_compare_type;

    __shared__ int countFound;
    countFound=0;
    __syncthreads();

    key_compare_type key_compare;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
    __syncwarp();
#endif

    size_type i = threadIdx.x + blockIdx.x * blockDim.x;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
    const unsigned int activemask = __ballot_sync(0xffffffff, i < probe_tbl_size);
#endif
    if ( i < probe_tbl_size ) {
    	const auto unused_key = multi_map->get_unused_key();
    	const auto end = multi_map->end();
    	const key_type probe_key = probe_tbl[i];
    	auto it = multi_map->find(probe_key);

    	bool running = (join_type == JoinType::LEFT_JOIN) || (end != it); // for left-joins we always need to add an output
    	bool found_match = false;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
        while ( __any_sync( activemask, running ) )
#else
        while ( __any( running ) )
#endif
        {
    		if ( running )
    		{
                if (join_type == JoinType::LEFT_JOIN && (end == it)) {
                    running = false;    // add once on the first iteration
                }
                else if ( key_compare( unused_key, it->first ) ) {
                    running = false;
                }
                else if (!key_compare( probe_key, it->first ) ||
                    ((probe_col2 != NULL) && (probe_col2[i] != build_col2[it->second])) ||
                    ((probe_col3 != NULL) && (probe_col3[i] != build_col3[it->second]))) {
                    ++it;
                    running = (end != it);
                }
                else {
					atomicAdd(&countFound,1) ;
                    ++it;
                    running = (end != it);
                    found_match = true;
                }

                if ((join_type == JoinType::LEFT_JOIN) && (!running) && (!found_match)) {
                    atomicAdd(&countFound,1);
                }
		  }
	   }
    }
    __syncthreads();
    if (threadIdx.x==0)
        atomicAdd(globalCounterFound,countFound);


}



template<
    JoinType join_type,
    typename multimap_type,
    typename key_type,
    typename key2_type,
    typename key3_type,
    typename size_type,
    typename joined_type,
    int block_size,
    int output_cache_size>
__global__ void probe_hash_tbl(
    multimap_type * multi_map,
    const key_type* probe_tbl,
    const size_type probe_tbl_size,
    const key2_type* probe_col2, const key2_type* build_col2,
    const key3_type* probe_col3, const key3_type* build_col3,
    joined_type * const joined,
    size_type* const current_idx,
    const size_type max_size,
    const size_type offset = 0,
    const bool optimized = false)
{
    typedef typename multimap_type::key_equal key_compare_type;
    __shared__ int current_idx_shared[block_size/warp_size];
    __shared__ joined_type joined_shared[block_size/warp_size][output_cache_size];

    const int warp_id = threadIdx.x/warp_size;
    const int lane_id = threadIdx.x%warp_size;

    key_compare_type key_compare;

    if ( 0 == lane_id )
        current_idx_shared[warp_id] = 0;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
    __syncwarp();
#endif

    size_type i = threadIdx.x + blockIdx.x * blockDim.x;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
    const unsigned int activemask = __ballot_sync(0xffffffff, i < probe_tbl_size);
#endif
    if ( i < probe_tbl_size ) {
        const auto unused_key = multi_map->get_unused_key();
        const auto end = multi_map->end();
        const key_type probe_key = probe_tbl[i];
        auto it = multi_map->find(probe_key);

        bool running = (join_type == JoinType::LEFT_JOIN) || (end != it);	// for left-joins we always need to add an output
	bool found_match = false;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
        while ( __any_sync( activemask, running ) )
#else
        while ( __any( running ) )
#endif
        {
            if ( running )
            {
		if (join_type == JoinType::LEFT_JOIN && (end == it)) {
		    running = false;	// add once on the first iteration
		}
		else if ( key_compare( unused_key, it->first ) ) {
                    running = false;
                }
                else if (!key_compare( probe_key, it->first ) ||
			 ((probe_col2 != NULL) && (probe_col2[i] != build_col2[it->second])) ||
			 ((probe_col3 != NULL) && (probe_col3[i] != build_col3[it->second]))) {
                    ++it;
                    running = (end != it);
                }
                else {
		    add_pair_to_cache(offset+i, it->second, current_idx_shared, warp_id, joined_shared[warp_id]);

                    ++it;
                    running = (end != it);
		    found_match = true;
                }
		if ((join_type == JoinType::LEFT_JOIN) && (!running) && (!found_match)) {
		  add_pair_to_cache(offset+i, JoinNoneValue, current_idx_shared, warp_id, joined_shared[warp_id]);
		}
            }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
            __syncwarp(activemask);
#endif
            //flush output cache if next iteration does not fit
            if ( current_idx_shared[warp_id]+warp_size >= output_cache_size ) {
                // count how many active threads participating here which could be less than warp_size
#if defined(CUDA_VERSION) && CUDA_VERSION < 9000
                const unsigned int activemask = __ballot(1);
#endif
                int num_threads = __popc(activemask);
                unsigned long long int output_offset = 0;
                if ( 0 == lane_id )
                    output_offset = atomicAdd( current_idx, current_idx_shared[warp_id] );
                output_offset = cub::ShuffleIndex(output_offset, 0, warp_size, activemask);

                for ( int shared_out_idx = lane_id; shared_out_idx<current_idx_shared[warp_id]; shared_out_idx+=num_threads ) {
                    size_type thread_offset = output_offset + shared_out_idx;
                    if (thread_offset < max_size)
		       joined[thread_offset] = joined_shared[warp_id][shared_out_idx];
                }
#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
                __syncwarp(activemask);
#endif
                if ( 0 == lane_id )
                    current_idx_shared[warp_id] = 0;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
                __syncwarp(activemask);
#endif
            }
        }

        //final flush of output cache
        if ( current_idx_shared[warp_id] > 0 ) {
            // count how many active threads participating here which could be less than warp_size
#if defined(CUDA_VERSION) && CUDA_VERSION < 9000
            const unsigned int activemask = __ballot(1);
#endif
            int num_threads = __popc(activemask);
            unsigned long long int output_offset = 0;
            if ( 0 == lane_id )
                output_offset = atomicAdd( current_idx, current_idx_shared[warp_id] );
            output_offset = cub::ShuffleIndex(output_offset, 0, warp_size, activemask);

            for ( int shared_out_idx = lane_id; shared_out_idx<current_idx_shared[warp_id]; shared_out_idx+=num_threads ) {
                size_type thread_offset = output_offset + shared_out_idx;
		if (thread_offset < max_size)
                   joined[thread_offset] = joined_shared[warp_id][shared_out_idx];
            }
        }
    }
}

template<
    typename multimap_type,
    typename key_type,
    typename size_type,
    typename joined_type,
    int block_size>
__global__ void probe_hash_tbl_uniq_keys(
    multimap_type * multi_map,
    const key_type* probe_tbl,
    const size_type probe_tbl_size,
    joined_type * const joined,
    size_type* const current_idx,
    const size_type offset)
{
    __shared__ int current_idx_shared;
    __shared__ size_type output_offset_shared;
    __shared__ joined_type joined_shared[block_size];
    if ( 0 == threadIdx.x ) {
        output_offset_shared = 0;
        current_idx_shared = 0;
    }
    
    __syncthreads();

    size_type i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < probe_tbl_size ) {
        const auto end = multi_map->end();
        auto it = multi_map->find(probe_tbl[i]);
        if ( end != it ) {
            joined_type joined_val;
            joined_val.first = offset+i;
            joined_val.second = it->second;
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

