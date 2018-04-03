/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                       \
{                                                                                                  \
    cudaError_t cudaStatus = call;                                                                 \
    if ( cudaSuccess != cudaStatus )                                                               \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);    \
}
#endif

#include "concurrent_unordered_multimap.cuh"
#include <cub/cub.cuh>

constexpr int warp_size = 32;

template<typename multimap_type>
__global__ void build_hash_tbl(
    multimap_type * const multi_map,
    const typename multimap_type::key_type* const build_tbl,
    const typename multimap_type::size_type build_tbl_size)
{
    const typename multimap_type::mapped_type i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < build_tbl_size ) {
      multi_map->insert( thrust::make_pair( build_tbl[i], i ) );
    }
}

template<
    typename multimap_type,
    typename key_type,
    typename size_type,
    typename joined_type,
    int block_size,
    int output_cache_size>
__global__ void probe_hash_tbl(
    multimap_type * multi_map,
    const key_type* probe_tbl,
    const size_type probe_tbl_size,
    joined_type * const joined,
    size_type* const current_idx,
    const size_type offset,
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

    int i = threadIdx.x + blockIdx.x * blockDim.x;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
    const int activemask = __ballot_sync(0xffffffff, i < probe_tbl_size);
#endif
    if ( i < probe_tbl_size ) {
        //OPT: merging equal_range and writing of output values could avoid redundant memory rountrips
        auto range = multi_map->equal_range(probe_tbl[i]);
        bool running = (range.first != range.second);
        auto it = range.first;
        while ( __any_sync( activemask, running ) )
        {
            if ( running )
            {
                if (!optimized && !key_compare( probe_tbl[i], it->first ) ) {
                    ++it;
                    running = (it != range.second);
                    continue;
                }
                joined_type joined_val;
                joined_val.x = it->second;
                joined_val.y = offset+i;

                int my_current_idx = atomicAdd( current_idx_shared+warp_id, 1 );
                //its guranteed to fit into the shared cache
                joined_shared[warp_id][my_current_idx] = joined_val;

                ++it;
                running = (it != range.second);
            }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
            __syncwarp(activemask);
#endif
            //flush output cache if next iteration does not fit
            if ( current_idx_shared[warp_id]+warp_size >= output_cache_size ) {
	        // count how many active threads participating here which could be less than warp_size
	        int num_threads = __popc(activemask);
                int output_offset = 0;
                if ( 0 == lane_id )
                    output_offset = atomicAdd( current_idx, current_idx_shared[warp_id] );
                output_offset = cub::ShuffleIndex(output_offset, 0, warp_size, activemask);

                for ( int shared_out_idx = lane_id; shared_out_idx<current_idx_shared[warp_id]; shared_out_idx+=num_threads ) {
                    joined[output_offset+shared_out_idx] = joined_shared[warp_id][shared_out_idx];
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
	    int num_threads = __popc(activemask);
            int output_offset = 0;
            if ( 0 == lane_id )
                output_offset = atomicAdd( current_idx, current_idx_shared[warp_id] );
            output_offset = cub::ShuffleIndex(output_offset, 0, warp_size, activemask);

            for ( int shared_out_idx = lane_id; shared_out_idx<current_idx_shared[warp_id]; shared_out_idx+=num_threads ) {
                joined[output_offset+shared_out_idx] = joined_shared[warp_id][shared_out_idx];
            }
        }
    }
}

