#ifndef GDF_UTILS_H
#define GDF_UTILS_H

#include "cudf.h"

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE_CALLABLE __host__ __device__ __forceinline__
#define CUDA_DEVICE_CALLABLE __device__ __forceinline__
#define CUDA_LAUNCHABLE __global__
#else
#define CUDA_HOST_DEVICE_CALLABLE
#define CUDA_DEVICE_CALLABLE
#define CUDA_LAUNCHABLE
#endif


CUDA_HOST_DEVICE_CALLABLE 
bool gdf_is_valid(const gdf_valid_type *valid, gdf_index_type pos) {
	if ( nullptr != valid )
		return (valid[pos / GDF_VALID_BITSIZE] >> (pos % GDF_VALID_BITSIZE)) & 1;
	else
		return true;
}

CUDA_HOST_DEVICE_CALLABLE
gdf_size_type gdf_get_num_chars_bitmask(gdf_size_type size) { 
	return (( size + ( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE ); 
}

#endif
