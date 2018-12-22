#ifndef GDF_UTILS_H
#define GDF_UTILS_H

#include "cudf.h"
#include "division.hpp"

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
	if ( valid )
		return (valid[pos / GDF_VALID_BITSIZE] >> (pos % GDF_VALID_BITSIZE)) & 1;
	else
		return true;
}


/**
 * Calculates the size in bytes of a valid mask for a given column's size.
 *
 * @note This is the size in bytes of a valid mask, where the individual
 * mask elements are of size `sizeof(gdf_valid_type)`.
 *
 * @param[in] column_size the number of elements (rows) in the column
 * @return the size in bytes of the valid mask required for the specified column size
 */
CUDA_HOST_DEVICE_CALLABLE
gdf_size_type get_number_of_bytes_for_valid(gdf_size_type column_size) {
    return gdf::util::div_round_up_safe(column_size, GDF_VALID_BITSIZE);
}

#endif
