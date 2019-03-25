#include "parsing_utils.cuh"

#include <cuda_runtime.h>

#include <vector>

#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/error_utils.hpp"

/**---------------------------------------------------------------------------*
 * @brief Sums up the number of occurences of characters from a set
 *
 * Does not load the entire buffer into the GPU memory at any time, so it can 
 * be used with buffers of any size.
 *
 * @param[in] h_data Pointer to the csv data in host memory
 * @param[in] h_size Size of the input data, in bytes
 * @param[in] keys Vector containing the keys to count in the buffer
 * @param[in,out] raw_csv Structure containing the csv parsing parameters
 * and intermediate results
 *
 * @return gdf_error
 *---------------------------------------------------------------------------**/
gdf_error countAllFromSet(const char *h_data, size_t h_size, std::vector<char> keys, gdf_size_type &rec_cnt)
{
	const size_t chunk_count = (h_size + max_chunk_bytes - 1) / max_chunk_bytes;

	char* d_chunk = nullptr;
	RMM_TRY(RMM_ALLOC (&d_chunk, min(max_chunk_bytes, h_size), 0)); 

	for (size_t ci = 0; ci < chunk_count; ++ci) {
		const auto h_chunk = h_data + ci * max_chunk_bytes;
		const auto chunk_bytes = std::min((size_t)(h_size - ci * max_chunk_bytes), max_chunk_bytes);

		// Copy chunk to device
		CUDA_TRY(cudaMemcpyAsync(d_chunk, h_chunk, chunk_bytes, cudaMemcpyDefault, (cudaStream_t)0));

		for (char key: keys) {
			rec_cnt += thrust::count(rmm::exec_policy()->on(0), d_chunk, d_chunk + chunk_bytes, key);
		}
	}

	RMM_TRY( RMM_FREE(d_chunk, 0) );

	CUDA_TRY(cudaGetLastError());

	return GDF_SUCCESS;
}