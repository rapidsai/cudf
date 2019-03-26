#include "parsing_utils.cuh"

#include <cuda_runtime.h>

#include <vector>

#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/error_utils.hpp"

constexpr size_t max_chunk_bytes = 256*1024*1024; // 256MB
constexpr int bytes_per_find_thread = 64;

/**---------------------------------------------------------------------------*
 * @brief Searches the input character array for each of characters in a set
 * and sums up the number of occurences.
 *
 * Does not load the entire buffer into the GPU memory at any time, so it can 
 * be used with buffers of any size.
 *
 * @param[in] h_data Pointer to the data in host memory
 * @param[in] h_size Size of the input data, in bytes
 * @param[in] keys Vector containing the keys to count in the buffer
 * @param[out] count Total number of occurences of all keys
 *
 * @return gdf_error
 *---------------------------------------------------------------------------**/
gdf_error countAllFromSet(const char *h_data, size_t h_size, std::vector<char> keys, gdf_size_type &count)
{
	const size_t chunk_count = (h_size + max_chunk_bytes - 1) / max_chunk_bytes;

	char* d_chunk = nullptr;
	RMM_TRY(RMM_ALLOC (&d_chunk, min(max_chunk_bytes, h_size), 0)); 

	for (size_t ci = 0; ci < chunk_count; ++ci) {
		const auto h_chunk = h_data + ci * max_chunk_bytes;
		const auto chunk_bytes = std::min((size_t)(h_size - ci * max_chunk_bytes), max_chunk_bytes);

		// Copy chunk to device
		CUDA_TRY(cudaMemcpyAsync(d_chunk, h_chunk, chunk_bytes, cudaMemcpyDefault));

		for (char key: keys) {
			count += thrust::count(rmm::exec_policy()->on(0), d_chunk, d_chunk + chunk_bytes, key);
		}
	}

	RMM_TRY( RMM_FREE(d_chunk, 0) );

	CUDA_TRY(cudaGetLastError());

	return GDF_SUCCESS;
}

/**---------------------------------------------------------------------------*
 * @brief CUDA kernel that finds all occurences of a character in the given 
 * character array. The positions are stored in the output array.
 * 
 * @param[in] data Pointer to the input character array
 * @param[in] size Number of bytes in the input array
 * @param[in] offset Offset to add to the output positions
 * @param[in] key Character to find in the array
 * @param[in,out] count Pointer to the number of found occurences
 * @param[out] positions Array containing the output positions
 * 
 * @return void
 *---------------------------------------------------------------------------**/
 __global__ void findAll(char *data, size_t size, size_t offset, const char key,
	gdf_size_type* count, ll_uint_t* positions) {

	// thread IDs range per block, so also need the block id
	const long tid = threadIdx.x + (blockDim.x * blockIdx.x);
	const long did = tid * bytes_per_find_thread;
	
	const char *raw = (data + did);

	const long byteToProcess = ((did + bytes_per_find_thread) < size) ?
									bytes_per_find_thread :
									(size - did);

	// Process the data
	for (long i = 0; i < byteToProcess; i++) {
		if (raw[i] == key) {
			const auto pos = atomicAdd(count, 1ull);
			positions[pos] = did + offset + i;
		}
	}
}

/**---------------------------------------------------------------------------*
 * @brief For each of the characters in the input set, find and saves all 
 * cccurences in the input host array of characters.
 * The positions are stored in the output device array.
 * 
 * Does not load the entire file into the GPU memory at any time, so it can 
 * be used to parse large files. Output array needs to be preallocated.
 * 
 * @param[in] h_data Pointer to the input character array
 * @param[in] h_size Number of bytes in the input array
 * @param[in] keys Vector containing the keys to count in the buffer
 * @param[in] result_offset Offset to add to the output positions
 * @param[out] positions Array containing the output positions
 * 
 * @return gdf_error with error code on failure, otherwise GDF_SUCCESS
 *---------------------------------------------------------------------------**/
gdf_error findAllFromSet(const char *h_data, size_t h_size, std::vector<char> keys, ll_uint_t result_offset,
	ll_uint_t *positions) {

	char* d_chunk = nullptr;
	RMM_TRY(RMM_ALLOC (&d_chunk, min(max_chunk_bytes, h_size), 0)); 
	
	gdf_size_type*	d_count;
	RMM_TRY(RMM_ALLOC((void**)&d_count, sizeof(gdf_size_type), 0) );
	CUDA_TRY(cudaMemsetAsync(d_count, 0ull, sizeof(gdf_size_type)));

	int blockSize;		// suggested thread count to use
	int minGridSize;	// minimum block count required
	CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, findAll) );

	const size_t chunk_count = (h_size + max_chunk_bytes - 1) / max_chunk_bytes;
	for (size_t ci = 0; ci < chunk_count; ++ci) {	
		const auto chunk_offset = ci * max_chunk_bytes;	
		const auto h_chunk = h_data + chunk_offset;
		const auto chunk_bytes = std::min((size_t)(h_size - ci * max_chunk_bytes), max_chunk_bytes);
		const auto chunk_bits = (chunk_bytes + bytes_per_find_thread - 1) / bytes_per_find_thread;
		const int gridSize = (chunk_bits + blockSize - 1) / blockSize;

		// Copy chunk to device
		CUDA_TRY(cudaMemcpyAsync(d_chunk, h_chunk, chunk_bytes, cudaMemcpyDefault));

		for (char key: keys) {
			findAll <<< gridSize, blockSize >>> (
				d_chunk, chunk_bytes, chunk_offset + result_offset, key,
				d_count, positions);
		}
	}

	gdf_size_type h_count = 0;
	CUDA_TRY(cudaMemcpy(&h_count, d_count, sizeof(gdf_size_type), cudaMemcpyDefault));
	thrust::sort(rmm::exec_policy()->on(0), positions, positions + h_count);

	RMM_TRY(RMM_FREE(d_count, 0)); 
	RMM_TRY(RMM_FREE(d_chunk, 0));

	CUDA_TRY(cudaGetLastError());

	return GDF_SUCCESS;
}