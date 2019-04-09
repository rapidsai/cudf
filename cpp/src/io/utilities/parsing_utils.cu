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

/**
 * @file parsing_utils.cu Utility functions for parsing plain-text files
 *
 */


#include "parsing_utils.cuh"

#include <cuda_runtime.h>

#include <vector>
#include <memory>
#include <iostream>

#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/error_utils.hpp"

// When processing the input in chunks, this is the maximum size of each chunk.
// Only one chunk is loaded on the GPU at a time, so this value is chosen to
// be small enough to fit on the GPU in most cases.
constexpr size_t max_chunk_bytes = 256*1024*1024; // 256MB

constexpr int bytes_per_find_thread = 64;

using pos_key_pair = thrust::pair<uint64_t,char>;

template <typename T>
struct rmm_deleter {
 void operator()(T *ptr) { RMM_FREE(ptr, 0); }
};
template <typename T>
using device_ptr = std::unique_ptr<T, rmm_deleter<T>>;

/**---------------------------------------------------------------------------*
 * @brief Sets the specified element of the array to the passed value
 *---------------------------------------------------------------------------**/
template<class T, class V>
__device__ __forceinline__
void setElement(T* array, gdf_size_type idx, const T& t, const V& v){
	array[idx] = t;
}

/**---------------------------------------------------------------------------*
 * @brief Sets the specified element of the array of pairs using the two passed
 * parameters.
 *---------------------------------------------------------------------------**/
template<class T, class V>
__device__ __forceinline__
void setElement(thrust::pair<T, V>* array, gdf_size_type idx, const T& t, const V& v) {
	array[idx] = {t, v};
}

/**---------------------------------------------------------------------------*
 * @brief Overloads the setElement() functions for void* arrays.
 * Does not do anything, indexing is not allowed with void* arrays.
 *---------------------------------------------------------------------------**/
template<class T, class V>
__device__ __forceinline__
void setElement(void* array, gdf_size_type idx, const T& t, const V& v) {
}

/**---------------------------------------------------------------------------*
 * @brief CUDA kernel that finds all occurrences of a character in the given 
 * character array. If the 'positions' parameter is not void*,
 * positions of all occurrences are stored in the output array.
 * 
 * @param[in] data Pointer to the input character array
 * @param[in] size Number of bytes in the input array
 * @param[in] offset Offset to add to the output positions
 * @param[in] key Character to find in the array
 * @param[in,out] count Pointer to the number of found occurrences
 * @param[out] positions Array containing the output positions
 * 
 * @return void
 *---------------------------------------------------------------------------**/
template<class T>
 __global__ 
 void countAndSetPositions(char *data, uint64_t size, uint64_t offset, const char key, gdf_size_type* count,
	T* positions) {

	// thread IDs range per block, so also need the block id
	const uint64_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
	const uint64_t did = tid * bytes_per_find_thread;
	
	const char *raw = (data + did);

	const long byteToProcess = ((did + bytes_per_find_thread) < size) ?
									bytes_per_find_thread :
									(size - did);

	// Process the data
	for (long i = 0; i < byteToProcess; i++) {
		if (raw[i] == key) {
			const auto idx = atomicAdd(count, (gdf_size_type)1);
			setElement(positions, idx, did + offset + i, key);
		}
	}
}

/**---------------------------------------------------------------------------*
 * @brief Searches the input character array for each of characters in a set.
 * Sums up the number of occurrences. If the 'positions' parameter is not void*,
 * positions of all occurrences are stored in the output device array.
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
 * @return gdf_size_type total number of occurrences
 *---------------------------------------------------------------------------**/
template<class T>
gdf_size_type findAllFromSet(const char *h_data, size_t h_size, const std::vector<char>& keys, uint64_t result_offset,
	T *positions) {

	char* d_chunk = nullptr;
	RMM_TRY(RMM_ALLOC (&d_chunk, min(max_chunk_bytes, h_size), 0));
	device_ptr<char> chunk_deleter(d_chunk);

	gdf_size_type*	d_count;
	RMM_TRY(RMM_ALLOC((void**)&d_count, sizeof(gdf_size_type), 0) );
	device_ptr<gdf_size_type> count_deleter(d_count);
	CUDA_TRY(cudaMemsetAsync(d_count, 0ull, sizeof(gdf_size_type)));

	int blockSize;		// suggested thread count to use
	int minGridSize;	// minimum block count required
	CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, countAndSetPositions<T>) );

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
			countAndSetPositions<T> <<< gridSize, blockSize >>> (
				d_chunk, chunk_bytes, chunk_offset + result_offset, key,
				d_count, positions);
		}
	}

	gdf_size_type h_count = 0;
	CUDA_TRY(cudaMemcpy(&h_count, d_count, sizeof(gdf_size_type), cudaMemcpyDefault));
	return h_count;
}

/**---------------------------------------------------------------------------*
 * @brief Searches the input character array for each of characters in a set
 * and sums up the number of occurrences.
 *
 * Does not load the entire buffer into the GPU memory at any time, so it can 
 * be used with buffers of any size.
 *
 * @param[in] h_data Pointer to the data in host memory
 * @param[in] h_size Size of the input data, in bytes
 * @param[in] keys Vector containing the keys to count in the buffer
 *
 * @return gdf_size_type total number of occurrences
 *---------------------------------------------------------------------------**/
gdf_size_type countAllFromSet(const char *h_data, size_t h_size, const std::vector<char>& keys) {
	return findAllFromSet<void>(h_data, h_size, keys, 0, nullptr);
 }

template gdf_size_type findAllFromSet<uint64_t>(const char *h_data, size_t h_size, const std::vector<char>& keys, uint64_t result_offset,
	uint64_t *positions);

template gdf_size_type findAllFromSet<pos_key_pair>(const char *h_data, size_t h_size, const std::vector<char>& keys, uint64_t result_offset,
	pos_key_pair *positions);

struct BlockSumArray {
		int16_t* d_sums = nullptr;
		uint64_t length;
		uint64_t block_size;
		BlockSumArray(uint64_t len, uint64_t bsize): length(len), block_size(bsize){}
};

class BlockSumPyramid {
	const uint16_t aggregation_rate = 32;
	std::vector<BlockSumArray> levels;
	BlockSumArray* d_levels;

public:
	BlockSumPyramid(int count){
		levels.emplace_back(count/aggregation_rate, aggregation_rate);
		RMM_ALLOC(&levels.back().d_sums, levels.back().length*sizeof(BlockSumArray), 0);

		while (levels.back().length >= aggregation_rate) {
			const auto& prev_level = levels.back();
			levels.emplace_back(prev_level.length/aggregation_rate, prev_level.block_size*aggregation_rate);
			RMM_ALLOC(&levels.back().d_sums, levels.back().length*sizeof(BlockSumArray), 0);
		}
	
		RMM_ALLOC(&d_levels, levels.size()*sizeof(BlockSumArray), 0);
		cudaMemcpyAsync(d_levels, levels.data(), levels.size()*sizeof(BlockSumArray), cudaMemcpyDefault);
	}

	auto operator[](int lvl) const {return levels[lvl];}
	auto deviceGetLevels() const noexcept {return d_levels;}
	size_t getHeigth() const noexcept {return levels.size();}
	constexpr int getAggregationRate() const noexcept {return aggregation_rate;}

	~BlockSumPyramid() {
		for (auto& lvl: levels) {
			RMM_FREE(lvl.d_sums, 0);
		}
		RMM_FREE(d_levels, 0);
	}
};


__global__
void sumBracketsKernel(
	pos_key_pair* brackets, int bracket_count,
	char open_bracket, char closed_bracket,
	BlockSumArray sum_array) {
	const uint64_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
	const uint64_t did = tid * sum_array.block_size;


	if (tid >= sum_array.length)
		return;

	auto* start = brackets + did;
	int16_t csum = 0;
	for (int i = 0; i < sum_array.block_size; ++i) {
		if ((start + i)->second == open_bracket) ++csum;
		if ((start + i)->second == closed_bracket) --csum;
	}
	sum_array.d_sums[tid] = csum;
}

void sumBrackets(
	pos_key_pair* brackets, int bracket_count,
	char open_bracket, char closed_bracket,
	const BlockSumArray& sum_array) {
	int blockSize;
	int minGridSize;
	CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
		sumBracketsKernel));

	// Calculate actual block count to use based on records count
	int gridSize = (sum_array.length + blockSize - 1) / blockSize;

	sumBracketsKernel<<<gridSize, blockSize>>>(brackets, bracket_count, open_bracket, closed_bracket, sum_array);

	CUDA_TRY(cudaGetLastError());
};

__global__
void aggregateSumKernel(BlockSumArray in, BlockSumArray aggregate){
	const uint64_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
	const int aggregate_group_size = aggregate.block_size / in.block_size;
	const uint64_t did = tid * aggregate_group_size;

	if (tid >= aggregate.length)
		return;

	int16_t sum = 0;
	for (int i = did; i < did + aggregate_group_size; ++i)
		sum += in.d_sums[i];

	aggregate.d_sums[tid] = sum;
}

void aggregateSum(const BlockSumArray& in, const BlockSumArray& aggregate){
	int blockSize;
	int minGridSize;
	CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
		aggregateSumKernel));

	// Calculate actual block count to use based on records count
	int gridSize = (aggregate.length + blockSize - 1) / blockSize;

	aggregateSumKernel<<<gridSize, blockSize>>>(in, aggregate);

	CUDA_TRY(cudaGetLastError());
};

__global__
void assignLevelsKernel(
	pos_key_pair* brackets, uint64_t count,
	BlockSumArray* sum_pyramid, int pyramid_height,
	char open_bracket, char closed_bracket,
	int16_t* lvls) {
	const uint64_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
	const auto aggregation_rate = sum_pyramid[0].block_size;
	const uint64_t did = tid * aggregation_rate;

	if (did >= count)
		return;

	// find the previous sum
	int lvl = pyramid_height - 1;
	int sum = 0;
	int block_idx = 0;
	int offset = did;
	while(offset) {
		while(offset < sum_pyramid[lvl].block_size && lvl > 0) {
			--lvl; block_idx *= aggregation_rate;
		}
		while(offset >= sum_pyramid[lvl].block_size) {
			offset -= sum_pyramid[lvl].block_size;
			sum += sum_pyramid[lvl].d_sums[block_idx];
			++block_idx;
		}
	}

	for (int i = did; i < min(did + aggregation_rate, count); ++i){
		if (brackets[i].second == open_bracket)
			lvls[i] = ++sum;
		else if (brackets[i].second == closed_bracket)
			lvls[i] = sum--;
	}
}

void assignLevels(pos_key_pair* brackets, uint64_t count,
	const BlockSumPyramid& sum_pyramid,
	char open_bracket, char closed_bracket,
	int16_t* lvls) {
	int blockSize;
	int minGridSize;
	CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
		assignLevelsKernel));

	// Calculate actual block count to use based on records count
	const int threadCnt = (count + sum_pyramid.getAggregationRate() - 1) / sum_pyramid.getAggregationRate();
	const int gridSize = (threadCnt + blockSize - 1) / blockSize;

	assignLevelsKernel<<<gridSize, blockSize>>>(
		brackets, count,
		sum_pyramid.deviceGetLevels(), sum_pyramid.getHeigth(),
		open_bracket, closed_bracket,
		lvls);

	CUDA_TRY(cudaGetLastError());
};


// return a unique_ptr, once they are merged
int16_t* getBracketLevels(pos_key_pair* brackets, int count, char open_bracket, char closed_bracket){
	// Probably should be done outside of this function
	thrust::sort(rmm::exec_policy()->on(0), brackets, brackets + count);

	// total level difference for each segment of brackets in the file
	BlockSumPyramid aggregated_sums(count);
	
	// aggregate sums
	sumBrackets(brackets, count, open_bracket, closed_bracket, aggregated_sums[0]);
	for (size_t level_idx = 1; level_idx < aggregated_sums.getHeigth(); ++level_idx)
		aggregateSum(aggregated_sums[level_idx - 1], aggregated_sums[level_idx]);

	// assign levels
	int16_t* d_levels = nullptr;
	RMM_ALLOC(&d_levels, sizeof(int16_t) * count, 0);
	assignLevels(brackets, count, aggregated_sums, open_bracket, closed_bracket, d_levels);

	return d_levels;
}