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
#include "io/utilities/wrapper_utils.hpp"

// When processing the input in chunks, this is the maximum size of each chunk.
// Only one chunk is loaded on the GPU at a time, so this value is chosen to
// be small enough to fit on the GPU in most cases.
constexpr size_t max_chunk_bytes = 256*1024*1024; // 256MB

constexpr int bytes_per_find_thread = 64;

using pos_key_pair = thrust::pair<uint64_t,char>;

template <typename T>
constexpr T divCeil(T dividend, T divisor) noexcept { return (dividend + divisor - 1) / divisor; }

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

	const size_t chunk_count = divCeil(h_size, max_chunk_bytes);
	for (size_t ci = 0; ci < chunk_count; ++ci) {	
		const auto chunk_offset = ci * max_chunk_bytes;	
		const auto h_chunk = h_data + chunk_offset;
		const int chunk_bytes = std::min((size_t)(h_size - ci * max_chunk_bytes), max_chunk_bytes);
		const auto chunk_bits = divCeil(chunk_bytes, bytes_per_find_thread);
		const int gridSize = divCeil(chunk_bits, blockSize);

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

/**
 * @brief A class representing an array of partial sums, stored on the GPU memory.
 *
 * The object is just a reference to the device memory,
 * it does not own the device memory.
 **/
struct BlockSumArray {
		int16_t* d_sums = nullptr;	///< Array of partial sums
		uint64_t length = 0;		///< Length of the array
		uint64_t block_size;		///< The number of elements aggregated into each partial sum

		BlockSumArray(uint64_t len, uint64_t bsize): length(len), block_size(bsize){}
};

/**
 * @brief A class that stores a pyramid of aggregated sums, in the GPU memory.
 *
 * Each level of the pyramid is aggregation_rate times smaller than the previous, rounded down.
 * Objects of this type own the allocated memory.
 **/
class BlockSumPyramid {
	const uint16_t aggregation_rate_ = 32;	///< Aggregation rate between each level of the pyramid
	BlockSumArray* d_levels_ = nullptr;		///< Device array of partial sums, largest to smallest
	std::vector<BlockSumArray> levels_;		///< Host array of the partial sums on device, largest to smallest

public:
	BlockSumPyramid(int count){
		int prev_lvl_cnt = count;
		int prev_lvl_block_size = 1;
		while (prev_lvl_cnt >= aggregation_rate_) {
			levels_.emplace_back(prev_lvl_cnt/aggregation_rate_, prev_lvl_block_size*aggregation_rate_);
			RMM_ALLOC(&levels_.back().d_sums, levels_.back().length*sizeof(int16_t), 0);
			prev_lvl_cnt = levels_.back().length;
			prev_lvl_block_size = levels_.back().block_size;
		}

		if (!levels_.empty()) {	
			RMM_ALLOC(&d_levels_, levels_.size()*sizeof(BlockSumArray), 0);
			cudaMemcpyAsync(d_levels_, levels_.data(), levels_.size()*sizeof(BlockSumArray), cudaMemcpyDefault);
		}
	}

	auto operator[](int lvl) const {return levels_[lvl];}
	auto deviceGetLevels() const noexcept {return d_levels_;}
	size_t getHeight() const noexcept {return levels_.size();}
	constexpr auto getAggregationRate() const {return aggregation_rate_;}

	// disable copying
	BlockSumPyramid(BlockSumPyramid&) = delete;
	BlockSumPyramid& operator=(BlockSumPyramid&) = delete;

	~BlockSumPyramid() {
		for (auto& lvl: levels_) {
			RMM_FREE(lvl.d_sums, 0);
		}
		RMM_FREE(d_levels_, 0);
	}
};

/**---------------------------------------------------------------------------*
 * @brief CUDA kernel that aggregates bracket nesting levels for each block
 * in the input array.
 *
 * Each sum is the level difference between the first bracket in the block,
 * and the first bracket in the next block (if any). For example, "[[]]" = 0,
 * because all open brackets are closed. "[[]" = 1, because the one unmatched
 * open bracket would raise the level of all subsequent lemenets.
 * 
 * @param[in] brackets Array of brackets, in (offset, char) format
 * @param[in] bracket_count Number of brackets
 * @param[in] open_chars Array of characters to treat as open brackets
 * @param[in] close_chars Array of characters to treat as open brackets
 * @param[in] bracket_char_cnt Number of bracket character pairs
 * @param[in, out] sum_array Array of partial sums
 * 
 * @return void
 *---------------------------------------------------------------------------**/
__global__
void sumBracketsKernel(
	pos_key_pair* brackets, int bracket_count,
	const char* open_chars, const char* close_chars, int bracket_char_cnt,
	BlockSumArray sum_array) {
	const uint64_t sum_idx = threadIdx.x + (blockDim.x * blockIdx.x);
	const uint64_t first_in_idx = sum_idx * sum_array.block_size;

	if (sum_idx >= sum_array.length)
		return;

	int16_t sum = 0;
	for (uint64_t in_idx = first_in_idx; in_idx < first_in_idx + sum_array.block_size; ++in_idx) {
		for (int bchar_idx = 0; bchar_idx < bracket_char_cnt; ++bchar_idx) {
			if (brackets[in_idx].second == open_chars[bchar_idx]) {
				++sum; 
				break;
			}
			if (brackets[in_idx].second == close_chars[bchar_idx]) {
				--sum; 
				break;
			}
		}
	}
	sum_array.d_sums[sum_idx] = sum;
}

/**---------------------------------------------------------------------------*
 * @brief Wrapper around sumBracketsKernel
 *
 * @param[in] brackets Array of brackets, in (offset, char) format
 * @param[in] bracket_count Number of brackets
 * @param[in] open_chars Array of characters to treat as open brackets
 * @param[in] close_chars Array of characters to treat as close brackets
 * @param[in] bracket_char_cnt Number of bracket character pairs
 * @param[in, out] sum_array Array of partial sums
 * 
 * @return void
 *---------------------------------------------------------------------------**/
void sumBrackets(
	pos_key_pair* brackets, int bracket_count,
	char* open_chars, char* close_chars, int bracket_char_cnt,
	const BlockSumArray& sum_array) {
	int blockSize = 0;
	int minGridSize = 0;
	CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
		sumBracketsKernel));

	const int gridSize = divCeil(sum_array.length, (uint64_t)blockSize);

	sumBracketsKernel<<<gridSize, blockSize>>>(
		brackets, bracket_count,
		open_chars, close_chars, bracket_char_cnt,
		sum_array);
	CUDA_TRY(cudaGetLastError());
};

/**---------------------------------------------------------------------------*
 * @brief CUDA kernel that computes partial sums of the input elements
 * 
 * @param[in] elements Array of input elements to sum
 * @param[in, out] aggregate Array of partial sums
 * 
 * @return void
 *---------------------------------------------------------------------------**/
__global__
void aggregateSumKernel(BlockSumArray elements, BlockSumArray aggregate){
	const uint64_t aggregate_idx = threadIdx.x + (blockDim.x * blockIdx.x);
	const int aggregate_group_size = aggregate.block_size / elements.block_size;
	const uint64_t first_in_idx = aggregate_idx * aggregate_group_size;

	if (aggregate_idx >= aggregate.length)
		return;

	int16_t sum = 0;
	for (int in_idx = first_in_idx; in_idx < first_in_idx + aggregate_group_size; ++in_idx) {
		sum += elements.d_sums[in_idx];
	}

	aggregate.d_sums[aggregate_idx] = sum;
}

/**---------------------------------------------------------------------------*
 * @brief Wrapper around aggregateSumKernel
 * 
 * @param[in] elements Array of input elements to sum
 * @param[in, out] aggregate Array of partial sums
 * 
 * @return void
 *---------------------------------------------------------------------------**/
void aggregateSum(const BlockSumArray& elements, const BlockSumArray& aggregate){
	int blockSize = 0;
	int minGridSize = 0;
	CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
		aggregateSumKernel));

	const int gridSize = divCeil(aggregate.length, (uint64_t)blockSize);

	aggregateSumKernel<<<gridSize, blockSize>>>(elements, aggregate);
	CUDA_TRY(cudaGetLastError());
};

/**---------------------------------------------------------------------------*
 * @brief CUDA kernel that assigns levels to each bracket,
 * with 1 being the top level
 *
 * The algorithm uses the pyramid of partial sums to compute the levels
 * in parallel, in log(n) time per block of elements.
 * 
 * @param[in] brackets Array of brackets, in (offset, char) format
 * @param[in] count Number of brackets
 * @param[in] sum_pyramid 
 * @param[in] pyramid_height 
 * @param[in] open_chars Array of characters to treat as open brackets
 * @param[in] close_chars Array of characters to treat as close brackets
 * @param[in] bracket_char_cnt Number of bracket character pairs
 * @param[out] levels Array of outout levels
 * 
 * @return void
 *---------------------------------------------------------------------------**/
__global__
void assignLevelsKernel(
	pos_key_pair* brackets, uint64_t count,
	BlockSumArray* sum_pyramid, int pyramid_height,
	char* open_chars, char* close_chars, int bracket_char_cnt,
	int16_t* levels) {
	// Process the number of elements equal to the aggregation rate, if the pyramid is used
	// Process all elements otherwise
	const auto to_process = pyramid_height != 0 ? sum_pyramid[0].block_size : count;
	const uint64_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
	const uint64_t first_bracket_idx = tid * to_process;

	if (first_bracket_idx >= count)
		return;

	// Find the total sum of levels before the current block
	int sum = 0;
	if (pyramid_height != 0) {
		const auto aggregation_rate = sum_pyramid[0].block_size;
		int level = pyramid_height - 1;
		int block_idx = 0;
		int offset = first_bracket_idx;
		while(offset) {
			while(offset < sum_pyramid[level].block_size && level > 0) {
				--level; block_idx *= aggregation_rate;
			}
			while(offset >= sum_pyramid[level].block_size) {
				offset -= sum_pyramid[level].block_size;
				sum += sum_pyramid[level].d_sums[block_idx];
				++block_idx;
			}
		}
	}

	const auto last_bracket_idx = min(first_bracket_idx + to_process, count) - 1;
	for (uint64_t bracket_idx = first_bracket_idx; bracket_idx <= last_bracket_idx; ++bracket_idx){
		for (int bchar_idx = 0; bchar_idx < bracket_char_cnt; ++bchar_idx) {
			if (brackets[bracket_idx].second == open_chars[bchar_idx]) {
				levels[bracket_idx] = ++sum;
				break;
			}
			else if (brackets[bracket_idx].second == close_chars[bchar_idx]) {
				levels[bracket_idx] = sum--;
				break;
			}
		}
	}
}

/**---------------------------------------------------------------------------*
 * @brief Wrapper around assignLevelsKernel
 * 
 * @param[in] brackets Array of brackets, in (offset, char) format
 * @param[in] count Number of brackets
 * @param[in] sum_pyramid 
 * @param[in] pyramid_height 
 * @param[in] open_chars Array of characters to treat as open brackets
 * @param[in] close_chars Array of characters to treat as close brackets
 * @param[in] bracket_char_cnt Number of bracket character pairs
 * @param[out] levels Array of outout levels
 * 
 * @return void
 *---------------------------------------------------------------------------**/
void assignLevels(pos_key_pair* brackets, uint64_t count,
	const BlockSumPyramid& sum_pyramid,
	char* open_chars, char* close_chars, int bracket_char_cnt,
	int16_t* levels) {
	int blockSize;
	int minGridSize;
	CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
		assignLevelsKernel));

	const int threadCnt = divCeil(count, (uint64_t)sum_pyramid.getAggregationRate());
	const int gridSize = divCeil(threadCnt, blockSize);

	assignLevelsKernel<<<gridSize, blockSize>>>(
		brackets, count,
		sum_pyramid.deviceGetLevels(), sum_pyramid.getHeight(),
		open_chars, close_chars, bracket_char_cnt,
		levels);
	CUDA_TRY(cudaGetLastError());
};

/**---------------------------------------------------------------------------*
 * @brief Computes nested level for each of the brackets in the input array
 * 
 * The input array of brackets is sorted before levels are computed. 
 * Brackets at the top level are assigned level 1.
 * 
 * @param[in] brackets Array of brackets, in (offset, char) format
 * @param[in] count Number of brackets
 * @param[in] open_chars string of characters to treat as open brackets
 * @param[in] close_chars string of characters to treat as close brackets
 * 
 * @return device_ptr<int16_t> Device memory array of levels
 *---------------------------------------------------------------------------**/
device_ptr<int16_t> getBracketLevels(
	pos_key_pair* brackets, int count,
	const std::string& open_chars, const std::string& close_chars){
	// TODO: consider moving sort() out of this function
	thrust::sort(rmm::exec_policy()->on(0), brackets, brackets + count);

	// Total bracket level difference within each segment of brackets
	BlockSumPyramid aggregated_sums(count);
	
	assert(open_chars.size() == open_chars.size());

	// Copy the open/close chars to device
	char* d_open_chars = nullptr;
	RMM_ALLOC(&d_open_chars, open_chars.size() * sizeof(char), 0);
	device_ptr<char> d_open_chars_deleter(d_open_chars);
	CUDA_TRY(cudaMemcpyAsync(
		d_open_chars, open_chars.c_str(),
		open_chars.size() * sizeof(char), cudaMemcpyDefault));

	char* d_close_chars = nullptr;
	RMM_ALLOC(&d_close_chars, close_chars.size() * sizeof(char), 0);
	device_ptr<char> d_close_chars_deleter(d_open_chars);
	CUDA_TRY(cudaMemcpyAsync(
		d_close_chars, close_chars.c_str(),
		close_chars.size() * sizeof(char), cudaMemcpyDefault));

	if (aggregated_sums.getHeight() != 0) {
		sumBrackets(brackets, count, d_open_chars, d_close_chars, open_chars.size(), aggregated_sums[0]);
		for (size_t level_idx = 1; level_idx < aggregated_sums.getHeight(); ++level_idx)
			aggregateSum(aggregated_sums[level_idx - 1], aggregated_sums[level_idx]);
	}

	int16_t* d_levels = nullptr;
	RMM_ALLOC(&d_levels, sizeof(int16_t) * count, 0);
	assignLevels(brackets, count, aggregated_sums, d_open_chars, d_close_chars, open_chars.size(), d_levels);

	return device_ptr<int16_t>(d_levels);
}
