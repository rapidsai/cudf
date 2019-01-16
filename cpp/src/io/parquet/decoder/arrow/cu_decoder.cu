/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
 *     Copyright 2018 William Malpica <william@blazingdb.com>
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

#include "cu_decoder.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <algorithm>
#include <iostream>
#include <tuple>

#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>

#include "bpacking.cuh"
#include "util/pinned_allocator.cuh"

namespace gdf
{
namespace arrow
{
namespace internal {

CachingPinnedAllocator pinnedAllocator(2, 14, 29, 1024*1024*1024*1ull);
 
namespace detail
{

#define ARROW_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define ARROW_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))

#define ARROW_DEBUG (-1)
#define ARROW_INFO 0
#define ARROW_WARNING 1
#define ARROW_ERROR 2
#define ARROW_FATAL 3

class CerrLog
{
  public:
    CerrLog(int severity) // NOLINT(runtime/explicit)
        : severity_(severity),
          has_logged_(false)
    {
    }

    virtual ~CerrLog()
    {
        if (has_logged_)
        {
            std::cerr << std::endl;
        }
        if (severity_ == ARROW_FATAL)
        {
            std::exit(1);
        }
    }

    template <class T>
    CerrLog &operator<<(const T &t)
    {
        if (severity_ != ARROW_DEBUG)
        {
            has_logged_ = true;
            std::cerr << t;
        }
        return *this;
    }

  protected:
    const int severity_;
    bool has_logged_;
};
 

/// Returns the 'num_bits' least-significant bits of 'v'.
__device__  __host__  static inline uint64_t TrailingBits(uint64_t v,
                                                        int num_bits)
{
    if (ARROW_PREDICT_FALSE(num_bits == 0))
        return 0;
    if (ARROW_PREDICT_FALSE(num_bits >= 64))
        return v;
    int n = 64 - num_bits;
    return (v << n) >> n;
}

template <typename T>
__device__  __host__   inline void GetValue_(int num_bits, T *v, int max_bytes,
                                          const uint8_t *buffer,
                                          int *bit_offset, int *byte_offset,
                                          uint64_t *buffered_values)
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4800)
#endif
    *v = static_cast<T>(TrailingBits(*buffered_values, *bit_offset + num_bits) >> *bit_offset);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    *bit_offset += num_bits;

    if (*bit_offset >= 64)
    {
        *byte_offset += 8;
        *bit_offset -= 64;

        int bytes_remaining = max_bytes - *byte_offset;
        if (ARROW_PREDICT_TRUE(bytes_remaining >= 8))
        {
            memcpy(buffered_values, buffer + *byte_offset, 8);
        }
        else
        {
            memcpy(buffered_values, buffer + *byte_offset, bytes_remaining);
        }
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4800 4805)
#endif
        // Read bits of v that crossed into new buffered_values_
        *v = *v | static_cast<T>(TrailingBits(*buffered_values, *bit_offset)
                                 << (num_bits - *bit_offset));
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        // DCHECK_LE(*bit_offset, 64);
    }
}

} // namespace detail

template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator>
OutputIterator gpu_expand(InputIterator1 first1, InputIterator1 last1,
                          InputIterator2 first2, OutputIterator output)
{
    typedef typename thrust::iterator_difference<InputIterator1>::type
        difference_type;

    difference_type input_size = thrust::distance(first1, last1);
    difference_type output_size = thrust::reduce(first1, last1);

    // scan the counts to obtain output offsets for each input element
    thrust::device_vector<difference_type> output_offsets(input_size, 0);
    thrust::exclusive_scan(first1, last1, output_offsets.begin());

    // scatter the nonzero counts into their corresponding output positions
    thrust::device_vector<difference_type> output_indices(output_size, 0);
    thrust::scatter_if(thrust::counting_iterator<difference_type>(0),
                       thrust::counting_iterator<difference_type>(input_size),
                       output_offsets.begin(), first1, output_indices.begin());

    // compute max-scan over the output indices, filling in the holes
    thrust::inclusive_scan(output_indices.begin(), output_indices.end(),
                           output_indices.begin(),
                           thrust::maximum<difference_type>());

    // gather input values according to index array (output =
    // first2[output_indices])
    OutputIterator output_end = output;
    thrust::advance(output_end, output_size);
    thrust::gather(output_indices.begin(), output_indices.end(), first2, output);

    // return output + output_size
    thrust::advance(output, output_size);
    return output;
}

__host__ __device__ inline const uint32_t* unpack32(const uint32_t* in, uint32_t* out, int num_bits) {
    const uint32_t* (*UnpackFunctionPtr[])(const uint32_t* in, uint32_t* out) = {nullunpacker32, unpack1_32, unpack2_32, unpack3_32, unpack4_32, unpack5_32, unpack6_32, unpack7_32, unpack8_32, unpack9_32, unpack10_32, unpack11_32, unpack12_32, unpack13_32, unpack14_32, unpack15_32, unpack16_32, unpack17_32, unpack18_32, unpack19_32, unpack20_32, unpack21_32, unpack22_32, unpack23_32, unpack24_32, unpack25_32, unpack26_32, unpack27_32, unpack28_32, unpack29_32, unpack30_32, unpack31_32, unpack32_32};
    return UnpackFunctionPtr[num_bits](in, out);
}

template<class T>
struct unpack_functor
    : public thrust::binary_function<uint8_t, T, uint32_t>
{
    int num_bits;
    unpack_functor(int num_bits) : num_bits(num_bits) {

    }
    __host__ __device__ uint32_t operator()(uint8_t &input, T &output)
    {
        uint32_t *input_ptr = (uint32_t *)&input;
        uint32_t *output_ptr = (uint32_t *)&output;
        
        unpack32(input_ptr, output_ptr, num_bits);

        return 0;
    }
};

template<typename Func>
    __global__
    void decode_bitpacking_32sets(uint8_t *buffer, int *output, int *input_offsets, int *input_run_lengths, int num_sets,
    		int * output_offsets, short bit_width, int max_num_sets_in_run, Func unpack_func)
    {

    	extern __shared__ uint8_t temp[];

    	const short INPUT_BLOCK_BYTES = bit_width * 32 / 8;
    	const short OUTPUT_BLOCK_BYTES = 32 * 4;
    	const short BLOCK_SIZE = 32;
    	const short IO_BLOCK = INPUT_BLOCK_BYTES + OUTPUT_BLOCK_BYTES;  // size in bytes of INPUT and OUTPUT BLOCK

    	int index = blockIdx.x * blockDim.x + threadIdx.x;

    	int set_index = index/max_num_sets_in_run;

    	if (set_index < num_sets){
    		int intput_index = input_offsets[set_index] + INPUT_BLOCK_BYTES * (index % max_num_sets_in_run);
    		int output_index = output_offsets[set_index] + BLOCK_SIZE * (index % max_num_sets_in_run);

    		if ((INPUT_BLOCK_BYTES * (index % max_num_sets_in_run)) < input_run_lengths[set_index]*bit_width/8) { // if we want to actually process

    			uint8_t * temp_in = &temp[IO_BLOCK * threadIdx.x];
    			int *temp_out = (int*)&temp[IO_BLOCK * threadIdx.x + INPUT_BLOCK_BYTES];

    			for (int i = 0; i < INPUT_BLOCK_BYTES; i++){
    				temp_in[i] = buffer[intput_index + i];
    			}
    			unpack_func(temp_in[0], temp_out[0]);

    			for (int i = 0; i < BLOCK_SIZE; i++){
    				output[output_index + i] = temp_out[i];
    			}
    		}
    	}
    }

typedef thrust::tuple<int, int, int, int> Int4;

template<class T>
struct remainder_functor : public thrust::unary_function<Int4, int>
{
    int max_bytes;
    int num_bits;
    uint8_t *d_buffer;
    T *ptr_output;
    remainder_functor(int max_bytes, int num_bits, uint8_t *buffer,
                      T *ptr_output)
        : max_bytes(max_bytes), num_bits(num_bits), d_buffer(buffer), ptr_output(ptr_output)
    {
    }
    __device__ __host__ int operator()(Int4 tuple)
    {
        int bit_offset = thrust::get<0>(tuple);  // remainderBitOffsets[k];
        int byte_offset = thrust::get<1>(tuple); // remainderInputOffsets[k];
        uint64_t buffered_values = 0;

        int bytes_remaining = max_bytes - byte_offset;
        if (bytes_remaining >= 8)
        {
            memcpy(&buffered_values, d_buffer + byte_offset, 8);
        }
        else
        {
            memcpy(&buffered_values, d_buffer + byte_offset, bytes_remaining);
        }
        int i = thrust::get<2>(tuple); // remainderOutputOffsets[k];
        int batch_size = thrust::get<2>(tuple) + thrust::get<3>(tuple); // remainderOutputOffsets[k] + remainderSetSize[k];
        for (; i < batch_size; ++i)
        {
            detail::GetValue_(num_bits, &ptr_output[i], max_bytes, (uint8_t *)d_buffer,
                              &bit_offset, &byte_offset, &buffered_values);
        }
        return 0;
    }
};

template<typename T>
void gpu_bit_packing_remainder( thrust::device_vector<uint8_t> & d_buffer,
                                const std::vector<int> &remainderInputOffsets,
                                const std::vector<int> &remainderBitOffsets,
                                const std::vector<int> &remainderSetSize,
                                const std::vector<int> &remainderOutputOffsets,
                                thrust::device_vector<T>& d_output,
                                int num_bits)
{

    thrust::device_vector<int> d_remainder_input_offsets(remainderInputOffsets);
    thrust::device_vector<int> d_remainder_bit_offsets(remainderBitOffsets);
    thrust::device_vector<int> d_remainder_setsize(remainderSetSize);
    thrust::device_vector<int> d_remainder_output_offsets(remainderOutputOffsets);

    int max_bytes = d_buffer.size();
    auto zip_iterator_begin = thrust::make_zip_iterator(thrust::make_tuple(
        d_remainder_bit_offsets.begin(), d_remainder_input_offsets.begin(),
        d_remainder_output_offsets.begin(), d_remainder_setsize.begin()));
    auto zip_iterator_end = thrust::make_zip_iterator(thrust::make_tuple(
        d_remainder_bit_offsets.end(), d_remainder_input_offsets.end(),
        d_remainder_output_offsets.end(), d_remainder_setsize.end()));

    thrust::transform(
        thrust::device, zip_iterator_begin, zip_iterator_end,
        thrust::make_discard_iterator(),
        remainder_functor<T>(max_bytes, num_bits, d_buffer.data().get(),
                          d_output.data().get()));

}


template<typename T>
void gpu_bit_packing(const uint8_t *buffer, 
                     const int buffer_len,
                     const std::vector<int> &input_offset,
                     const std::vector<std::pair<uint32_t, uint32_t>>& bitpackset,
                     const std::vector<int> &output_offset,
                     thrust::device_vector<T>& d_output, 
                     int num_bits) 
{
    thrust::device_vector<int> d_output_offset(output_offset);
    int step_size = 32 * num_bits / 8;
    uint8_t* h_bit_buffer;
    pinnedAllocator.pinnedAllocate((void **)&h_bit_buffer, step_size * input_offset.size());

	thrust::host_vector<int> h_bit_offset;
	for (int i = 0; i < input_offset.size(); i++){
	    h_bit_offset.push_back(i*step_size);
	}
     int sum = 0;
    for (auto &&pair : bitpackset) {
	    memcpy ( &h_bit_buffer[sum] , &buffer[pair.first], pair.second );
        sum += pair.second;
	}
    thrust::device_vector<uint8_t> d_bit_buffer(h_bit_buffer, h_bit_buffer + step_size * input_offset.size());
    thrust::device_vector<int> d_bit_offset(h_bit_offset);

	thrust::transform(thrust::cuda::par,
	    thrust::make_permutation_iterator(d_bit_buffer.begin(), d_bit_offset.begin()),
	    thrust::make_permutation_iterator(d_bit_buffer.end(), d_bit_offset.end()),
	    thrust::make_permutation_iterator(d_output.begin(), d_output_offset.begin()),
	    thrust::make_discard_iterator(), unpack_functor<T>(num_bits));
    pinnedAllocator.pinnedFree(h_bit_buffer);
}
 
template<typename T>
int decode_using_gpu(const T * d_dictionary, int num_dictionary_values, T* d_output, const uint8_t *buffer, const int buffer_len,
                     const std::vector<uint32_t> &rle_runs,
                     const std::vector<uint64_t> &rle_values,
                     const std::vector<int> &input_offset,
					 const std::vector<int> &input_runlengths,
                     const std::vector<int> &output_offset,
                     const std::vector<int> &remainderInputOffsets,
                     const std::vector<int> &remainderBitOffsets,
                     const std::vector<int> &remainderSetSize,
                     const std::vector<int> &remainderOutputOffsets,
                     int num_bits, int batch_size)
{
    thrust::device_vector<int> d_indices(batch_size);

    {
    	thrust::device_vector<uint32_t> d_counts(rle_runs);
    	thrust::device_vector<uint64_t> d_values(rle_values);
    	gpu_expand(d_counts.begin(), d_counts.end(), d_values.begin(), d_indices.begin());
    }

    thrust::device_vector<uint8_t> d_buffer(buffer_len);
    thrust::copy(buffer, buffer + buffer_len, d_buffer.begin());
    if (input_offset.size() > 0){
    	unpack_functor<int> func(num_bits);
    	thrust::device_vector<int> d_input_offsets(input_offset);
    	thrust::device_vector<int> d_input_runlengths(input_runlengths);
    	thrust::device_vector<int> d_output_offset(output_offset);

    	int max_num_sets_in_run = thrust::reduce(thrust::device,
    			d_input_runlengths.begin(), d_input_runlengths.end(),
				0,
				thrust::maximum<int>());
    	max_num_sets_in_run = max_num_sets_in_run/32;

    	int max_total_sets = max_num_sets_in_run * input_offset.size();

    	int blocksize = std::min(128, max_total_sets);
    	int gridsize = (max_total_sets + blocksize - 1) / blocksize;

    	int shared_memory = blocksize * (num_bits * 32/8 + 32 * 4);

    	decode_bitpacking_32sets<<<gridsize, blocksize, shared_memory>>>(thrust::raw_pointer_cast(d_buffer.data()), thrust::raw_pointer_cast(d_indices.data()),
    			thrust::raw_pointer_cast(d_input_offsets.data()), thrust::raw_pointer_cast(d_input_runlengths.data()), input_offset.size(),
				thrust::raw_pointer_cast(d_output_offset.data()), num_bits, max_num_sets_in_run, func);

    }

    if (remainderInputOffsets.size() > 0){
    	gpu_bit_packing_remainder(d_buffer, remainderInputOffsets, remainderBitOffsets, remainderSetSize, remainderOutputOffsets, d_indices, num_bits);
    }
    
    thrust::gather(thrust::device,
                d_indices.begin(), d_indices.end(),
                d_dictionary,
                d_output);
    return batch_size;
}

template <typename T>
struct copy_functor : public thrust::unary_function<int, T>
{
    __host__ __device__ T operator()(int input)
    {
        return static_cast<T>(input);
    }
};

template<typename T>
int decode_def_levels(const uint8_t* buffer, const int buffer_len,
                const std::vector<uint32_t> &rle_runs,
                const std::vector<uint64_t> &rle_values,
                const std::vector<int>& input_offset,
				const std::vector<int>& input_runlengths,
                const std::vector<int>& output_offset,
                const std::vector<int>& remainderInputOffsets,
                const std::vector<int>& remainderBitOffsets,
                const std::vector<int>& remainderSetSize,
                const std::vector<int>& remainderOutputOffsets,
                int num_bits,
                T* output, int batch_size) 
{

	thrust::device_vector<int> d_indices(batch_size);

	{
		thrust::device_vector<uint32_t> d_counts(rle_runs);
		thrust::device_vector<uint64_t> d_values(rle_values);
		gpu_expand(d_counts.begin(), d_counts.end(), d_values.begin(), d_indices.begin());
	}

	thrust::device_vector<uint8_t> d_buffer(buffer_len);
	thrust::copy(buffer, buffer + buffer_len, d_buffer.begin());
	if (input_offset.size() > 0){
		unpack_functor<int> func(num_bits);
		thrust::device_vector<int> d_input_offsets(input_offset);
		thrust::device_vector<int> d_input_runlengths(input_runlengths);
		thrust::device_vector<int> d_output_offset(output_offset);

		int max_num_sets_in_run = thrust::reduce(thrust::device,
				d_input_runlengths.begin(), d_input_runlengths.end(),
				0,
				thrust::maximum<int>());
		max_num_sets_in_run = max_num_sets_in_run/32;

		int max_total_sets = max_num_sets_in_run * input_offset.size();

		int blocksize = std::min(128, max_total_sets);
		int gridsize = (max_total_sets + blocksize - 1) / blocksize;

		int shared_memory = blocksize * (num_bits * 32/8 + 32 * 4);

		decode_bitpacking_32sets<<<gridsize, blocksize, shared_memory>>>(thrust::raw_pointer_cast(d_buffer.data()), thrust::raw_pointer_cast(d_indices.data()),
				thrust::raw_pointer_cast(d_input_offsets.data()), thrust::raw_pointer_cast(d_input_runlengths.data()), input_offset.size(),
				thrust::raw_pointer_cast(d_output_offset.data()), num_bits, max_num_sets_in_run, func);

	}

	if (remainderInputOffsets.size() > 0){
		gpu_bit_packing_remainder(d_buffer, remainderInputOffsets, remainderBitOffsets, remainderSetSize, remainderOutputOffsets, d_indices, num_bits);
	}
    
    thrust::transform(thrust::device, d_indices.begin(), d_indices.end(), output, copy_functor<T>());
    return batch_size;
}
                
template<typename T>
int unpack_using_gpu(const uint8_t* buffer, const int buffer_len,
                const std::vector<int>& input_offset,
				const std::vector<int>& input_runlengths,
                const std::vector<int>& output_offset,
                const std::vector<int>& remainderInputOffsets,
                const std::vector<int>& remainderBitOffsets,
                const std::vector<int>& remainderSetSize,
                const std::vector<int>& remainderOutputOffsets,
                int num_bits,
                T* device_output, int batch_size) 
{

	thrust::device_vector<int> d_output_int(batch_size);
    thrust::device_vector<uint8_t> d_buffer(buffer_len);
    thrust::copy(buffer, buffer + buffer_len, d_buffer.begin());

    if (input_offset.size() > 0){

    	unpack_functor<int> func(num_bits);
    	thrust::device_vector<int> d_input_offsets(input_offset);
    	thrust::device_vector<int> d_input_runlengths(input_runlengths);
    	thrust::device_vector<int> d_output_offset(output_offset);

    	int max_num_sets_in_run = thrust::reduce(thrust::device,
    			d_input_runlengths.begin(), d_input_runlengths.end(),
				0,
				thrust::maximum<int>());
    	max_num_sets_in_run = max_num_sets_in_run/32;

    	int max_total_sets = max_num_sets_in_run * input_offset.size();

    	int blocksize = std::min(128, max_total_sets);
    	int gridsize = (max_total_sets + blocksize - 1) / blocksize;

    	int shared_memory = blocksize * (num_bits * 32/8 + 32 * 4);

    	decode_bitpacking_32sets<<<gridsize, blocksize, shared_memory>>>(thrust::raw_pointer_cast(d_buffer.data()), thrust::raw_pointer_cast(d_output_int.data()),
    			thrust::raw_pointer_cast(d_input_offsets.data()), thrust::raw_pointer_cast(d_input_runlengths.data()), input_offset.size(),
				thrust::raw_pointer_cast(d_output_offset.data()), num_bits, max_num_sets_in_run, func);

    }

    if (remainderInputOffsets.size() > 0){
    	gpu_bit_packing_remainder(d_buffer, remainderInputOffsets, remainderBitOffsets, remainderSetSize, remainderOutputOffsets, d_output_int, num_bits);
    }

    thrust::transform(thrust::device, d_output_int.begin(), d_output_int.end(), device_output, copy_functor<T>());
    return batch_size;
}


#define CONCRETIZE_FUNCTION(T) \
template int decode_using_gpu<T>(const T *dictionary, int num_dictionary_values, T* d_output, const uint8_t *buffer, const int buffer_len, \
                    const std::vector<uint32_t> &rle_runs, \
                    const std::vector<uint64_t> &rle_values, \
                    const std::vector<int> &input_offset, \
					const std::vector<int> &input_runlengths, \
                    const std::vector<int> &output_offset, \
                    const std::vector<int> &remainderInputOffsets, \
                    const std::vector<int> &remainderBitOffsets, \
                    const std::vector<int> &remainderSetSize, \
                    const std::vector<int> &remainderOutputOffsets, \
                    int num_bits, \
                    int batch_size \
                    )

CONCRETIZE_FUNCTION(bool);
CONCRETIZE_FUNCTION(int32_t);
CONCRETIZE_FUNCTION(int64_t);
CONCRETIZE_FUNCTION(float);
CONCRETIZE_FUNCTION(double);

#undef CONCRETIZE_FUNCTION 

template int unpack_using_gpu<bool>(const uint8_t* buffer, const int buffer_len, 
            const std::vector<int>& input_offset,
			const std::vector<int>& input_runlengths,
            const std::vector<int>& output_offset,  
            const std::vector<int>& remainderInputOffsets, 
            const std::vector<int>& remainderBitOffsets,  
            const std::vector<int>& remainderSetSize, 
            const std::vector<int>& remainderOutputOffsets, 
            int num_bits, 
            bool* device_output, int batch_size
            );


template int unpack_using_gpu<int16_t>(const uint8_t* buffer, const int buffer_len, 
            const std::vector<int>& input_offset, 
			const std::vector<int>& input_runlengths,
            const std::vector<int>& output_offset,  
            const std::vector<int>& remainderInputOffsets, 
            const std::vector<int>& remainderBitOffsets,  
            const std::vector<int>& remainderSetSize, 
            const std::vector<int>& remainderOutputOffsets, 
            int num_bits, 
            int16_t* output, int batch_size 
            );

template int decode_def_levels<int16_t>(const uint8_t* buffer, const int buffer_len,
                const std::vector<uint32_t> &rle_runs,
                const std::vector<uint64_t> &rle_values,
                const std::vector<int>& input_offset,
				const std::vector<int>& input_runlengths,
                const std::vector<int>& output_offset,
                const std::vector<int>& remainderInputOffsets,
                const std::vector<int>& remainderBitOffsets,
                const std::vector<int>& remainderSetSize,
                const std::vector<int>& remainderOutputOffsets,
                int num_bits,
                int16_t* output, int batch_size);

                

} // namespace internal
} // namespace arrow
} // namespace gdf
