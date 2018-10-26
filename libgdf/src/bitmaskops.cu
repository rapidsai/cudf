#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>
#include <gdf/cffi/functions.h>

#include <cuda_runtime.h>
#include <vector>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/device_vector.h>
#include "thrust_rmm_allocator.h"

// thrust::device_vector set to use rmmAlloc and rmmFree.
template <typename T>
using Vector = thrust::device_vector<T, rmm_allocator<T>>;


/*
 * bit_mask_null_counts Generated using the following code

#include <iostream>


int main()
{
	for (int i = 0 ; i != 256 ; i++) {
		int count = 0;
		for (int p = 0 ; p != 8 ; p++) {
			if (i & (1 << p)) {
				count++;
			}
		}
		std::cout<<(8-count)<<", ";
	}
	std::cout<<std::endl;
}
 */
std::vector<gdf_valid_type> bit_mask_null_counts = { 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0 };

unsigned char gdf_num_bits_zero_after_pos(unsigned char number, int pos){
	//if pos == 0 then its aligned
	if(pos == 0){
		return 0;
	}
	unsigned char count = 0;
	for (int p = pos ; p != 8 ; p++) {
		if (number & (number << p)) {
			count++;
		}
	}
	return (8 - pos) - count;
}

gdf_error all_bitmask_on(gdf_valid_type * valid_out, gdf_size_type & out_null_count, gdf_size_type num_values, cudaStream_t stream){
	gdf_size_type num_chars_bitmask = ( ( num_values +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );

	thrust::device_ptr<gdf_valid_type> valid_out_ptr = thrust::device_pointer_cast(valid_out);
	gdf_valid_type max_char = 255;
	rmm_temp_allocator allocator(stream);
	thrust::fill(thrust::cuda::par(allocator).on(stream),thrust::detail::make_normal_iterator(valid_out_ptr),thrust::detail::make_normal_iterator(valid_out_ptr + num_chars_bitmask),max_char);
	//we have no nulls so set all the bits in gdf_valid_type to 1
	out_null_count = 0;
	return GDF_SUCCESS;
}

gdf_error apply_bitmask_to_bitmask(gdf_size_type & out_null_count, gdf_valid_type * valid_out, gdf_valid_type * valid_left, gdf_valid_type * valid_right,
		cudaStream_t stream, gdf_size_type num_values){

	gdf_size_type num_chars_bitmask = ( ( num_values +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
	thrust::device_ptr<gdf_valid_type> valid_out_ptr = thrust::device_pointer_cast(valid_out);
	thrust::device_ptr<gdf_valid_type> valid_left_ptr = thrust::device_pointer_cast(valid_left);
	//here we are basically figuring out what is the last pointed to unsigned char that can contain part of the bitmask
	thrust::device_ptr<gdf_valid_type> valid_left_end_ptr = thrust::device_pointer_cast(valid_left + num_chars_bitmask );
	thrust::device_ptr<gdf_valid_type> valid_right_ptr = thrust::device_pointer_cast(valid_right);


	rmm_temp_allocator allocator(stream);
	thrust::transform(thrust::cuda::par(allocator).on(stream), thrust::detail::make_normal_iterator(valid_left_ptr),
			thrust::detail::make_normal_iterator(valid_left_end_ptr), thrust::detail::make_normal_iterator(valid_right_ptr),
			thrust::detail::make_normal_iterator(valid_out_ptr), thrust::bit_and<gdf_valid_type>());


	char * last_char = new char[1];
	cudaError_t error = cudaMemcpyAsync(last_char,valid_out + ( num_chars_bitmask-1),sizeof(gdf_valid_type),cudaMemcpyDeviceToHost,stream);


	Vector<gdf_valid_type> bit_mask_null_counts_device(bit_mask_null_counts);

	//this permutation iterator makes it so that each char basically gets replaced with its number of null counts
	//so if you sum up this perm iterator you add up all of the counts for null values per unsigned char
	thrust::permutation_iterator<Vector<gdf_valid_type>::iterator,thrust::detail::normal_iterator<thrust::device_ptr<gdf_valid_type> > >
	null_counts_iter( bit_mask_null_counts_device.begin(),thrust::detail::make_normal_iterator(valid_out_ptr));

	//you will notice that we subtract the number of zeros we found in the last character
	out_null_count = thrust::reduce(thrust::cuda::par(allocator).on(stream),null_counts_iter, null_counts_iter + num_chars_bitmask) - gdf_num_bits_zero_after_pos(*last_char,num_values % GDF_VALID_BITSIZE );

	delete[] last_char;
	return GDF_SUCCESS;
}

