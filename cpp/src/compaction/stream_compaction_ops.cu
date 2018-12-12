/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Felipe Aramburu <felipe@blazingdb.com>
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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

#include <cuda_runtime.h>
#include <vector>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/transform_iterator.h>

#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/miscellany.hpp"
#include "utilities/type_dispatcher.hpp"

//std lib
#include <map>


struct shift_left: public thrust::unary_function<gdf_valid_type,gdf_valid_type>
{

	gdf_valid_type num_bits;
	shift_left(gdf_valid_type num_bits): num_bits(num_bits){

	}

  __host__ __device__
  gdf_valid_type operator()(gdf_valid_type x) const
  {
    return x << num_bits;
  }
};

struct shift_right: public thrust::unary_function<gdf_valid_type,gdf_valid_type>
{

	gdf_valid_type num_bits;
	bool not_too_many;
	shift_right(gdf_valid_type num_bits, bool not_too_many)
		: num_bits(num_bits), not_too_many(not_too_many){

	}

  __host__ __device__
  gdf_valid_type operator()(gdf_valid_type x) const
  {
	    //if you want to force the shift to be fill bits with 0 you need to use an unsigned type
	  /*if (not_too_many) { // is the last 
		return  x; 
	  }*/
	  return *((unsigned char *) &x) >> num_bits;

  }
};
 
struct bit_or: public thrust::unary_function<thrust::tuple<gdf_valid_type,gdf_valid_type>,gdf_valid_type>
{
	 

	__host__ __device__
	gdf_valid_type operator()(thrust::tuple<gdf_valid_type,gdf_valid_type> x) const
	{
		return thrust::get<0>(x) | thrust::get<1>(x);
	}
};

struct is_bit_set
{
	__host__ __device__
	bool operator()(const thrust::tuple<gdf_size_type, thrust::device_ptr<gdf_valid_type>> value)
	{
		gdf_size_type position = thrust::get<0>(value);

		return gdf_is_valid(thrust::get<1>(value).get(), position);
	}
}; 

typedef thrust::tuple<thrust::counting_iterator<gdf_size_type>, thrust::constant_iterator<gdf_valid_type*>> mask_tuple;
typedef thrust::zip_iterator<mask_tuple> zipped_mask;

typedef thrust::transform_iterator<is_bit_set, zipped_mask> bit_set_iterator;

template<typename stencil_type>
struct is_stencil_true
{
	__host__ __device__
	bool operator()(const thrust::tuple<stencil_type, bit_set_iterator::value_type> value)
	{
		return thrust::get<1>(value) && (thrust::get<0>(value) != 0);
	}
};

struct bit_mask_pack_op : public thrust::unary_function<int64_t,gdf_valid_type>
{
	static_assert(sizeof(gdf_valid_type) == 1, "Unexpected size of gdf_valid_type");
	__host__ __device__
		gdf_valid_type operator()(const int64_t expanded)
		{
			gdf_valid_type result = 0;
			for(unsigned i = 0; i < GDF_VALID_BITSIZE; i++){
				unsigned char byte = (expanded >> (i * CHAR_BIT));
				result |= (byte & 1) << i;
			}
			return result;
		}
};

//zip the stencil and the valid iterator together
typedef thrust::tuple<thrust::detail::normal_iterator<thrust::device_ptr<int8_t>>, bit_set_iterator> zipped_stencil_tuple;
typedef thrust::zip_iterator<zipped_stencil_tuple> zipped_stencil_iterator;

struct apply_stencil_functor{
	template <typename col_type>
	__host__
	void operator()(gdf_column* col, gdf_column* output, zipped_stencil_iterator zipped_stencil_iter)
	{
		thrust::detail::normal_iterator<thrust::device_ptr<col_type> > input_start =
		thrust::detail::make_normal_iterator(thrust::device_pointer_cast((col_type *) col->data));
		thrust::detail::normal_iterator<thrust::device_ptr<col_type> > output_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((col_type *) output->data));
		thrust::detail::normal_iterator<thrust::device_ptr<col_type> > output_end =
				thrust::copy_if(input_start,input_start + col->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >::value_type >());
		output->size = output_end - output_start;
	}
};

//TODO: add a way for the space where we store temp bitmaps for compaction be allocated
//on the outside
gdf_error gpu_apply_stencil(gdf_column * col, gdf_column * stencil, gdf_column * output) {
	GDF_REQUIRE(output->size == col->size, GDF_COLUMN_SIZE_MISMATCH);
	GDF_REQUIRE(col->dtype == output->dtype, GDF_DTYPE_MISMATCH);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	size_t n_bytes = get_number_of_bytes_for_valid(stencil->size);

	zipped_mask  zipped_mask_stencil_iter(
		thrust::make_tuple(
			thrust::make_counting_iterator<gdf_size_type>(0),
			thrust::make_constant_iterator(stencil->valid)
		)
	);

	bit_set_iterator bit_set_stencil_iter = thrust::make_transform_iterator<is_bit_set, zipped_mask>(
			zipped_mask_stencil_iter,
			is_bit_set()
	);

	//well basically we are zipping up an iterator to the stencil, one to the bit masks, and one which lets us get the bit position based on our index
	zipped_stencil_iterator zipped_stencil_iter(
		thrust::make_tuple(
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int8_t * )stencil->data)),
				bit_set_stencil_iter
		));

	//NOTE!!!! the output column is getting set to a specific size but we are NOT compacting the allocation,
	//whoever calls that should handle that
	cudf::type_dispatcher(col->dtype, apply_stencil_functor{}, col, output, zipped_stencil_iter);

	if(col->valid != nullptr) {
		gdf_size_type num_values = col->size;

		//TODO:BRING OVER THE BITMASK!!!
		rmm::device_vector<gdf_valid_type> valid_bit_mask; //we are expanding the bit mask to an int8 because I can't envision an algorithm that operates on the bitmask that
		if(num_values % GDF_VALID_BITSIZE != 0){
			valid_bit_mask.resize(num_values + (GDF_VALID_BITSIZE - (num_values % GDF_VALID_BITSIZE))); //align this allocation on GDF_VALID_BITSIZE so we don't have to bounds check
		}else{
			valid_bit_mask.resize(num_values);
		}

		zipped_mask  zipped_mask_col_iter(
			thrust::make_tuple(
				thrust::make_counting_iterator<gdf_size_type>(0),
				thrust::make_constant_iterator(col->valid)
			)
		);

		bit_set_iterator bit_set_col_iter = thrust::make_transform_iterator<is_bit_set, zipped_mask>(
				zipped_mask_col_iter,
				is_bit_set()
		);

		//copy the bitmask to device_vector of int8
		thrust::copy(rmm::exec_policy(stream), bit_set_col_iter, bit_set_col_iter + num_values, valid_bit_mask.begin());

		//remove the values that don't pass the stencil
		thrust::copy_if(rmm::exec_policy(stream), valid_bit_mask.begin(), valid_bit_mask.begin() + num_values, zipped_stencil_iter, valid_bit_mask.begin(),
				is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t>>::value_type>());

		//recompact the values and store them in the output bitmask
		//we can group them into pieces of 8 because we aligned this earlier on when we made the device_vector
		thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > valid_bit_mask_group_8_iter =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) valid_bit_mask.data().get()));

		//you may notice that we can write out more bytes than our valid_num_bytes, this only happens when we are not aligned to  GDF_VALID_BITSIZE bytes, becasue the
		//arrow standard requires 64 byte alignment, this is a safe assumption to make
		thrust::transform(rmm::exec_policy(stream), valid_bit_mask_group_8_iter, valid_bit_mask_group_8_iter + ((num_values + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE),
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast(output->valid)),bit_mask_pack_op());
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	return GDF_SUCCESS;
} 

size_t  get_last_byte_length(size_t column_size) {
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);
    size_t length = column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
    if (n_bytes == 1 ) {
        length = column_size;
    }
    return  length;
}

size_t  get_right_byte_length(size_t column_size, size_t iter, size_t left_length) {
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);
    size_t length = column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
    if (iter == n_bytes - 1) { // the last one
        if (left_length + length > GDF_VALID_BITSIZE) {
            length = GDF_VALID_BITSIZE - left_length;
        }
    }
    else {
        length = GDF_VALID_BITSIZE - left_length;
    }
    return length;
}
 

 bool last_with_too_many_bits(size_t column_size, size_t iter, size_t left_length) {
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);
    size_t length = column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
    if (iter == n_bytes) { // the last one
        // the last one has to many bits
        if (left_length + length > GDF_VALID_BITSIZE) {
            return true;
        }
    }
    return false;
}


 gdf_valid_type concat_bins (gdf_valid_type A, gdf_valid_type B, int len_a, int len_b, bool has_next, size_t right_length){
    A = A << len_b;
    if (!has_next) {
        B = B << len_a;
        B = B >> len_a;
    } else {
        B = B >> right_length - len_b;
    }
    return  (A | B);
}

gdf_error gpu_concat(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
{
	GDF_REQUIRE( (lhs->dtype == output->dtype ) && ( rhs->dtype == output->dtype), GDF_VALIDITY_MISSING);
	GDF_REQUIRE(output->size == lhs->size + rhs->size, GDF_COLUMN_SIZE_MISMATCH);
	cudaStream_t stream;
	cudaStreamCreate(&stream);

    int type_width;
    auto result = get_column_byte_width(lhs, &type_width);
    GDF_REQUIRE(result == GDF_SUCCESS, GDF_UNSUPPORTED_DTYPE);

	cudaMemcpyAsync(output->data, lhs->data, type_width * lhs->size, cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync( (void *)( (int8_t*) (output->data) + type_width * lhs->size), rhs->data, type_width * rhs->size, cudaMemcpyDeviceToDevice, stream);
	
	int left_num_chars = get_number_of_bytes_for_valid(lhs->size);
	int right_num_chars = get_number_of_bytes_for_valid(rhs->size);
  	int output_num_chars = get_number_of_bytes_for_valid(output->size); 
					
	thrust::device_ptr<gdf_valid_type> left_device_bits = thrust::device_pointer_cast((gdf_valid_type *)lhs->valid);
	thrust::device_ptr<gdf_valid_type> right_device_bits = thrust::device_pointer_cast((gdf_valid_type *)rhs->valid);
	thrust::device_ptr<gdf_valid_type> output_device_bits = thrust::device_pointer_cast((gdf_valid_type *)output->valid);

	thrust::copy(left_device_bits, left_device_bits + left_num_chars, output_device_bits);
	
	gdf_valid_type shift_bits = (GDF_VALID_BITSIZE - (lhs->size % GDF_VALID_BITSIZE));
	if(shift_bits == 8){
		shift_bits = 0;
	}
	if (right_num_chars > 0) {
		size_t prev_len = get_last_byte_length(lhs->size);

		// copy all the rnbytes bytes  from right column
		if (shift_bits == 0) { 
			thrust::copy(right_device_bits, right_device_bits + right_num_chars, output_device_bits + left_num_chars);
		}
		else { 
			thrust::host_vector<gdf_valid_type> last_byte (2);
			thrust::copy (left_device_bits + left_num_chars - 1, left_device_bits + left_num_chars, last_byte.begin());
			thrust::copy (right_device_bits, right_device_bits + 1, last_byte.begin() + 1);
			        
			size_t curr_len = get_right_byte_length(rhs->size, 0, prev_len);

			if (1 != right_num_chars) {
				last_byte[1] = last_byte[1] >> prev_len;
			}
			auto flag = last_with_too_many_bits(rhs->size, 0 + 1, prev_len);
			size_t last_right_byte_length = rhs->size - GDF_VALID_BITSIZE * (right_num_chars - 1);
			last_byte[0] = concat_bins(last_byte[0], last_byte[1], prev_len, curr_len, flag, last_right_byte_length);

			thrust::copy( last_byte.begin(), last_byte.begin() + 1, output_device_bits + left_num_chars - 1);
			
			if(right_num_chars > 1)  {
				using first_iterator_type = thrust::transform_iterator<shift_left,rmm::device_vector<gdf_valid_type>::iterator>;
				using second_iterator_type = thrust::transform_iterator<shift_right,rmm::device_vector<gdf_valid_type>::iterator>;
				using offset_tuple = thrust::tuple<first_iterator_type, second_iterator_type>;
				using zipped_offset = thrust::zip_iterator<offset_tuple>;

				auto too_many_bits = last_with_too_many_bits(rhs->size, right_num_chars, prev_len);
				size_t last_byte_length = get_last_byte_length(rhs->size);

				if (last_byte_length >= (GDF_VALID_BITSIZE - shift_bits)) { //  
					thrust::host_vector<gdf_valid_type> last_byte (right_device_bits + right_num_chars - 1, right_device_bits + right_num_chars);
					last_byte[0] = last_byte[0] << GDF_VALID_BITSIZE - last_byte_length;
					thrust::copy( last_byte.begin(), last_byte.begin() + 1, right_device_bits + right_num_chars - 1);
				}
				
				zipped_offset  zipped_offset_iter(
						thrust::make_tuple(
								thrust::make_transform_iterator<shift_left, rmm::device_vector<gdf_valid_type>::iterator >(
										right_device_bits,
										shift_left(shift_bits)),
								
								thrust::make_transform_iterator<shift_right, rmm::device_vector<gdf_valid_type>::iterator >(
										right_device_bits + 1,
										shift_right(GDF_VALID_BITSIZE - shift_bits, !too_many_bits))
						)	
				);
				//so what this does is give you an iterator which gives you a tuple where you have your char, and the char after you, so you can get the last bits!
				using transformed_or = thrust::transform_iterator<bit_or, zipped_offset>;
				//now we want to make a transform iterator that ands these values together
				transformed_or ored_offset_iter =
						thrust::make_transform_iterator<bit_or,zipped_offset> (
								zipped_offset_iter,
								bit_or()
						);
				//because one of the iterators is + 1 we dont want to read the last char here since it could be past the end of our allocation
				thrust::copy( ored_offset_iter, ored_offset_iter + right_num_chars - 1, output_device_bits + left_num_chars);

				thrust::host_vector<gdf_valid_type> last_byte (right_device_bits + right_num_chars - 1, right_device_bits + right_num_chars);
				last_byte[0] = last_byte[0] >> GDF_VALID_BITSIZE - last_byte_length;
				thrust::copy( last_byte.begin(), last_byte.begin() + 1, right_device_bits + right_num_chars - 1);

				if ( !too_many_bits ) {
					thrust::host_vector<gdf_valid_type> last_byte (2);
					thrust::copy (right_device_bits + right_num_chars - 2, right_device_bits + right_num_chars - 1, last_byte.begin());
					thrust::copy (right_device_bits + right_num_chars - 1, right_device_bits + right_num_chars, last_byte.begin() + 1);
					last_byte[0] = last_byte[0] << last_byte_length | last_byte[1];
					thrust::copy( last_byte.begin(), last_byte.begin() + 1, output_device_bits + output_num_chars - 1);
				} 
			}
		}
		if( last_with_too_many_bits(rhs->size, right_num_chars, prev_len)){
			thrust::host_vector<gdf_valid_type> last_byte (right_device_bits + right_num_chars - 1, right_device_bits + right_num_chars);
			size_t prev_len = get_last_byte_length(lhs->size);
			size_t curr_len = get_right_byte_length(rhs->size, right_num_chars - 1,  prev_len);
			last_byte[0] = last_byte[0] << curr_len;
			last_byte[0] = last_byte[0] >> curr_len;
			thrust::copy( last_byte.begin(), last_byte.begin() + 1, output_device_bits + output_num_chars - 1);
		}
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	return GDF_SUCCESS;
}
