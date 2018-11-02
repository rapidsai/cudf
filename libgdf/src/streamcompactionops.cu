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

#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>


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

#include "thrust_rmm_allocator.h"

//std lib
#include <map>

// thrust::device_vector set to use rmmAlloc and rmmFree.
template <typename T>
using Vector = thrust::device_vector<T, rmm_allocator<T>>;


//wow the freaking example from iterator_adaptpr, what a break right!
template<typename Iterator>
class repeat_iterator
		: public thrust::iterator_adaptor<
		  repeat_iterator<Iterator>, // the first template parameter is the name of the iterator we're creating
		  Iterator                   // the second template parameter is the name of the iterator we're adapting
		  // we can use the default for the additional template parameters
		  >
{
public:
	// shorthand for the name of the iterator_adaptor we're deriving from
	typedef thrust::iterator_adaptor<
			repeat_iterator<Iterator>,
			Iterator
			> super_t;
	__host__ __device__
	repeat_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}
	// befriend thrust::iterator_core_access to allow it access to the private interface below
	friend class thrust::iterator_core_access;
private:
	// repeat each element of the adapted range n times
	unsigned int n;
	// used to keep track of where we began
	const Iterator begin;
	// it is private because only thrust::iterator_core_access needs access to it
	__host__ __device__
	typename super_t::reference dereference() const
	{
		return *(begin + (this->base() - begin) / n);
	}
};



typedef repeat_iterator<thrust::detail::normal_iterator<thrust::device_ptr<gdf_valid_type> > > gdf_valid_iterator;

gdf_size_type get_number_of_bytes_for_valid (gdf_size_type column_size) {
    return sizeof(gdf_valid_type) * (column_size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
}


// note: functor inherits from unary_function
struct modulus_bit_width : public thrust::unary_function<gdf_size_type,gdf_size_type>
{
	gdf_size_type n_bytes;
	gdf_size_type column_size;
	
	modulus_bit_width (gdf_size_type b_nytes, gdf_size_type column_size) {
		this->n_bytes = n_bytes;
		this->column_size = column_size;
	}
	__host__ __device__
	gdf_size_type operator()(gdf_size_type x) const
	{
		gdf_size_type col_position = x / 8;	
        gdf_size_type length_col = n_bytes != col_position+1 ? GDF_VALID_BITSIZE : column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
		//return x % GDF_VALID_BITSIZE;
		return (length_col - 1) - (x % 8);
		// x << 
	}
};


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
 

typedef thrust::transform_iterator<modulus_bit_width, thrust::counting_iterator<gdf_size_type> > bit_position_iterator;


template<typename stencil_type>
struct is_stencil_true
{
	__host__ __device__
	bool operator()(const thrust::tuple<stencil_type, gdf_valid_iterator::value_type, bit_position_iterator::value_type> value)
	{
		gdf_size_type position = thrust::get<2>(value);

		return ((thrust::get<1>(value) >> position) & 1) && (thrust::get<0>(value) != 0);
	}
};

struct is_bit_set
{
	__host__ __device__
	bool operator()(const thrust::tuple< gdf_valid_iterator::value_type, bit_position_iterator::value_type> value)
	{
		gdf_size_type position = thrust::get<1>(value);

		return ((thrust::get<0>(value) >> position) & 1);
	}
}; 

struct bit_mask_pack_op : public thrust::unary_function<int64_t,gdf_valid_type>
{
	__host__ __device__
		gdf_valid_type operator()(const int64_t expanded)
		{
			gdf_valid_type result = 0;
			for(unsigned int i = 0; i < GDF_VALID_BITSIZE; i++){
				// 0, 8, 16, ....,48,  56
				unsigned char byte = (expanded >> ( (GDF_VALID_BITSIZE - 1 - i )  * 8));
				result |= (byte & 1) << i;
			}
			return (result);
		}
};


std::map<gdf_dtype, int16_t> column_type_width = {{GDF_INT8, sizeof(int8_t)}, {GDF_INT16, sizeof(int16_t)},{GDF_INT32, sizeof(int32_t)}, {GDF_INT64, sizeof(int64_t)},
		{GDF_FLOAT32, sizeof(float)}, {GDF_FLOAT64, sizeof(double)} };

//because applying a stencil only needs to know the WIDTH of a type for copying to output, we won't be making a bunch of templated version to store this but rather
//storing a map from gdf_type to width
//TODO: add a way for the space where we store temp bitmaps for compaction be allocated
//on the outside
gdf_error gpu_apply_stencil(gdf_column *lhs, gdf_column * stencil, gdf_column * output){
	//OK: add a rquire here that output and lhs are the same size
	GDF_REQUIRE(output->size == lhs->size, GDF_COLUMN_SIZE_MISMATCH);
	GDF_REQUIRE(lhs->dtype == output->dtype, GDF_DTYPE_MISMATCH);
    GDF_REQUIRE(!lhs->valid || !lhs->null_count, GDF_VALIDITY_UNSUPPORTED);

	//find the width in bytes of this data type
	auto searched_item = column_type_width.find(lhs->dtype);
	int16_t width = searched_item->second; //width in bytes

	searched_item = column_type_width.find(stencil->dtype);
	int16_t stencil_width= searched_item->second; //width in bytes

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	rmm_temp_allocator allocator(stream);
	auto exec = thrust::cuda::par(allocator).on(stream);

	size_t n_bytes = get_number_of_bytes_for_valid(stencil->size);

	bit_position_iterator bit_position_iter(thrust::make_counting_iterator<gdf_size_type>(0), modulus_bit_width(n_bytes, stencil->size));
	gdf_valid_iterator valid_iterator(thrust::detail::make_normal_iterator(thrust::device_pointer_cast(stencil->valid)),GDF_VALID_BITSIZE);
	//TODO: can probably make this happen with some kind of iterator so it can work on any width size

	//zip the stencil and the valid iterator together
	typedef thrust::tuple<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >,gdf_valid_iterator, bit_position_iterator > zipped_stencil_tuple;
	typedef thrust::zip_iterator<zipped_stencil_tuple> zipped_stencil_iterator;

	//what kind of shit is that you might wonder?
	//well basically we are zipping up an iterator to the stencil, one to the bit masks, and one which lets us get the bit position based on our index
	zipped_stencil_iterator zipped_stencil_iter(
			thrust::make_tuple(
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int8_t * )stencil->data)),
					valid_iterator,
					thrust::make_transform_iterator<modulus_bit_width, thrust::counting_iterator<gdf_size_type> >(
							thrust::make_counting_iterator<gdf_size_type>(0),
							modulus_bit_width(n_bytes, stencil->size))
			));

	//NOTE!!!! the output column is getting set to a specific size  but we are NOT compacting the allocation,
	//whoever calls that should handle that
	if(width == 1){
		thrust::detail::normal_iterator<thrust::device_ptr<int8_t> > input_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int8_t *) lhs->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int8_t> > output_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int8_t *) output->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int8_t> > output_end =
				thrust::copy_if(exec,input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >::value_type >());
		output->size = output_end - output_start;
	}else if(width == 2){
		thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > input_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int16_t *) lhs->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > output_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int16_t *) output->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > output_end =
				thrust::copy_if(exec,input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >::value_type >());
		output->size = output_end - output_start;
	}else if(width == 4){
		thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > input_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int32_t *) lhs->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > output_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int32_t *) output->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > output_end =
				thrust::copy_if(exec,input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >::value_type >());
		output->size = output_end - output_start;
	}else if(width == 8){
		thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > input_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) lhs->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > output_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) output->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > output_end =
				thrust::copy_if(exec,input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >::value_type >());
		output->size = output_end - output_start;
	}

	gdf_size_type num_values = lhs->size;
	//TODO:BRING OVER THE BITMASK!!!
	//need to store a prefix sum
	//align to size 8
	Vector<gdf_valid_type> valid_bit_mask; //we are expanding the bit mask to an int8 because I can't envision an algorithm that operates on the bitmask that
	if(num_values % GDF_VALID_BITSIZE != 0){
		valid_bit_mask.resize(num_values + (GDF_VALID_BITSIZE - (num_values % GDF_VALID_BITSIZE))); //align this allocation on GDF_VALID_BITSIZE so we don't have to bounds check
	}else{
		valid_bit_mask.resize(num_values);
	}

	// doesn't require the use for a prefix sum which will have size 8 * num rows which is much larger than this

	typedef thrust::tuple<gdf_valid_iterator, bit_position_iterator > mask_tuple;
	typedef thrust::zip_iterator<mask_tuple> zipped_mask;


	zipped_mask  zipped_mask_iter(
			thrust::make_tuple(
					valid_iterator,
					thrust::make_transform_iterator<modulus_bit_width, thrust::counting_iterator<gdf_size_type> >(
							thrust::make_counting_iterator<gdf_size_type>(0),
							modulus_bit_width(n_bytes, stencil->size))
			)
	);

	typedef thrust::transform_iterator<is_bit_set, zipped_mask > bit_set_iterator;
	bit_set_iterator bit_set_iter = thrust::make_transform_iterator<is_bit_set,zipped_mask>(
			zipped_mask_iter,
			is_bit_set()
	);

	//copy the bitmask to device_vector of int8
	thrust::copy(exec, bit_set_iter, bit_set_iter + num_values, valid_bit_mask.begin());

	//remove the values that don't pass the stencil
	thrust::remove_if(exec,valid_bit_mask.begin(), valid_bit_mask.begin() + num_values,zipped_stencil_iter, is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >::value_type >());

	//recompact the values and store them in the output bitmask
	//we can group them into pieces of 8 because we aligned this earlier on when we made the device_vector
	thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > valid_bit_mask_group_8_iter =
			thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) valid_bit_mask.data().get()));


	//you may notice that we can write out more bytes than our valid_num_bytes, this only happens when we are not aligned to  GDF_VALID_BITSIZE bytes, becasue the
	//arrow standard requires 64 byte alignment, this is a safe assumption to make
	thrust::transform(exec, valid_bit_mask_group_8_iter, valid_bit_mask_group_8_iter + ((num_values + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE),
			thrust::detail::make_normal_iterator(thrust::device_pointer_cast(output->valid)),bit_mask_pack_op());

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

	int type_width = column_type_width[ lhs->dtype ];

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
				using first_iterator_type = thrust::transform_iterator<shift_left,Vector<gdf_valid_type>::iterator>;
				using second_iterator_type = thrust::transform_iterator<shift_right,Vector<gdf_valid_type>::iterator>;
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
								thrust::make_transform_iterator<shift_left, Vector<gdf_valid_type>::iterator >(
										right_device_bits,
										shift_left(shift_bits)),
								
								thrust::make_transform_iterator<shift_right, Vector<gdf_valid_type>::iterator >(
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


