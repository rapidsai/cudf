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

//std lib
#include <map>




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

size_t get_number_of_bytes_for_valid (size_t column_size) {
    return sizeof(gdf_valid_type) * (column_size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
}


// note: functor inherits from unary_function
struct modulus_bit_width : public thrust::unary_function<gdf_size_type,gdf_size_type>
{
	size_t n_bytes;
	size_t column_size;
	
	modulus_bit_width (size_t b_nytes, size_t column_size) {
		this->n_bytes = n_bytes;
		this->column_size = column_size;
	}
	__host__ __device__
	gdf_size_type operator()(gdf_size_type x) const
	{
		int col_position = x / 8;	
        int length_col = n_bytes != col_position+1 ? GDF_VALID_BITSIZE : column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
		//return x % GDF_VALID_BITSIZE;
		return (length_col - 1) - (x % 8);
		// x << 
	}
};

// note: functor inherits from unary_function
struct shift_operator : public thrust::unary_function<gdf_valid_type,gdf_valid_type>
{
	int num_bits;
	shift_operator (int num_bits) {
		this->num_bits = num_bits;
	}
	
	__host__ __device__
	gdf_valid_type operator()(gdf_valid_type x) const
	{
		return x << num_bits;
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
/*
	//before
	for(int i = 0; i < column->size; i++){
		int col_position = i / 8;
		int bit_offset = i % 8;
	}
	//now 
	for(int i = 0; i < column->size; i++) {
        int col_position =  i / 8;
		// i = col_position * 8
        int length_col = n_bytes != col_position+1 ? GDF_VALID_BITSIZE : column->size - GDF_VALID_BITSIZE * (n_bytes - 1);
        int bit_offset =  (length_col - 1) - (i % 8);
    }
*/
struct bit_mask_pack_op : public thrust::unary_function<int64_t,gdf_valid_type>
{
	__host__ __device__
		gdf_valid_type operator()(const int64_t expanded)
		{
			gdf_valid_type result = 0;
			for(int i = 0; i < GDF_VALID_BITSIZE; i++){
				// 0, 8, 16, ....,48,  56
				unsigned char byte = (expanded >> ( (GDF_VALID_BITSIZE - 1 - i )  * 8));
				result |= (byte & 1) << i;
			}
			return (result);
		}
};

/*
damn im passing out, ok how im going to do this, ill derefence the fucking entire character, offsetting with the iterator i make however the fuck that works right?
		then i will make it so that i zip a counting iterator as well and use that counting iterator to figure out which bit to and
		i think something like a repeat iterator or somethign else will suffce

 */

std::map<gdf_dtype, int16_t> column_type_width = {{GDF_INT8, sizeof(int8_t)}, {GDF_INT16, sizeof(int16_t)},{GDF_INT32, sizeof(int32_t)}, {GDF_INT64, sizeof(int64_t)},
		{GDF_FLOAT32, sizeof(float)}, {GDF_FLOAT64, sizeof(double)} };
//because applying a stencil only needs to know the WIDTH of a type for copying to output, we won't be making a bunch of templated version to store this but rather
//storing a map from gdf_type to width
gdf_error gpu_apply_stencil(gdf_column *lhs, gdf_column * stencil, gdf_column * output){
	//OK: add a rquire here that output and lhs are the same size
	GDF_REQUIRE(output->size == lhs->size, GDF_COLUMN_SIZE_MISMATCH);


	//find the width in bytes of this data type
	auto searched_item = column_type_width.find(lhs->dtype);
	int16_t width = searched_item->second; //width in bytes

	searched_item = column_type_width.find(stencil->dtype);
	int16_t stencil_width= searched_item->second; //width in bytes

	cudaStream_t stream;
	cudaStreamCreate(&stream);

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
				thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >::value_type >());
		output->size = output_end - output_start;
	}else if(width == 2){
		thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > input_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int16_t *) lhs->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > output_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int16_t *) output->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > output_end =
				thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >::value_type >());
		output->size = output_end - output_start;
	}else if(width == 4){
		thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > input_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int32_t *) lhs->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > output_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int32_t *) output->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > output_end =
				thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >::value_type >());
		output->size = output_end - output_start;
	}else if(width == 8){
		thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > input_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) lhs->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > output_start =
				thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) output->data));
		thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > output_end =
				thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >::value_type >());
		output->size = output_end - output_start;
	}

	gdf_size_type num_values = lhs->size;
	//TODO:BRING OVER THE BITMASK!!!
	//need to store a prefix sum
	//align to size 8
	thrust::device_vector<gdf_valid_type> valid_bit_mask; //we are expanding the bit mask to an int8 because I can't envision an algorithm that operates on the bitmask that
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
	thrust::copy(thrust::cuda::par.on(stream), bit_set_iter, bit_set_iter + num_values, valid_bit_mask.begin());

	//remove the values that don't pass the stencil
	thrust::remove_if(thrust::cuda::par.on(stream),valid_bit_mask.begin(), valid_bit_mask.begin() + num_values,zipped_stencil_iter, is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >::value_type >());

	//recompact the values and store them in the output bitmask
	//we can group them into pieces of 8 because we aligned this earlier on when we made the device_vector
	thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > valid_bit_mask_group_8_iter =
			thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) valid_bit_mask.data().get()));


	//you may notice that we can write out more bytes than our valid_num_bytes, this only happens when we are not aligned to  GDF_VALID_BITSIZE bytes, becasue the
	//arrow standard requires 64 byte alignment, this is a safe assumption to make
	thrust::transform(thrust::cuda::par.on(stream), valid_bit_mask_group_8_iter, valid_bit_mask_group_8_iter + ((num_values + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE),
			thrust::detail::make_normal_iterator(thrust::device_pointer_cast(output->valid)),bit_mask_pack_op());

	cudaStreamSynchronize(stream);

	cudaStreamDestroy(stream);

	return GDF_SUCCESS;

}

size_t  valid_left_length(gdf_column *column) {
    int  n_bytes = get_number_of_bytes_for_valid(column->size);
    size_t length = column->size - GDF_VALID_BITSIZE * (n_bytes - 1);
    if (n_bytes == 1 ) {
        length = column->size;
    }
    return  length;
}

struct valid_array_iterator{
    gdf_column* column;
    size_t iter;
    size_t n_bytes;
    size_t init_length;
    gdf_valid_type init_value; 
	size_t number_of_calls;
    valid_array_iterator(gdf_column* column, gdf_valid_type init, size_t init_size, size_t  init_index = 1) {
        this->column = column;
        this->n_bytes =  sizeof(int8_t) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
        this->init_value = init;
        this->init_length = init_size;
        this->iter = init_index;
		this->number_of_calls = 0;
    }

    template <typename Functor>
    void for_each(Functor output_functor) {
        gdf_valid_type prev = this->init_value;
        size_t prev_length = this->init_length;

        gdf_valid_type current;
        size_t current_length;
        std::tie(current, current_length) = next_node();

        size_t length = column->size - GDF_VALID_BITSIZE * (n_bytes - 1);
        while (true) {
            auto result = concat_bins(prev, current, prev_length, current_length, last_with_too_many_bits(), length);
            output_functor(result, iter);
			number_of_calls++;
            auto result_size = prev_length + current_length;
            if ( !has_next() )
                break;
            prev_length = this->init_length;
            prev = this->column->valid[iter - 1];
            std::tie(current, current_length) = next_node();
        }
        if (last_with_too_many_bits()) {
            auto len = length - current_length;
            auto result = this->column->valid[n_bytes - 1];
            result = result << current_length;
            result = result >> current_length;
            output_functor(result, iter + 1);
			number_of_calls++;
        }
    }
    bool last_with_too_many_bits() {
        size_t length = column->size - GDF_VALID_BITSIZE * (n_bytes - 1);
        if (iter == n_bytes) { // the last one
            // the last one has to many bits
            if (this->init_length + length > GDF_VALID_BITSIZE) {
                return true;
            }
        }
        return false;
    }

    std::tuple<gdf_valid_type, size_t> next_node() {
        gdf_valid_type valid;
        size_t length = column->size - GDF_VALID_BITSIZE * (n_bytes - 1);
        if (iter == n_bytes - 1) { // the last one
            valid = this->column->valid[iter];
            // the last one has to many bits
            if (this->init_length + length > GDF_VALID_BITSIZE) {
                length = GDF_VALID_BITSIZE - this->init_length;
            }
        }
        else {
            length = GDF_VALID_BITSIZE - this->init_length;
            valid = this->column->valid[iter] >> this->init_length;
        }
        iter++;
        return std::make_tuple(valid, length);
    }

	auto concat_bins (gdf_valid_type A, gdf_valid_type B, int len_a, int len_b, bool has_next = false, size_t right_length = -1) -> gdf_valid_type  {
		A = A << len_b;
		if (!has_next) {
			B = B << len_a;
			B = B >> len_a;
		} else {
			B = B >> right_length - len_b;
		}
		return  (A | B);
	}
    bool has_next() {
        return iter < n_bytes;
    }
};

gdf_valid_type * gdf_valid_from_device(gdf_column* column, cudaStream_t &stream) {
    gdf_valid_type * host_valid_out;
    size_t n_bytes = get_number_of_bytes_for_valid(column->size);
    host_valid_out = new gdf_valid_type[n_bytes];
    cudaMemcpyAsync(host_valid_out, column->valid, n_bytes, cudaMemcpyDeviceToHost, stream);
    return host_valid_out;
}

void gdf_copy_valid_from_host_to_device (gdf_column *column, gdf_column *device,  size_t lnbytes, size_t n_bytes, cudaStream_t &stream) {
    gdf_valid_type *host_valid = column->valid;
    cudaMemcpyAsync(device->valid + sizeof(gdf_valid_type) * (lnbytes - 1), host_valid, n_bytes, cudaMemcpyHostToDevice, stream);
}

gdf_error gpu_concat(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
{
	GDF_REQUIRE( (lhs->dtype == output->dtype ) && ( rhs->dtype == output->dtype), GDF_VALIDITY_MISSING);
	GDF_REQUIRE(output->size == lhs->size + rhs->size, GDF_COLUMN_SIZE_MISMATCH);
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	//@todo: check if  lsh->dtype is NOT GDF_invalid
	int type_width = column_type_width[ lhs->dtype ];

	//copy data 
	cudaMemcpyAsync(output->data, lhs->data, type_width * lhs->size, cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync( (void *)( (int8_t*) (output->data) + type_width * lhs->size), rhs->data, type_width * rhs->size, cudaMemcpyDeviceToDevice, stream);
	
	int lnbytes = get_number_of_bytes_for_valid(lhs->size);
	int rnbytes = get_number_of_bytes_for_valid(rhs->size);
  
	if (lnbytes > 1) {
		cudaMemcpyAsync(output->valid, lhs->valid, sizeof(gdf_valid_type) * (lnbytes - 1), cudaMemcpyDeviceToDevice, stream);
	}
	int last_char_index = sizeof(gdf_valid_type) * lnbytes - 1;
	gdf_valid_type* left_char = new gdf_valid_type[1];
	cudaError_t error = cudaMemcpyAsync(left_char, &lhs->valid[last_char_index], sizeof(gdf_valid_type), cudaMemcpyDeviceToHost, stream);
	size_t len_prev = valid_left_length(lhs);

	if (lnbytes == 0) {
        cudaMemcpyAsync(output->valid, rhs->valid, sizeof(gdf_valid_type) * rnbytes, cudaMemcpyDeviceToDevice, stream);
    }
    else if (rhs->size > 0) {
		gdf_column rhs_host = *rhs;
        rhs_host.valid = gdf_valid_from_device(rhs, stream);
        gdf_valid_type * host_output_valid = new gdf_valid_type[rnbytes];
		valid_array_iterator iter(&rhs_host, *left_char, len_prev, 0);
		iter.for_each( [&host_output_valid, &lnbytes] (gdf_valid_type result, size_t iter) {
 			std::memcpy ( host_output_valid + sizeof(gdf_valid_type) * (iter - 1) , &result, sizeof(gdf_valid_type));
        });
		cudaMemcpyAsync(output->valid + sizeof(gdf_valid_type) * (lnbytes - 1), host_output_valid, iter.number_of_calls, cudaMemcpyHostToDevice, stream);
	
		delete [] host_output_valid;
		delete [] left_char;
	} else if (lnbytes == 1){
		cudaMemcpyAsync(output->valid, lhs->valid, sizeof(gdf_valid_type), cudaMemcpyDeviceToDevice, stream);
    }
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	return GDF_SUCCESS;
}
