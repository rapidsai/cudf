
#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>


#include <cuda_runtime.h>
#include <vector>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/device_vector.h>

//std lib
#include <map>

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

template <typename LeftType,typename RightType,typename ResultType >
struct gdf_equals_op : public thrust::binary_function< LeftType, RightType, ResultType>
{
	__host__ __device__
	ResultType operator()(LeftType x, RightType y)
	{
		return x == y;
	}
};

template <typename LeftType,typename RightType,typename ResultType >
struct gdf_not_equals_op : public thrust::binary_function< LeftType, RightType, ResultType>
{
	__host__ __device__
	ResultType operator()(LeftType x, RightType y)
	{
		return x != y;
	}
};

template <typename LeftType,typename RightType,typename ResultType >
struct gdf_greater_than_op : public thrust::binary_function< LeftType, RightType, ResultType>
{
	__host__ __device__
	ResultType operator()(LeftType x, RightType y)
	{
		return x > y;
	}
};

template <typename LeftType,typename RightType,typename ResultType >
struct gdf_greater_than_or_equals_op : public thrust::binary_function< LeftType, RightType, ResultType>
{
	__host__ __device__
	ResultType operator()(LeftType x, RightType y)
	{
		return x >= y;
	}
};

template <typename LeftType,typename RightType,typename ResultType >
struct gdf_less_than_op : public thrust::binary_function< LeftType, RightType, ResultType>
{
	__host__ __device__
	ResultType operator()(LeftType x, RightType y)
	{
		return x > y;
	}
};

template <typename LeftType,typename RightType,typename ResultType >
struct gdf_less_than_or_equals_op : public thrust::binary_function< LeftType, RightType, ResultType>
{
	__host__ __device__
	ResultType operator()(LeftType x, RightType y)
	{
		return x >= y;
	}
};





/**
 * @brief takes two columns data and their valid bitmasks and performs a comparison operation returning a column of type bool
 *
 * Takes two thrust::iterator_adaptor implemented iterators and performs a filter operation on them that it outputs into a third thust::iterator_adaptor dervied iterator.
 * We are not making assumptions about what kind of data is being passed into it for these pointers so
 *
 * @param begin_left an iterator that implements thrust::iterator_adaptor
 * @param begin_right an iterator that implements thrust::iterator_adaptor
 * @param result an iterator that implements thrust::iterator_adaptor
 * @param operation an enum telling us what kind of comparision operation we are trying to do
 * @param num_values the number of rows in our columns
 * @param valid_left left column null bitmask (1 = not null)
 * @param valid_right right column null bitmask
 * @param valid_out output column null bitmask
 * @param left_null_count tells us if there are any nulls in the left column
 * @param right_null_count tells us if there are any nulls in the right column
 * @param
 */
template<typename IteratorTypeLeft, typename IteratorTypeRight, typename IteratorTypeResult,
class LeftType = typename IteratorTypeLeft::value_type, class RightType = typename IteratorTypeRight::value_type, class ResultType = typename IteratorTypeResult::value_type>
void gpu_filter_op(IteratorTypeLeft begin_left, IteratorTypeRight begin_right, IteratorTypeResult result, gdf_comparison_operator operation, gdf_size_type num_values,
		gdf_valid_type * valid_left, gdf_valid_type * valid_right, gdf_valid_type * valid_out, gdf_size_type left_null_count, gdf_size_type right_null_count, gdf_size_type & out_null_count, cudaStream_t stream){

	//TODO: be able to pass in custom comparison operators so we can handle types that have not implemented these oeprators

	IteratorTypeLeft end_left = begin_left + num_values;


	//regardless of nulls we perform the same operation
	//the nulls we are going to and together
	if (operation == GDF_EQUALS) {

		gdf_equals_op<LeftType, RightType, ResultType> op;
		thrust::transform(thrust::cuda::par.on(stream), begin_left, end_left, begin_right, result, op);

	} else if (operation == GDF_NOT_EQUALS) {
		gdf_not_equals_op<LeftType, RightType, ResultType> op;
		thrust::transform(thrust::cuda::par.on(stream), begin_left, end_left, begin_right, result, op);

	} else if (operation == GDF_GREATER_THAN_OR_EQUALS) {
		gdf_greater_than_or_equals_op<LeftType, RightType, ResultType> op;
		thrust::transform(thrust::cuda::par.on(stream), begin_left, end_left, begin_right, result, op);

	} else if (operation == GDF_GREATER_THAN) {
		gdf_greater_than_op<LeftType, RightType, ResultType> op;
		thrust::transform(thrust::cuda::par.on(stream), begin_left, end_left, begin_right, result, op);

	} else if (operation == GDF_LESS_THAN) {
		gdf_less_than_op<LeftType, RightType, ResultType> op;
		thrust::transform(thrust::cuda::par.on(stream), begin_left, end_left, begin_right, result, op);

	} else if (operation == GDF_LESS_THAN_OR_EQUALS) {
		gdf_less_than_or_equals_op<LeftType, RightType, ResultType> op;
		thrust::transform(thrust::cuda::par.on(stream),begin_left, end_left, begin_right, result, op);

	}

	gdf_size_type num_chars_bitmask = ( ( num_values +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
	//TODO: if we could make sure that these things aligned on 8 byte boundaries we could probable do this more efficiently as an unsigned long long
	if((left_null_count == 0) && (right_null_count == 0) ){
		thrust::device_ptr<gdf_valid_type> valid_out_ptr = thrust::device_pointer_cast(valid_out);
		gdf_valid_type max_char = 255;
		thrust::fill(thrust::cuda::par.on(stream),thrust::detail::make_normal_iterator(valid_out_ptr),thrust::detail::make_normal_iterator(valid_out_ptr + num_chars_bitmask),max_char);
		//we have no nulls so set all the bits in gdf_valid_type to 1
		out_null_count = 0;

	}else if(valid_right == valid_left){
		//this is often the case if we are passing in the same column to operate on itself
		//or when we are sending something like  a constant_iterator for the right hand side, allows us some shortcuts

		cudaError_t error = cudaMemcpyAsync(valid_out,valid_left,num_chars_bitmask * sizeof(gdf_valid_type),cudaMemcpyDeviceToDevice,stream);


		out_null_count = left_null_count;

	}else{


		thrust::device_ptr<gdf_valid_type> valid_out_ptr = thrust::device_pointer_cast(valid_out);
		thrust::device_ptr<gdf_valid_type> valid_left_ptr = thrust::device_pointer_cast(valid_left);
		//here we are basically figuring out what is the last pointed to unsigned char that can contain part of the bitmask
		thrust::device_ptr<gdf_valid_type> valid_left_end_ptr = thrust::device_pointer_cast(valid_left + num_chars_bitmask );
		thrust::device_ptr<gdf_valid_type> valid_right_ptr = thrust::device_pointer_cast(valid_right);

		//TODO:: I am assuming
		thrust::transform(thrust::cuda::par.on(stream), thrust::detail::make_normal_iterator(valid_left_ptr),
				thrust::detail::make_normal_iterator(valid_left_end_ptr), thrust::detail::make_normal_iterator(valid_right_ptr),
				thrust::detail::make_normal_iterator(valid_out_ptr), thrust::bit_and<gdf_valid_type>());

		//figure out how to count nulls from the bitmask on gpu
		//a fast algorithm would create a look up table that stores how many bits are set for each value
		//then we make a thrust::permutation iterator based on those values, we can hold it in gpu memory if we want o rmake it part of initializing
		//we can also copy it in adhoc its really not that big at all


		char * last_char = new char[1];
		cudaError_t error = cudaMemcpyAsync(last_char,valid_out + ( num_chars_bitmask-1),sizeof(gdf_valid_type),cudaMemcpyDeviceToHost,stream);


		thrust::device_vector<gdf_valid_type> bit_mask_null_counts_device(bit_mask_null_counts);

		//this permutation iterator makes it so that each char basically gets replaced with its number of null counts
		//so if you sum up this perm iterator you add up all of the counts for null values per unsigned char
		thrust::permutation_iterator<thrust::device_vector<gdf_valid_type>::iterator,thrust::detail::normal_iterator<thrust::device_ptr<gdf_valid_type> > >
		null_counts_iter( bit_mask_null_counts_device.begin(),thrust::detail::make_normal_iterator(valid_out_ptr));

		//you will notice that we subtract the number of zeros we found in the last character
		out_null_count = thrust::reduce(null_counts_iter, null_counts_iter + num_chars_bitmask) - gdf_num_bits_zero_after_pos(*last_char,num_values % GDF_VALID_BITSIZE );

		delete[] last_char;

	}

	cudaStreamSynchronize(stream);

}


template<typename T>
gdf_error gpu_comparison_static_templated(gdf_column *lhs, T value, gdf_column *output,gdf_comparison_operator operation){
	GDF_REQUIRE(lhs->size == output->size, GDF_COLUMN_SIZE_MISMATCH);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	if(lhs->dtype == GDF_INT8){
		thrust::device_ptr<int8_t> left_ptr((int8_t *) lhs->data);
		if(output->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> out_ptr((float *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> out_ptr((double *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}

	}else if(lhs->dtype == GDF_INT16){
		thrust::device_ptr<int16_t> left_ptr((int16_t *) lhs->data);

		if(output->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> out_ptr((float *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> out_ptr((double *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}

	}else if(lhs->dtype == GDF_INT32){
		thrust::device_ptr<int32_t> left_ptr((int32_t *) lhs->data);

		if(output->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> out_ptr((float *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> out_ptr((double *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}

	}else if(lhs->dtype == GDF_INT64){
		thrust::device_ptr<int64_t> left_ptr((int64_t *) lhs->data);

		if(output->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> out_ptr((float *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> out_ptr((double *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}

	}else if(lhs->dtype == GDF_FLOAT32){
		thrust::device_ptr<float> left_ptr((float *) lhs->data);
		if(output->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> out_ptr((float *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> out_ptr((double *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}

	}else if(lhs->dtype == GDF_FLOAT64){
		thrust::device_ptr<double> left_ptr((double *) lhs->data);
		if(output->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> out_ptr((float *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}else if(output->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> out_ptr((double *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
					lhs->null_count,lhs->null_count,output->null_count,stream
			);

		}

	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);


	return GDF_SUCCESS;

}

gdf_error gpu_comparison_static_i8(gdf_column *lhs, int8_t value, gdf_column *output,gdf_comparison_operator operation){
	return gpu_comparison_static_templated(lhs, value, output,operation);
}

gdf_error gpu_comparison_static_i16(gdf_column *lhs, int16_t value, gdf_column *output,gdf_comparison_operator operation){
	return gpu_comparison_static_templated(lhs, value, output,operation);
}

gdf_error gpu_comparison_static_i32(gdf_column *lhs, int32_t value, gdf_column *output,gdf_comparison_operator operation){
	return gpu_comparison_static_templated(lhs, value, output,operation);
}

gdf_error gpu_comparison_static_i64(gdf_column *lhs, int64_t value, gdf_column *output,gdf_comparison_operator operation){
	return gpu_comparison_static_templated(lhs, value, output,operation);
}

gdf_error gpu_comparison_static_f32(gdf_column *lhs, float value, gdf_column *output,gdf_comparison_operator operation){
	return gpu_comparison_static_templated(lhs, value, output,operation);
}

gdf_error gpu_comparison_static_f64(gdf_column *lhs, double value, gdf_column *output,gdf_comparison_operator operation){
	return gpu_comparison_static_templated(lhs, value, output,operation);
}




gdf_error gpu_comparison(gdf_column *lhs, gdf_column *rhs, gdf_column *output,gdf_comparison_operator operation){
	GDF_REQUIRE(lhs->size == rhs->size, GDF_COLUMN_SIZE_MISMATCH);
	GDF_REQUIRE(lhs->size == output->size, GDF_COLUMN_SIZE_MISMATCH);
	//TODO: consider adding more requirements like that the columns be well defined in their type
	//I commented this out because I am not sure if we want to require the output be an int8
	//GDF_REQUIRE(output->dtype == GDF_INT8,GDF_UNSUPPORTED_DTYPE);

	// SO... I know the follow code looks, questionable, but the other option is to have a shitload of function definitions
	// given that our gdf_columns very conveniently carry around their types with them, this seems to be to be simpler
	// than having tons of function definitions. it also makes it so much nicer to just type gpu_filter(lhs,rhs,output);
	// also we are making it so that we can send any types here, the only one which is debatable I feel is output which
	// we could decide to always have be an int8 since the output is a boolean



	cudaStream_t stream;
	cudaStreamCreate(&stream);



	if(lhs->dtype == GDF_INT8){
		thrust::device_ptr<int8_t> left_ptr((int8_t *) lhs->data);
		if(rhs->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> right_ptr((int8_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> right_ptr((int16_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> right_ptr((int32_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> right_ptr((int64_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> right_ptr((float *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> right_ptr((double *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}
	}else if(lhs->dtype == GDF_INT16){
		thrust::device_ptr<int16_t> left_ptr((int16_t *) lhs->data);
		if(rhs->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> right_ptr((int8_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> right_ptr((int16_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> right_ptr((int32_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> right_ptr((int64_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> right_ptr((float *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> right_ptr((double *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}
	}else if(lhs->dtype == GDF_INT32){
		thrust::device_ptr<int32_t> left_ptr((int32_t *) lhs->data);
		if(rhs->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> right_ptr((int8_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> right_ptr((int16_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> right_ptr((int32_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> right_ptr((int64_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> right_ptr((float *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> right_ptr((double *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}
	}else if(lhs->dtype == GDF_INT64){
		thrust::device_ptr<int64_t> left_ptr((int64_t *) lhs->data);
		if(rhs->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> right_ptr((int8_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> right_ptr((int16_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> right_ptr((int32_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> right_ptr((int64_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> right_ptr((float *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> right_ptr((double *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}
	}else if(lhs->dtype == GDF_FLOAT32){
		thrust::device_ptr<float> left_ptr((float *) lhs->data);
		if(rhs->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> right_ptr((int8_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> right_ptr((int16_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> right_ptr((int32_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> right_ptr((int64_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> right_ptr((float *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> right_ptr((double *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}
	}else if(lhs->dtype == GDF_FLOAT64){
		thrust::device_ptr<double> left_ptr((double *) lhs->data);
		if(rhs->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> right_ptr((int8_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> right_ptr((int16_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> right_ptr((int32_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> right_ptr((int64_t *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> right_ptr((float *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}else if(rhs->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> right_ptr((double *) rhs->data);
			if(output->dtype == GDF_INT8){

				thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}else if(output->dtype == GDF_INT16){
				thrust::device_ptr<int16_t> out_ptr((int16_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT32){
				thrust::device_ptr<int32_t> out_ptr((int32_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_INT64){
				thrust::device_ptr<int64_t> out_ptr((int64_t *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT32){
				thrust::device_ptr<float> out_ptr((float *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);

			}else if(output->dtype == GDF_FLOAT64){
				thrust::device_ptr<double> out_ptr((double *) output->data);
				gpu_filter_op(
						thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
						thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
						lhs->null_count,rhs->null_count,output->null_count,stream
				);
			}

		}
	}



	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);



	return GDF_SUCCESS;
}


/*
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

template<typename stencil_type>
struct is_stencil_true
{
	__host__ __device__
	bool operator()(const thrust::tuple<stencil_type, gdf_valid_iterator::value_type> value)
	{
		return (thrust::get<1>(value) == 1) && (thrust::get<0>(value) != 0);
	}
};



std::map<gdf_dtype, int16_t> column_type_width = {{GDF_INT8, sizeof(int8_t)}, {GDF_INT16, sizeof(int16_t)},{GDF_INT32, sizeof(int32_t)}, {GDF_INT64, sizeof(int64_t)},
		{GDF_FLOAT32, sizeof(float)}, {GDF_FLOAT64, sizeof(double)} };
//because applying a stencil only needs to know the WIDTH of a type for copying to output, we won't be making a bunch of templated version to store this but rather
//storing a map from gdf_type to width
gdf_error gpu_apply_stencil(gdf_column *lhs, gdf_column * stencil, gdf_column * output){
	//TODO: add a rquire here that output and lhs are the same size


	//find the width in bytes of this data type
	auto searched_item = column_type_width.find(lhs->dtype);
	int16_t width = searched_item->second; //width in bytes

	searched_item = column_type_width.find(stencil->dtype);
	int16_t stencil_width= searched_item->second; //width in bytes

	cudaStream_t stream;
	cudaStreamCreate(&stream);



	gdf_valid_iterator valid_iterator(thrust::detail::make_normal_iterator(thrust::device_pointer_cast(stencil->valid)),GDF_VALID_BITSIZE);
	//TODO: can probably make this happen with some kind of iterator so it can work on any width size
	if(stencil_width == 1){
		//zip the stencil and the valid iterator together
		typedef thrust::tuple<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >,gdf_valid_iterator> zipped_stencil_tuple;
		typedef thrust::zip_iterator<zipped_stencil_tuple> zipped_stencil_iterator;

		zipped_stencil_iterator zipped_stencil_iter(thrust::make_tuple(thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int8_t * )stencil->data)),valid_iterator));

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

	}else if(stencil_width == 2){
		typedef thrust::tuple<thrust::detail::normal_iterator<thrust::device_ptr<int16_t> >,gdf_valid_iterator> zipped_stencil_tuple;
		typedef thrust::zip_iterator<zipped_stencil_tuple> zipped_stencil_iterator;

		zipped_stencil_iterator zipped_stencil_iter(thrust::make_tuple(thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int16_t * )stencil->data)),valid_iterator));

		if(width == 1){
			thrust::detail::normal_iterator<thrust::device_ptr<int8_t> > input_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int8_t *) lhs->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int8_t> > output_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int8_t *) output->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int8_t> > output_end =
					thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int16_t> >::value_type >());
			output->size = output_end - output_start;
		}else if(width == 2){
			thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > input_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int16_t *) lhs->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > output_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int16_t *) output->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > output_end =
					thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int16_t> >::value_type >());
			output->size = output_end - output_start;
		}else if(width == 4){
			thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > input_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int32_t *) lhs->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > output_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int32_t *) output->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > output_end =
					thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int16_t> >::value_type >());
			output->size = output_end - output_start;
		}else if(width == 8){
			thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > input_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) lhs->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > output_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) output->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > output_end =
					thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int16_t> >::value_type >());
			output->size = output_end - output_start;
		}
	}else if(stencil_width == 4){
		typedef thrust::tuple<thrust::detail::normal_iterator<thrust::device_ptr<int32_t> >,gdf_valid_iterator> zipped_stencil_tuple;
		typedef thrust::zip_iterator<zipped_stencil_tuple> zipped_stencil_iterator;

		zipped_stencil_iterator zipped_stencil_iter(thrust::make_tuple(thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int32_t * )stencil->data)),valid_iterator));

		if(width == 1){
			thrust::detail::normal_iterator<thrust::device_ptr<int8_t> > input_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int8_t *) lhs->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int8_t> > output_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int8_t *) output->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int8_t> > output_end =
					thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int32_t> >::value_type >());
			output->size = output_end - output_start;
		}else if(width == 2){
			thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > input_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int16_t *) lhs->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > output_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int16_t *) output->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > output_end =
					thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int32_t> >::value_type >());
			output->size = output_end - output_start;
		}else if(width == 4){
			thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > input_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int32_t *) lhs->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > output_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int32_t *) output->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > output_end =
					thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int32_t> >::value_type >());
			output->size = output_end - output_start;
		}else if(width == 8){
			thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > input_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) lhs->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > output_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) output->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > output_end =
					thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int32_t> >::value_type >());
			output->size = output_end - output_start;
		}
	}else if(stencil_width == 8){
		typedef thrust::tuple<thrust::detail::normal_iterator<thrust::device_ptr<int64_t> >,gdf_valid_iterator> zipped_stencil_tuple;
		typedef thrust::zip_iterator<zipped_stencil_tuple> zipped_stencil_iterator;

		zipped_stencil_iterator zipped_stencil_iter(thrust::make_tuple(thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t * )stencil->data)),valid_iterator));

		if(width == 1){
			thrust::detail::normal_iterator<thrust::device_ptr<int8_t> > input_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int8_t *) lhs->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int8_t> > output_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int8_t *) output->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int8_t> > output_end =
					thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int64_t> >::value_type >());
			output->size = output_end - output_start;
		}else if(width == 2){
			thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > input_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int16_t *) lhs->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > output_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int16_t *) output->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int16_t> > output_end =
					thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int64_t> >::value_type >());
			output->size = output_end - output_start;
		}else if(width == 4){
			thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > input_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int32_t *) lhs->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > output_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int32_t *) output->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int32_t> > output_end =
					thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int64_t> >::value_type >());
			output->size = output_end - output_start;
		}else if(width == 8){
			thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > input_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) lhs->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > output_start =
					thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) output->data));
			thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > output_end =
					thrust::copy_if(thrust::cuda::par.on(stream),input_start,input_start + lhs->size,zipped_stencil_iter,output_start,is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int64_t> >::value_type >());
			output->size = output_end - output_start;
		}
	}

	cudaStreamSynchronize(stream);

	cudaStreamDestroy(stream);

	return GDF_SUCCESS;

}*/
