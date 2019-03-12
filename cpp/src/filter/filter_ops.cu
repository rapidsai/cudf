
#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"
#include "cudf/functions.h"
#include "bitmask/bitmask_ops.h"
#include "rmm/thrust_rmm_allocator.h"

#include <cuda_runtime.h>
#include <vector>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_adaptor.h>

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
		thrust::transform(rmm::exec_policy(stream)->on(stream), begin_left, end_left, begin_right, result, op);

	} else if (operation == GDF_NOT_EQUALS) {
		gdf_not_equals_op<LeftType, RightType, ResultType> op;
		thrust::transform(rmm::exec_policy(stream)->on(stream), begin_left, end_left, begin_right, result, op);

	} else if (operation == GDF_GREATER_THAN_OR_EQUALS) {
		gdf_greater_than_or_equals_op<LeftType, RightType, ResultType> op;
		thrust::transform(rmm::exec_policy(stream)->on(stream), begin_left, end_left, begin_right, result, op);

	} else if (operation == GDF_GREATER_THAN) {
		gdf_greater_than_op<LeftType, RightType, ResultType> op;
		thrust::transform(rmm::exec_policy(stream)->on(stream), begin_left, end_left, begin_right, result, op);

	} else if (operation == GDF_LESS_THAN) {
		gdf_less_than_op<LeftType, RightType, ResultType> op;
		thrust::transform(rmm::exec_policy(stream)->on(stream), begin_left, end_left, begin_right, result, op);

	} else if (operation == GDF_LESS_THAN_OR_EQUALS) {
		gdf_less_than_or_equals_op<LeftType, RightType, ResultType> op;
		thrust::transform(rmm::exec_policy(stream)->on(stream),begin_left, end_left, begin_right, result, op);

	}


	//TODO: if we could make sure that these things aligned on 8 byte boundaries we could probable do this more efficiently as an unsigned long long
	if((left_null_count == 0) && (right_null_count == 0) ){

		gdf_error error = all_bitmask_on(valid_out,  out_null_count,num_values,stream);
	}else if(valid_right == valid_left){
		//this is often the case if we are passing in the same column to operate on itself
		//or when we are sending something like  a constant_iterator for the right hand side, allows us some shortcuts
		gdf_size_type num_bitmask_elements = gdf_num_bitmask_elements(num_values);
		cudaError_t error = cudaMemcpyAsync(valid_out,valid_left,num_bitmask_elements * sizeof(gdf_valid_type),cudaMemcpyDeviceToDevice,stream);
		out_null_count = left_null_count;

	}else{
		apply_bitmask_to_bitmask( out_null_count,  valid_out, valid_left, valid_right,
				 stream, num_values);

	}

	cudaStreamSynchronize(stream);

}


// stencil: plantilla! 
// 
template<typename T>
gdf_error gdf_comparison_static_templated(gdf_column *lhs, T value, gdf_column *output,gdf_comparison_operator operation){
	GDF_REQUIRE(lhs->size == output->size, GDF_COLUMN_SIZE_MISMATCH);

	GDF_REQUIRE(output->dtype == GDF_INT8, GDF_COLUMN_SIZE_MISMATCH);
	
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	if(lhs->dtype == GDF_INT8){
		thrust::device_ptr<int8_t> left_ptr((int8_t *) lhs->data);
		
		thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
		gpu_filter_op(
				thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
				thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
				lhs->null_count,lhs->null_count,output->null_count,stream
		);

	}else if(lhs->dtype == GDF_INT16){
		thrust::device_ptr<int16_t> left_ptr((int16_t *) lhs->data);

		thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
		gpu_filter_op(
				thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
				thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
				lhs->null_count,lhs->null_count,output->null_count,stream
		);
 
	}else if(lhs->dtype == GDF_INT32){
		thrust::device_ptr<int32_t> left_ptr((int32_t *) lhs->data);

		thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
		gpu_filter_op(
				thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
				thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
				lhs->null_count,lhs->null_count,output->null_count,stream
		);
	}else if(lhs->dtype == GDF_INT64){
		thrust::device_ptr<int64_t> left_ptr((int64_t *) lhs->data);

		thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
		gpu_filter_op(
				thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
				thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
				lhs->null_count,lhs->null_count,output->null_count,stream
		);

	}else if(lhs->dtype == GDF_FLOAT32){
		thrust::device_ptr<float> left_ptr((float *) lhs->data);
		thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
		gpu_filter_op(
				thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
				thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
				lhs->null_count,lhs->null_count,output->null_count,stream
		);

	}else if(lhs->dtype == GDF_FLOAT64){
		thrust::device_ptr<double> left_ptr((double *) lhs->data);
		thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
		gpu_filter_op(
				thrust::detail::make_normal_iterator(left_ptr),thrust::constant_iterator<T>(value),
				thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,lhs->valid,output->valid,
				lhs->null_count,lhs->null_count,output->null_count,stream
		);
	}
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	return GDF_SUCCESS;
}

gdf_error gdf_comparison_static_i8(gdf_column *lhs, int8_t value, gdf_column *output,gdf_comparison_operator operation){
	return gdf_comparison_static_templated(lhs, value, output,operation);
}

gdf_error gdf_comparison_static_i16(gdf_column *lhs, int16_t value, gdf_column *output,gdf_comparison_operator operation){
	return gdf_comparison_static_templated(lhs, value, output,operation);
}

gdf_error gdf_comparison_static_i32(gdf_column *lhs, int32_t value, gdf_column *output,gdf_comparison_operator operation){
	return gdf_comparison_static_templated(lhs, value, output,operation);
}

gdf_error gdf_comparison_static_i64(gdf_column *lhs, int64_t value, gdf_column *output,gdf_comparison_operator operation){
	return gdf_comparison_static_templated(lhs, value, output,operation);
}

gdf_error gdf_comparison_static_f32(gdf_column *lhs, float value, gdf_column *output,gdf_comparison_operator operation){
	return gdf_comparison_static_templated(lhs, value, output,operation);
}

gdf_error gdf_comparison_static_f64(gdf_column *lhs, double value, gdf_column *output,gdf_comparison_operator operation){
	return gdf_comparison_static_templated(lhs, value, output,operation);
}




gdf_error gdf_comparison(gdf_column *lhs, gdf_column *rhs, gdf_column *output,gdf_comparison_operator operation){
	GDF_REQUIRE(lhs->size == rhs->size, GDF_COLUMN_SIZE_MISMATCH);
	GDF_REQUIRE(lhs->size == output->size, GDF_COLUMN_SIZE_MISMATCH);

	GDF_REQUIRE(output->dtype == GDF_INT8, GDF_COLUMN_SIZE_MISMATCH);
	
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
		 	

			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			//... 

		}else if(rhs->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> right_ptr((int16_t *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);

		}else if(rhs->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> right_ptr((int32_t *) rhs->data);
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
		}else if(rhs->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> right_ptr((int64_t *) rhs->data);
		
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
		}else if(rhs->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> right_ptr((float *) rhs->data);
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
		}else if(rhs->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> right_ptr((double *) rhs->data);
		
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
		}
	}else if(lhs->dtype == GDF_INT16){
		thrust::device_ptr<int16_t> left_ptr((int16_t *) lhs->data);
		if(rhs->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> right_ptr((int8_t *) rhs->data);
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
		}else if(rhs->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> right_ptr((int16_t *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);

		}else if(rhs->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> right_ptr((int32_t *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> right_ptr((int64_t *) rhs->data);

			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> right_ptr((float *) rhs->data);
		
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> right_ptr((double *) rhs->data);
		 
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			 
		}
	}else if(lhs->dtype == GDF_INT32 || lhs->dtype == GDF_STRING_CATEGORY){
		thrust::device_ptr<int32_t> left_ptr((int32_t *) lhs->data);
		if(rhs->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> right_ptr((int8_t *) rhs->data);
			 
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> right_ptr((int16_t *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);

		}else if(rhs->dtype == GDF_INT32 || rhs->dtype == GDF_STRING_CATEGORY){
			thrust::device_ptr<int32_t> right_ptr((int32_t *) rhs->data);
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> right_ptr((int64_t *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> right_ptr((float *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> right_ptr((double *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}
	}else if(lhs->dtype == GDF_INT64){
		thrust::device_ptr<int64_t> left_ptr((int64_t *) lhs->data);
		if(rhs->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> right_ptr((int8_t *) rhs->data);
		
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> right_ptr((int16_t *) rhs->data);

			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> right_ptr((int32_t *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> right_ptr((int64_t *) rhs->data);
		
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> right_ptr((float *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> right_ptr((double *) rhs->data);
		
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);

		}
	}else if(lhs->dtype == GDF_FLOAT32){
		thrust::device_ptr<float> left_ptr((float *) lhs->data);
		if(rhs->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> right_ptr((int8_t *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);

		}else if(rhs->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> right_ptr((int16_t *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);

		}else if(rhs->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> right_ptr((int32_t *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> right_ptr((int64_t *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
		
		}else if(rhs->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> right_ptr((float *) rhs->data);
		
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> right_ptr((double *) rhs->data);
		
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}
	}else if(lhs->dtype == GDF_FLOAT64){
		thrust::device_ptr<double> left_ptr((double *) lhs->data);
		if(rhs->dtype == GDF_INT8){
			thrust::device_ptr<int8_t> right_ptr((int8_t *) rhs->data);
		
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_INT16){
			thrust::device_ptr<int16_t> right_ptr((int16_t *) rhs->data);
		
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_INT32){
			thrust::device_ptr<int32_t> right_ptr((int32_t *) rhs->data);
		
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_INT64){
			thrust::device_ptr<int64_t> right_ptr((int64_t *) rhs->data);
			
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_FLOAT32){
			thrust::device_ptr<float> right_ptr((float *) rhs->data);
		
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);
			
		}else if(rhs->dtype == GDF_FLOAT64){
			thrust::device_ptr<double> right_ptr((double *) rhs->data);
		
			thrust::device_ptr<int8_t> out_ptr((int8_t *) output->data);
			gpu_filter_op(
					thrust::detail::make_normal_iterator(left_ptr),thrust::detail::make_normal_iterator(right_ptr),
					thrust::detail::make_normal_iterator(out_ptr),operation,lhs->size,lhs->valid,rhs->valid,output->valid,
					lhs->null_count,rhs->null_count,output->null_count,stream
			);			
		}
	} 



	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);



	return GDF_SUCCESS;
}

