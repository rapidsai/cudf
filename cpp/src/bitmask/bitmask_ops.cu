#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"
#include "cudf/functions.h"
#include "rmm/thrust_rmm_allocator.h"
#include "bitmask/legacy_bitmask.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/device_vector.h>



gdf_error all_bitmask_on(gdf_valid_type * valid_out, gdf_size_type & out_null_count, gdf_size_type num_values, cudaStream_t stream){
	gdf_size_type num_bitmask_elements = gdf_num_bitmask_elements( num_values );

	gdf_valid_type max_char = 255;
	thrust::fill(rmm::exec_policy(stream)->on(stream),
				 valid_out,
				 valid_out + num_bitmask_elements,
				 max_char);
	//we have no nulls so set all the bits in gdf_valid_type to 1
	out_null_count = 0;
	return GDF_SUCCESS;
}

gdf_error apply_bitmask_to_bitmask(gdf_size_type & out_null_count, gdf_valid_type * valid_out, gdf_valid_type * valid_left, gdf_valid_type * valid_right,
		cudaStream_t stream, gdf_size_type num_values){

	gdf_size_type num_bitmask_elements = gdf_num_bitmask_elements( num_values );

	thrust::transform(rmm::exec_policy(stream)->on(stream),
					  valid_left,
					  valid_left + num_bitmask_elements,
					  valid_right,
					  valid_out,
					  thrust::bit_and<gdf_valid_type>());

	gdf_size_type non_nulls;
	auto error = gdf_count_nonzero_mask(valid_out, num_values, &non_nulls);
	out_null_count = num_values - non_nulls;
	return error;
}
