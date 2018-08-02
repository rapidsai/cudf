#ifndef GDF_BITMASK_OPS_H
#define GDF_BITMASK_OPS_H

#include <cuda_runtime.h>


gdf_error all_bitmask_on(gdf_valid_type * valid_out, gdf_size_type & out_null_count, gdf_size_type num_values, cudaStream_t stream);



gdf_error apply_bitmask_to_bitmask(gdf_size_type & out_null_count, gdf_valid_type * valid_out, gdf_valid_type * valid_left, gdf_valid_type * valid_right,
		cudaStream_t stream, gdf_size_type num_values);
#endif
