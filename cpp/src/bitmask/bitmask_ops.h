#ifndef GDF_BITMASK_OPS_H
#define GDF_BITMASK_OPS_H

#include <cuda_runtime.h>

/**---------------------------------------------------------------------------*
 * @brief Sets all bits in input valid mask to 1
 * 
 * @param valid_out preallocated output valid mask
 * @param out_null_count number of nulls (0 bits) in valid mask. Always set to 0
 * @param num_values number of values in column associated with output mask
 * @param stream cuda stream to run in
 * @return gdf_error 
 *---------------------------------------------------------------------------**/
gdf_error all_bitmask_on(gdf_valid_type * valid_out, gdf_size_type & out_null_count, gdf_size_type num_values, cudaStream_t stream);

/**---------------------------------------------------------------------------*
 * @brief Computes bitwise AND on two valid masks and sets it in output
 * 
 * @param out_null_count number of nulls (0 bits) in output valid mask
 * @param valid_out preallocated mask to set the result values in
 * @param valid_left input valid mask 1
 * @param valid_right input valid mask 2
 * @param stream cuda stream to run in
 * @param num_values number of values in each input mask valid_left and valid_right
 * @return gdf_error 
 *---------------------------------------------------------------------------**/
gdf_error apply_bitmask_to_bitmask(gdf_size_type & out_null_count, gdf_valid_type * valid_out, gdf_valid_type * valid_left, gdf_valid_type * valid_right,
		cudaStream_t stream, gdf_size_type num_values);
#endif
