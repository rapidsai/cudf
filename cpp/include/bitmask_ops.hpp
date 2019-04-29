#ifndef GDF_BITMASK_OPS_H
#define GDF_BITMASK_OPS_H

#include <cuda_runtime.h>
#include <rmm/thrust_rmm_allocator.h>
#include <types.hpp>

/**---------------------------------------------------------------------------*
 * @brief Sets all bits in input valid mask to 1
 *
 * @param valid_out preallocated output valid mask
 * @param out_null_count number of nulls (0 bits) in valid mask. Always set to 0
 * @param num_values number of values in column associated with output mask
 * @param stream cuda stream to run in
 * @return gdf_error
 *---------------------------------------------------------------------------**/
gdf_error all_bitmask_on(gdf_valid_type* valid_out,
                         gdf_size_type& out_null_count,
                         gdf_size_type num_values, cudaStream_t stream);

/**---------------------------------------------------------------------------*
 * @brief Computes bitwise AND on two valid masks and sets it in output
 *
 * @param out_null_count number of nulls (0 bits) in output valid mask
 * @param valid_out preallocated mask to set the result values in
 * @param valid_left input valid mask 1
 * @param valid_right input valid mask 2
 * @param stream cuda stream to run in
 * @param num_values number of values in each input mask valid_left and
 * valid_right
 * @return gdf_error
 *---------------------------------------------------------------------------**/
gdf_error apply_bitmask_to_bitmask(gdf_size_type& out_null_count,
                                   gdf_valid_type* valid_out,
                                   gdf_valid_type* valid_left,
                                   gdf_valid_type* valid_right,
                                   cudaStream_t stream,
                                   gdf_size_type num_values);

/**
 * @brief  Counts the number of valid bits for the specified number of rows
 * in a validity bitmask.
 *
 * If the bitmask is null, returns a count equal to the number of rows.
 *
 * @param[in] masks The validity bitmask buffer in device memory
 * @param[in] num_rows The number of bits to count
 * @param[out] count The number of valid bits in the buffer from [0, num_rows)
 *
 * @returns  GDF_SUCCESS upon successful completion
 *
 */
gdf_error gdf_count_nonzero_mask(gdf_valid_type const* masks,
                                 gdf_size_type num_rows, gdf_size_type* count);

/** ---------------------------------------------------------------------------*
 * @brief Concatenate the validity bitmasks of multiple columns
 *
 * Accounts for the differences between lengths of columns and their bitmasks
 * (e.g. because gdf_valid_type is larger than one bit).
 *
 * @param[out] output_mask The concatenated mask
 * @param[in] output_column_length The total length (in data elements) of the
 *                                 concatenated column
 * @param[in] masks_to_concat The array of device pointers to validity bitmasks
 *                            for the columns to concatenate
 * @param[in] column_lengths An array of lengths of the columns to concatenate
 * @param[in] num_columns The number of columns to concatenate
 * @return gdf_error GDF_SUCCESS or GDF_CUDA_ERROR if there is a runtime CUDA
           error
 *
 ---------------------------------------------------------------------------**/
gdf_error gdf_mask_concat(gdf_valid_type* output_mask,
                          gdf_size_type output_column_length,
                          gdf_valid_type* masks_to_concat[],
                          gdf_size_type* column_lengths,
                          gdf_size_type num_columns);

namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief Computes a bitmask indicating the presence of NULL values in rows of a
 * table.
 *
 * If a row `i` in `table` contains one or more NULL values, then bit `i` in the
 * returned bitmask will be 0.
 *
 * Otherwise, bit `i` will be 1.
 *
 * @param table The table to compute the row bitmask of.
 * @return bit_mask::bit_mask_t* The bitmask indicating the presence of NULLs in
 * a row
 *---------------------------------------------------------------------------**/
rmm::device_vector<bit_mask::bit_mask_t> row_bitmask(cudf::table const& table,
                                                     cudaStream_t stream = 0);
}  // namespace cudf

#endif
