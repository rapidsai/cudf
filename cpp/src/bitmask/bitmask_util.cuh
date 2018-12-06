/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef _BITMASK_UTIL_H_
#define _BITMASK_UTIL_H_

#pragma once



#include "cudf.h"
#include "rmm/rmm.h"
#include "utilities/error_utils.h"
#include <cuda_runtime_api.h>


namespace bitmask {

//----------------------------------------------------------------------------------------------------------
//			Host and Device Utility Functions
//----------------------------------------------------------------------------------------------------------
namespace util {

	/**
	 * determine the bitmap that contains a record
	 * @param[in]  record_idx    The record index
	 * @return the bitmap index
	 */
	__host__ __device__  int which_bitmap_record(int record_idx) {
		return (record_idx / GDF_VALID_BITSIZE);
	}

	/**
	 * determine which bit in a bitmap relates to a record
	 * @param[in]  record_idx    The record index
	 * @return which bit within the bitmap
	 */
	__host__ __device__ int which_bit(int record_idx) {
		return (record_idx % GDF_VALID_BITSIZE);
	}

}	// end util namespace



//----------------------------------------------------------------------------------------------------------
//			Device Only Utility Functions
//----------------------------------------------------------------------------------------------------------
namespace device {

	/**
	 * Check to see if a record is Valid (aka not null)
	 *
	 * @param[in] valid        the device memory containing the valid bitmaps
	 * @param[in] record_idx   the record index to check
	 *
	 * @return  true if record is valid, false if record is null
	 */
	__device__ bool is_valid(gdf_valid_type * valid, int record_idx) {
		int rec = bitmask::util::which_bitmap_record(record_idx);
		int bit = bitmask::util::which_bit(record_idx);

		int status = atomicAnd(&valid[rec], (1U << bit));

		return ( status == 0) ? false : true;

	}


	/**
	 * Set a bit
	 *
	 * @param[in] valid         the valid memory array
	 * @param[in] record_idx    the record index
	 *
	 * @return gdf_error - did it work or not
	 */
	__device__ gdf_error set_bit(gdf_valid_type * valid, int record_idx) {
		int rec = bitmask::util::which_bitmap_record(record_idx);
		int bit = bitmask::util::which_bit(record_idx);

		atomicOr( &valid[rec],   (1U << bit));		// set the bit

		return ( cudaPeekAtLastError() == cudaSuccess) ? GDF_SUCCESS : GDF_CUDA_ERROR;
	}


	/**
	 * Clear a bit
	 *
	 * @param[in] valid         the valid memory array
	 * @param[in] record_idx    the record index
	 *
	 * @return gdf_error - did it work or not
	 */
	__device__ gdf_error clear_bit(gdf_valid_type * valid, int record_idx) {
		int rec = bitmask::util::which_bitmap_record(record_idx);
		int bit = bitmask::util::which_bit(record_idx);

		atomicAnd( &valid[rec], ~(1U << bit));		// clear the bit

		return ( cudaPeekAtLastError() == cudaSuccess) ? GDF_SUCCESS : GDF_CUDA_ERROR;
	}

} // end of device namespace


//----------------------------------------------------------------------------------------------------------
//		Utility Functions that launch kernels
//----------------------------------------------------------------------------------------------------------
namespace kernel {





}




//----------------------------------------------------------------------------------------------------------
//			Host Utility Functions
//----------------------------------------------------------------------------------------------------------
namespace host {
	/**
	 * Allocate device space for the valid bitmap.
	 *
	 * @param[out] gdf_valid_type *      pointer to where device memory will be allocated and returned
	 * @param[in]  number_of_records     number of records
	 * @param[in]  fill_value            optional, should the memory be initialized to all 0 or 1s. All other values indicate un-initialized
	 * @return error status
	 */
	gdf_valid_type * create_bitmap(int number_of_records, int fill_value = -1) {

		gdf_valid_type *valid_d;

		int num_bitmasks = (number_of_records + (GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE;

		RMM_ALLOC((void**)&valid_d, 	sizeof(gdf_valid_type) * num_bitmasks, 0);

		if (valid_d == NULL)
			return valid_d;

		if (fill_value == 0) {      cudaMemset(valid_d,	0,          sizeof(gdf_valid_type) * num_bitmasks);  }
		else if (fill_value == 1) { cudaMemset(valid_d,	0xFFFFFFFF, sizeof(gdf_valid_type) * num_bitmasks);  }

		return valid_d;
	}


	/**
	 * Check to see if a record is Valid (aka not null)
	 *
	 * @param[in] valid        the device memory containing the valid bitmaps
	 * @param[in] record_idx   the record index to check
	 *
	 * @return  true if record is valid, false if record is null
	 */
	bool is_valid(gdf_valid_type * valid, int record_idx) {

		int h_bitm;

		int rec = bitmask::util::which_bitmap_record(record_idx);
		int bit = bitmask::util::which_bit(record_idx);

		CUDA_TRY( cudaMemcpy(&h_bitm, &valid[rec], sizeof(h_bitm), cudaMemcpyDeviceToHost) );

		int status = valid[rec] & (1U << bit);

		return ( status == 0) ? false : true;

	}


	int num_of_bitmask(int number_of_records) {
		int num_bitmasks = (number_of_records + (GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE;

		return num_bitmasks;
	}


	gdf_size_type num_bytes(int number_of_records) {
		int num_bitmasks = (number_of_records + (GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE;

		return sizeof(gdf_valid_type) * num_bitmasks;
	}

}  // end of host namespace



//----------------------------------------------------------------------------------------------------------
//			Device Mask Utility Functions
//----------------------------------------------------------------------------------------------------------
namespace test {

	bool is_valid(gdf_valid_type * valid, int record_idx) {

		int rec = bitmask::util::which_bitmap_record(record_idx);
		int bit = bitmask::util::which_bit(record_idx);

		int status = valid[rec] & (1U << bit);

		return ( status == 0) ? false : true;
	}


	gdf_error set_bit(gdf_valid_type * valid, int record_idx) {

		int rec = bitmask::util::which_bitmap_record(record_idx);
		int bit = bitmask::util::which_bit(record_idx);

		valid[rec] = valid[rec] |  (1U << bit);		// set the bit

		return GDF_SUCCESS;
	}


	/**
	 * Clear a bit
	 *
	 * @param[in] valid         the valid memory array
	 * @param[in] record_idx    the record index
	 *
	 * @return gdf_error - did it work or not
	 */
	gdf_error clear_bit(gdf_valid_type * valid, int record_idx) {

		int rec = bitmask::util::which_bitmap_record(record_idx);
		int bit = bitmask::util::which_bit(record_idx);

		valid[rec] = valid[rec] & ~(1U << bit);		// clear the bit

		return GDF_SUCCESS;
	}


}  // end of test namespace















namespace tdb {
//----------------------------------------------------------------------------------------------------------
//			Device Mask Utility Functions
//----------------------------------------------------------------------------------------------------------

/**
 * Given an input mask, slice it based on a start and end bit
 *
 * @param[in]
 * @param[in]
 * @param[in]
 * @param[out]
 *
 * @return
 */
__device__ gdf_error slice_mask(gdf_valid_type * input_mask, int start, int stop, gdf_valid_type * output_mask) {
	return GDF_UNSUPPORTED_METHOD;
}





/**
 * Given an input mask, slice it based on a start and end bit
 *
 * @param[in]
 * @param[in]
 * @param[in]
 * @param[out]
 *
 * @return
 */
__device__ gdf_error bool_as_mask(gdf_column * bool_array, gdf_valid_type * bitmask ) {
	return GDF_UNSUPPORTED_METHOD;

}


__device__ gdf_error mask_as_bool(gdf_column* output, gdf_valid_type *mask) {
	return GDF_UNSUPPORTED_METHOD;

}


__device__ gdf_error mask_from_float_array(gdf_valid_type *output_mask, float *array) {
	return GDF_UNSUPPORTED_METHOD;

}
}


} // namespace bitutil



#endif
