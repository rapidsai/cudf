
/*
 * Copyright 2018 BlazingDB, Inc.
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


#pragma once

#ifndef _BIT_UTIL_H_
#define _BIT_UTIL_H_

#include "rmm/rmm.h"
#include "utilities/error_utils.h"
#include <cuda_runtime_api.h>


namespace gdf {
namespace util {

static constexpr int RECORDS_PER_BITMAP = 32;
using ValidType = gdf_valid_type;

/**
 * determine the bitmap that contains a record
 * @param[in]  record_idx    The record index
 * @return the bitmap index
 */
__device__  int which_bitmap_record(int record_idx) { return (record_idx / RECORDS_PER_BITMAP);  }


/**
 * determine which bit in a bitmap for a record
 *
 * @param[in]  record_idx    The record index
 * @return which bit within the bitmap
 */
__device__ int which_bit(int record_idx) { return (record_idx % RECORDS_PER_BITMAP);  }


/**
 * Set the value of a bit to either 0 or 1
 *
 * @param[in] valid         pointer to the device memory
 * @param[in] record_idx    the record index
 * @param[in] value         the value, must be either 0 or 1
 */
__device__ gdf_error set_bit_value(gdf_valid_type * valid, int record_idx, unsigned int value) {

	if ( value > 1)
		return GDF_INVALID_API_CALL;

	int rec = which_bitmap_record(record_idx);
	int bit = which_bit(record_idx);

	if (value == 0)
		atomicAnd( &valid[rec], (value << bit));
	else
		atomicOr( &valid[rec], (value << bit));

	if ( cudaPeekAtLastError() == cudaSuccess)
		return GDF_SUCCESS;
	else
		return GDF_CUDA_ERROR;
}


/**
 * Check to see if a record is not NULL (aka valid)
 *
 * @param[in] valid        the device memory containing the valid bitmaps
 * @param[in] record_idx   the record index to check
 *
 * @return  true/false on if the record is valid
 */
__device__ bool is_valid(gdf_valid_type * valid, int record_idx) {

	int rec = which_bitmap_record(record_idx);
	int bit = which_bit(record_idx);

	int status = atomicAnd(&valid[rec], (1 << bit));

	return (status == 0) ? false : true;
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
	return set_bit_value(valid, record_idx, 1U);
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
	return set_bit_value(valid, record_idx, 0U);
}


__device__ gdf_error slice_mask(gdf_valid_type * input_mask, int start, int stop, gdf_valid_type * output_mask) {
	return GDF_UNSUPPORTED_METHOD;
}


__device__ gdf_error bool_as_mask(gdf_column * bool_array, gdf_valid_type * bitmask ) {
	return GDF_UNSUPPORTED_METHOD;

}


__device__ gdf_error mask_as_bool(gdf_column* output, gdf_valid_type *mask) {
	return GDF_UNSUPPORTED_METHOD;

}


__device__ gdf_error mask_from_float_array(gdf_valid_type *output_mask, float *array) {
	return GDF_UNSUPPORTED_METHOD;

}



/**
 * Allocate device space for the valid bitmap.
 *
 * @param[out] gdf_valid_type *      pointer to where device memory will be allocated and returned
 * @param[in]  number_of_records     number of records
 * @param[in]  fill_value            optional, should the memory be initialized to all 0 or 1s. All other values indicate un-initialized
 */
gdf_error create_bitmap(gdf_valid_type *valid, int number_of_records, int fill_value = -1) {

	int num_bits_rec = (number_of_records + (RECORDS_PER_BITMAP - 1)) / RECORDS_PER_BITMAP;

	RMM_TRY( RMM_ALLOC((void**)&valid, 	sizeof(ValidType) * num_bits_rec, 0) );

	if (fill_value == 0) {      CUDA_TRY( cudaMemset(valid,	0,          sizeof(ValidType) * num_bits_rec));  }
	else if (fill_value == 1) { CUDA_TRY( cudaMemset(valid,	0xFFFFFFFF, sizeof(ValidType) * num_bits_rec));  }

	return GDF_SUCCESS;
}














// Instead of this function, use gdf_get_num_chars_bitmask from gdf/utils.h
//__host__ __device__ __forceinline__
//  size_t
//  valid_size(size_t column_length)
//{
//  const size_t n_ints = (column_length / ValidSize) + ((column_length % ValidSize) ? 1 : 0);
//  return n_ints * sizeof(ValidType);
//}

// Instead of this function, use gdf_is_valid from gdf/utils.h
///__host__ __device__ __forceinline__ bool get_bit(const gdf_valid_type* const bits, size_t i)
///{
///  return  bits == nullptr? true :  bits[i >> size_t(3)] & (1 << (i & size_t(7)));
///}

//__host__ __device__ __forceinline__
//  uint8_t
//  byte_bitmask(size_t i)
//{
//  static uint8_t kBitmask[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
//  return kBitmask[i];
//}
//
//__host__ __device__ __forceinline__
//  uint8_t
//  flipped_bitmask(size_t i)
//{
//  static uint8_t kFlippedBitmask[] = { 254, 253, 251, 247, 239, 223, 191, 127 };
//  return kFlippedBitmask[i];
//}
//
//__host__ __device__ __forceinline__ void turn_bit_on(uint8_t* const bits, size_t i)
//{
//  bits[i / 8] |= byte_bitmask(i % 8);
//}
//
//__host__ __device__ __forceinline__ void turn_bit_off(uint8_t* const bits, size_t i)
//{
//  bits[i / 8] &= flipped_bitmask(i % 8);
//}
//
__host__ __device__ __forceinline__ size_t last_byte_index(size_t column_size)
{
  return (column_size + 8 - 1) / 8;
}

static inline std::string chartobin(gdf_valid_type c, size_t size = 8)
{
  std::string bin;
  bin.resize(size);
  bin[0] = 0;
  size_t i;
  for (i = 0; i < size; i++) {
    bin[i] = (c % 2) + '0';
    c /= 2;
  }
  return bin;
}

static inline std::string gdf_valid_to_str(gdf_valid_type* valid, size_t column_size)
{
  size_t last_byte = gdf::util::last_byte_index(column_size);
  std::string response;
  for (size_t i = 0; i < last_byte; i++) {
    size_t n_bits = last_byte != i + 1 ? 8 : column_size - 8 * (last_byte - 1);
    auto result = chartobin(valid[i], n_bits);
    response += std::string(result);
  }
  return response;
}

} // namespace util
} // namespace gdf


#endif
