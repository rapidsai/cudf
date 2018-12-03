
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
#pragma once

namespace gdf {
namespace util {

static constexpr int ValidSize = 32;
using ValidType = uint32_t;




/**
 * Allocate device space for the valid bitmap
 *
 * @param[out] gdf_valid_type *      pointer to where device memory will be allocated and returned
 * @param[in]  number_of_records     number of records
 * @param[in]  fill_value            optional, should the memory be initialized to all 0 or 1s. All other values indicate un-initialized
 */
gdf_error create_bitmap(gdf_valid_type *valid, int number_of_records, int fill_value = -1) {

	int num_bits_rec = (number_of_records + (ValidSize - 1)) / ValidSize;

	RMM_TRY( RMM_ALLOC((void**)&valid, 	sizeof(ValidType) * num_bits_rec, 0) );

	if (fill_value == 0) {      CUDA_TRY( cudaMemset(valid,	0,           sizeof(ValidType) * num_bits_rec));  }
	else if (fill_value == 1) { CUDA_TRY( cudaMemset(valid,	0xFFFFFFFF, sizeof(ValidType) * num_bits_rec));  }

	return GDF_SUCCESS;
}


/**
 * Check to see if a record is not NULL (aka valid)
 *
 * @param[out]
 * @param[in]
 *
 */
__device__ bool is_valid(gdf_valid_type * valid_masks, int record_idx) {

	int rec = whichBitmapRecord(record_idx);
	int bit = whichBit(record_idx);

	int status = atomicAnd(valid_masks[rec], (1 << bit));

	return (status == 0) ? false : true;
}


__device__ gdf_error set_bit(int record_idx) {
	return setBitValue(record_idx, 1);
}

__device__ gdf_error clear_bit(int record_idx) {
	return setBitValue(record_idx, 0);
}


__device__ gdf_error setBitValue(int record_idx, int value) {

	if ( value < 0 || value > 1)
		return GDF_INVALID_API_CALL;

	int rec = whichBitmapRecord(record_idx);
	int bit = whichBit(record_idx);

	int status = atomicOr(valid_masks[rec], (value << bit));

	if ( status == valid_masks[rec])
		return GDF_SUCCESS;
	else
		GDF_CUDA_ERROR;
}




__device__ int whichBitmapRecord(int record_idx) { return (record_idx/ValidSize);  }
__device__ int whichBit(int record_idx) { return (bit % ValidSize);  }










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
//__host__ __device__ __forceinline__ size_t last_byte_index(size_t column_size)
//{
//  return (column_size + 8 - 1) / 8;
//}

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
