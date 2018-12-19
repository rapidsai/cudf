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

#ifndef _BITMASK_H_
#define _BITMASK_H_

#pragma once

#include "cudf.h"
#include "rmm/rmm.h"
#include "utilities/error_utils.h"
#include <cuda_runtime_api.h>


/* ---------------------------------------------------------------------------- */
/**
 * @Synopsis  Valid Bitmask 
 */
/* ---------------------------------------------------------------------------- */
class ValidBit {
	
public:
	typedef uint32_t 	valid_type;
	
	ValidBit(gdf_valid_type *valid, int bitlength): valid_(valid), bitlength_(bitlength) {}
	ValidBit(const ValidBit &util): valid_(util.valid_), bitlength_(util.bitlength_) {}
	ValidBit(int num_records) {
		int bitlength_ = num_elements(num_records);
		
		RMM_ALLOC((void**)&valid_, 	sizeof(gdf_valid_type) * bitlength_, 0);
	}
	

  /**
   * Check to see if a record is Valid (aka not null)
   * @param[in] record_idx   the record index to check
   * @return  true if record is valid, false if record is null
   */
  template <typename T>
  __device__ bool is_valid(T record_idx) const {
    int rec = which_word(record_idx);
    int bit = which_bit(record_idx);

	int status = atomicAnd(&valid_[rec], (gdf_size_type{1} << bit));

	return ( status == 0) ? false : true;
  }

  template <typename T>
  __device__ static bool is_valid(T record_idx, gdf_valid_type *v) const {
    int rec = which_word(record_idx);
    int bit = which_bit(record_idx);

	int status = atomicAnd(&v[rec], (gdf_size_type{1} << bit));

	return ( status == 0) ? false : true;
  }


  //----------------------------------------------------
  /**
   * Set a bit
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void set_bit(T record_idx) {
    int rec = util_.WhichWord(record_idx);
    int bit = util_.WhichBit(record_idx);

    atomicOr( &valid_()[rec], (gdf_valid_type{1} << bit));
  }

  template <typename T>
  __device__ void static set_bit(T record_idx, gdf_valid_type *v) {
    int rec = util_.WhichWord(record_idx);
    int bit = util_.WhichBit(record_idx);

    atomicOr( &v[rec], (gdf_valid_type{1} << bit));
  }

  //----------------------------------------------------
  /**
   * Clear a bit in a thread-safe manner
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void clear_bit(T record_idx) {
    int rec = util_.WhichWord(record_idx);
    int bit = util_.WhichBit(record_idx);

    atomicAnd( &valid_[rec], ~(gdf_valid_type{1} << bit));
  }

  template <typename T>
  __device__ void static clear_bit(T record_idx, gdf_valid_type *v ) {
    int rec = util_.WhichWord(record_idx);
    int bit = util_.WhichBit(record_idx);

    atomicAnd( &v[rec], ~(gdf_valid_type{1} << bit));
  }


  //-------------------------------------------------------
  /**
   *  Count the number of ones set in the bit mask
   *
   *  @return the number of ones set in the bit mask
   */
  __device__ int count_ones() const {
    int sum = 0;
    for (int i = 0 ; i < NumWords() ; ++i) {
      sum += __popc(valid_[i]);
    }

    return sum;
  }  
  

  __device__ static int count_ones(gdf_valid_type *v) const {
    int sum = 0;
    for (int i = 0 ; i < NumWords() ; ++i) {
      sum += __popc(v[i]);
    }

    return sum;
  }

  
  /**
   * Get length
   *
   * @return length of bitmask in bits
   */
  __host__ __device__ int length() const {
    return bitlength_;
  }

  /**
   * determine the bitmap that contains a record
   * @param[in]  record_idx    The record index
   * @return the bitmap index
   */
  template <typename T>
  __host__ __device__ int static which_word(T record_idx) const {
    return (record_idx / GDF_VALID_BITSIZE);
  }

  /**
   * determine which bit in a bitmap relates to a record
   * @param[in]  record_idx    The record index
   * @return which bit within the bitmap
   */
  template <typename T>
  __host__ __device__ int static which_bit(T record_idx) const {
    return (record_idx % GDF_VALID_BITSIZE);
  }

  /**
   * determine how many words need to be allocated
   * @param[in]  number_of_records    The number of bits in the mask
   * @return the number of words to allocate
   */
  template <typename T>
  __host__ __device__ static int num_elements(T number_of_records) {
    return (number_of_records + (GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE;
  }

  /**
   * Get the number of words in this bit mask
   * @return the number of words
   */
  inline
  __host__ __device__ int get_num_elements() const {
    return NumWords(bitlength_);
  }

private:
  /**
   *   array of entries containing the bitmask
   */
  gdf_valid_type *valid_;
  int             bitlength_;
};






#endif
