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

#ifndef _BIT_CONTAINER_H_
#define _BIT_CONTAINER_H_

#pragma once

#include <string.h>

#include "cudf.h"
#include "bit_mask.h"
#include "rmm/rmm.h"
#include "utilities/error_utils.h"
#include <cuda_runtime_api.h>

using bit_mask_t = bit_mask::bit_mask_t;


/* ---------------------------------------------------------------------------- */
/**
 * @Synopsis  Class for managing bit containers on the device
 */
/* ---------------------------------------------------------------------------- */
class BitMask {
public:
   __host__ __device__ BitMask(bit_mask_t *valid, int bitlength): valid_(valid), bitlength_(bitlength) {}
   __host__ __device__ BitMask(const BitMask &other): valid_(other.valid_), bitlength_(other.bitlength_) {}

   __host__ __device__ BitMask &operator=(const BitMask &other) {
     valid_ = other.valid_;
     bitlength_ = other.bitlength_;
     return *this;
   }

  /**
   * Check to see if a record is Valid (aka not null)
   *
   * @param[in] record_idx   the record index to check
   *
   * @return  true if record is valid, false if record is null
   */
  template <typename T>
  __device__ bool IsValid(T record_idx) const {
    return bit_mask::is_valid(valid_, record_idx);
  }

  /**
   * Set a bit (not thread-save)
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void SetBitUnsafe(T record_idx) {
    bit_mask::set_bit_unsafe(valid_, record_idx);
  }


  /**
   * Clear a bit (not thread-safe)
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void ClearBitUnsafe(T record_idx) {
    bit_mask::clear_bit_unsafe(record_idx);
  }

  /**
   * Set a bit in a thread-safe manner
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void SetBit(T record_idx) {
    int rec = bit_mask::details::which_word(record_idx);
    int bit = bit_mask::details::which_bit(record_idx);

    atomicOr( &valid_[rec], (1U << bit));
  }


  /**
   * Clear a bit in a thread-safe manner
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void ClearBit(T record_idx) {
    int rec = bit_mask::details::which_word(record_idx);
    int bit = bit_mask::details::which_bit(record_idx);

    atomicAnd( &valid_[rec], ~(1U << bit));
  }

  /**
   * Get the number of words in this bit container
   *
   * @return the number of words
   */
  __device__ int NumWords() const {
    return bit_mask::num_words(bitlength_);
  }

  /**
   * Getter for valid
   *
   * @return pointer to valid bit container
   */
  __host__ __device__ bit_mask_t *GetValid() const {
    return valid_;
  }

  /**
   * Get length
   *
   * @return length of bit container in bits
   */
  __host__ __device__ int Length() const {
    return bitlength_;
  }

private:
  bit_mask_t      *valid_;
  int              bitlength_;
};

#endif
