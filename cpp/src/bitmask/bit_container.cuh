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
#include "bit_container.h"
#include "rmm/rmm.h"
#include "utilities/error_utils.h"
#include <cuda_runtime_api.h>


/* ---------------------------------------------------------------------------- */
/**
 * @Synopsis  Base class for managing bit masks.
 */
/* ---------------------------------------------------------------------------- */
class Util {
public:
  __device__ __host__
  Util(bit_container_t *valid, int bitlength): valid_(valid), bitlength_(bitlength) {}

  __device__ __host__
  Util(const Util &util): valid_(util.valid_), bitlength_(util.bitlength_) {}

  /**
   * Check to see if a record is Valid (aka not null)
   *
   * @param[in] record_idx   the record index to check
   *
   * @return  true if record is valid, false if record is null
   */
  template <typename T>
  __host__ __device__
  bool IsValid(T record_idx) const {
    return is_valid(valid_, record_idx);
  }

  /**
   * Set a bit
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __host__ __device__
  void SetBit(T record_idx) {
    int rec = which_word(record_idx);
    int bit = which_bit(record_idx);

    valid_[rec] = valid_[rec] | (1U << bit);
  }

  /**
   * Clear a bit
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __host__ __device__
  void ClearBit(T record_idx) {
    int rec = which_word(record_idx);
    int bit = which_bit(record_idx);

    valid_[rec] = valid_[rec] & (~(1U << bit));
  }

  /**
   * Getter for valid
   *
   * @return pointer to valid bitmask
   */
  __host__ __device__
  bit_container_t *GetValid() const {
    return valid_;
  }

  /**
   * Get length
   *
   * @return length of bitmask in bits
   */
  __host__ __device__
  int Length() const {
    return bitlength_;
  }

  /**
   * Get the number of words in this bit mask
   *
   * @return the number of words
   */
  inline
  __host__ __device__
  int NumWords() const {
    return num_words(bitlength_);
  }

private:
  /**
   *   array of entries containing the bitmask
   */
};


/* ---------------------------------------------------------------------------- */
/**
 * @Synopsis  Class for managing bit masks on the device
 */
/* ---------------------------------------------------------------------------- */
class BitContainer {
public:
   __device__ BitContainer(bit_container_t *valid, int bitlength): valid_(valid), bitlength_(bitlength) {}
   __device__ BitContainer(const BitContainer &other): valid_(other.valid_), bitlength_(other.bitlength_) {}

  /**
   * Check to see if a record is Valid (aka not null)
   *
   * @param[in] record_idx   the record index to check
   *
   * @return  true if record is valid, false if record is null
   */
  template <typename T>
  __device__ bool IsValid(T record_idx) const {
    return bit_container::is_valid(record_idx);
  }

  /**
   * Set a bit (not thread-save)
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void SetBitUnsafe(T record_idx) {
    bit_container::set_bit_unsafe(valid_, record_idx);
  }


  /**
   * Clear a bit (not thread-safe)
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void ClearBitUnsafe(T record_idx) {
    bit_container::clear_bit_unsafe(record_idx);
  }

  /**
   * Set a bit in a thread-safe manner
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void SetBit(T record_idx) {
    int rec = bit_container::which_word(record_idx);
    int bit = bit_container::which_bit(record_idx);

    atomicOr( &valid_[rec], (1U << bit));
  }


  /**
   * Clear a bit in a thread-safe manner
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void ClearBit(T record_idx) {
    int rec = which_word(record_idx);
    int bit = which_bit(record_idx);

    atomicAnd( &valid_[rec], ~(1U << bit));
  }

  /**
   *  Count the number of ones set in the bit mask
   *
   *  @return the number of ones set in the bit mask
   */
  __device__ int CountOnes() const {
    int sum = 0;
    for (int i = 0 ; i < NumWords() ; ++i) {
      sum += __popc(GetValid()[i]);
    }

    return sum;
  }

  /**
   * Get the number of words in this bit mask
   *
   * @return the number of words
   */
  __device__ int NumWords() const {
    return bit_container::num_words(bitlength_);
  }

  /**
   * Getter for valid
   *
   * @return pointer to valid bitmask
   */
  __device__ bit_container_t *GetValid() const {
    return valid_;
  }

  /**
   * Get length
   *
   * @return length of bitmask in bits
   */
  int Length() const {
    return bitlength_;
  }

private:
  bit_container_t *valid_;
  int              bitlength_;
};

#endif
