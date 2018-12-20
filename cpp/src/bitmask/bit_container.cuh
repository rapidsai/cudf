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

using bit_container_t = bit_container::bit_container_t;


/* ---------------------------------------------------------------------------- */
/**
 * @Synopsis  Class for managing bit containers on the device
 */
/* ---------------------------------------------------------------------------- */
class BitContainer {
public:
   __host__ __device__ BitContainer(bit_container_t *valid, int bitlength): valid_(valid), bitlength_(bitlength) {}
   __host__ __device__ BitContainer(const BitContainer &other): valid_(other.valid_), bitlength_(other.bitlength_) {}

   __host__ __device__ BitContainer &operator=(const BitContainer &other) {
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
    return bit_container::is_valid(valid_, record_idx);
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
    int rec = bit_container::which_word(record_idx);
    int bit = bit_container::which_bit(record_idx);

    atomicAnd( &valid_[rec], ~(1U << bit));
  }

  /**
   * Get the number of words in this bit container
   *
   * @return the number of words
   */
  __device__ int NumWords() const {
    return bit_container::num_words(bitlength_);
  }

  /**
   * Getter for valid
   *
   * @return pointer to valid bit container
   */
  __host__ __device__ bit_container_t *GetValid() const {
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


  /**
   * Allocate device space for the valid bitmap.
   *
   * @param[in]  number_of_records     number of records
   * @param[in]  fill_value            optional, should the memory be initialized to all 0 or 1s. All other values indicate un-initialized
   * @return pointer to allocated space on the device
   */
  template <typename T>
  __host__
  static bit_container_t * CreateBitContainer(T number_of_records, int fill_value = -1) {
    bit_container_t *container_d = nullptr;
    int num_bit_containers = bit_container::num_words(number_of_records);

    RMM_ALLOC((void**)&container_d, sizeof(bit_container_t) * num_bit_containers, 0);
    if (container_d == nullptr)
      return container_d;

    if (fill_value == 0) {
      cudaMemset(container_d, 0, sizeof(bit_container_t) * num_bit_containers);
    } else if (fill_value == 1) {
      cudaMemset(container_d, 0xff, sizeof(bit_container_t) * num_bit_containers);

      //
      //  Need to worry about the case where we set bits past the end
      //
      int bits_used_in_last_word = number_of_records - (num_bit_containers - 1) * bit_container::BITS_PER_WORD;
      bit_container_t temp = 1U << bits_used_in_last_word;
      temp--;

      PutWord(temp, &container_d[num_bit_containers - 1]);
    }

    return container_d;
  }

  /**
   *  Copy data between host and device
   *
   *  @param[out] dst      - the address of the destination
   *  @param[in] src       - the address of the source
   *  @param[in] num_bits  - the number of bits in the bit container
   *  @param[in] kind      - the direction of the copy
   */
  inline __host__
  static void CopyBitContainer(bit_container_t *dst, const bit_container_t *src, size_t num_bits, enum cudaMemcpyKind kind) {
    cudaMemcpy(dst, src, bit_container::num_words(num_bits) * sizeof(bit_container_t), kind);
  }

  /**
   * Deallocate device space for the valid bitmap
   *
   * @param[in]  valid   The pointer to device space that we wish to deallocate
   */
  inline __host__
  static void DestroyBitContainer(bit_container_t *valid) {
    //RMM_FREE((void **) &valid, 0);
    cudaFree((void **) &valid);
  }

  /**
   * Get a single word of bits from the device
   *
   * @param[out]  word - address on host where the bits will be stored
   * @param[out]  device_word - address on the device containing the bits to fetch
   */
  inline __host__
  static void GetWord(bit_container_t *word, const bit_container_t *device_word) {
    cudaMemcpy(word, device_word, sizeof(bit_container_t), cudaMemcpyDeviceToHost);
  }
	  
  /**
   * Put a single word of bits to the device
   *
   * @param[out]  word - address on host containing bits to store
   * @param[out]  device_word - address on the device where the bits will be stored
   */
  inline __host__
  static void PutWord(bit_container_t word, bit_container_t *device_word) {
    cudaMemcpy(device_word, &word, sizeof(bit_container_t), cudaMemcpyHostToDevice);
  }
      
private:
  bit_container_t *valid_;
  int              bitlength_;
};

#endif
