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

#ifndef BIT_CONTAINER_H
#define BIT_CONTAINER_H

#include "utilities/cudf_utils.h"

namespace bit_mask {
  typedef uint32_t     bit_mask_t;
  constexpr gdf_size_type BITS_PER_WORD{sizeof(bit_mask_t) * 8};
  
  /**
   * determine the number of bit_mask_t words are used
   * @param[in]  size    Number of bits in the bitmask
   * @return the number of words
   */
  CUDA_HOST_DEVICE_CALLABLE
  constexpr gdf_size_type num_words(gdf_size_type size) { 
    return (( size + ( BITS_PER_WORD - 1)) / BITS_PER_WORD ); 
  }

  /**
   *  Copy data between host and device
   *
   *  @param[out] dst      - the address of the destination
   *  @param[in] src       - the address of the source
   *  @param[in] num_bits  - the number of bits in the bit container
   *  @param[in] kind      - the direction of the copy
   *  @return GDF_SUCCESS on success, the CUDA error on failure
   */
  inline gdf_error copy_bit_mask(bit_mask_t *dst, const bit_mask_t *src, size_t num_bits, enum cudaMemcpyKind kind) {
    CUDA_TRY(cudaMemcpy(dst, src, num_words(num_bits) * sizeof(bit_mask_t), kind));
    return GDF_SUCCESS;
  }

  /**
   * Deallocate device space for the valid bitmap
   *
   *  @param[in]  valid   The pointer to device space that we wish to deallocate
   *  @return GDF_SUCCESS on success, the CUDA error on failure
   */
  inline gdf_error destroy_bit_mask(bit_mask_t *valid) {
    RMM_TRY(RMM_FREE(valid, 0));
    return GDF_SUCCESS;
  }

  /**
   * Get a single word of bits from the device
   *
   *  @param[out]  word - address on host where the bits will be stored
   *  @param[out]  device_word - address on the device containing the bits to fetch
   *  @return GDF_SUCCESS on success, the CUDA error on failure
   */
  inline gdf_error get_word(bit_mask_t *word, const bit_mask_t *device_word) {
    CUDA_TRY(cudaMemcpy(word, device_word, sizeof(bit_mask_t), cudaMemcpyDeviceToHost));
    return GDF_SUCCESS;
  }
	  
  /**
   * Put a single word of bits to the device
   *
   * @param[out]  word - address on host containing bits to store
   * @param[out]  device_word - address on the device where the bits will be stored
   */
  inline gdf_error put_word(bit_mask_t word, bit_mask_t *device_word) {
    CUDA_TRY(cudaMemcpy(device_word, &word, sizeof(bit_mask_t), cudaMemcpyHostToDevice));
    return GDF_SUCCESS;
  }

  /**
   * Allocate device space for the valid bitmap.
   *
   * @param[out] mask                  address of the bit mask pointer
   * @param[in]  number_of_records     number of records
   * @param[in]  fill_value            optional, should the memory be initialized to all 0 or 1s. All other
   *                                   values indicate un-initialized.  Default is uninitialized
   * @param[in]  padding               optional, specifies byte boundary the data should be padded to.
   *                                   Defaults to 64 bytes, meaning the space allocated will be rounded
   *                                   up to the next multiple of 64 bytes.
   * @return GDF_SUCCESS on success, the RMM or CUDA error on error
   */
  gdf_error create_bit_mask(bit_mask_t **mask, gdf_size_type number_of_records, int fill_value = -1, gdf_size_type padding = 64) {
    //
    //  To handle padding, we will round the number_of_records up to the next padding boundary, then identify how many words
    //  that equates to.  Then we can allocate the appropriate amount of storage.
    //
    gdf_size_type num_padding_blocks = (number_of_records + 8 * padding - 1) / (8 * padding);
    gdf_size_type num_words = bit_mask::num_words(num_padding_blocks * 8 * padding);

    RMM_TRY(RMM_ALLOC(mask, sizeof(bit_mask_t) * num_words, 0));

    if (fill_value == 0) {
      //
      //  Padding with zero is an easy case
      //
      CUDA_TRY(cudaMemset(*mask, 0, sizeof(bit_mask_t) * num_words));
    } else if (fill_value == 1) {
      //
      //  Padding with one is a more complex case, as we want the first
      //  number_of_records bits set to 1, and all of the remaining
      //  bits set to 0.
      //
      //  We accomplish this by identifying the number of used words.
      //  We'll set all words from 0 to used_words - 1 to 1, all words
      //  from used_words to num_words - 1 to 0.  Finally, the last
      //  used word will be constructed to set to a left mask of ones
      //  and stored out on the device.
      //
      gdf_size_type used_words = bit_mask::num_words(number_of_records);
      CUDA_TRY(cudaMemset(*mask, 0xff, sizeof(bit_mask_t) * used_words));
      CUDA_TRY(cudaMemset(&(*mask)[used_words], 0, sizeof(bit_mask_t) * (num_words - used_words)));

      //
      //  Need to worry about the case where we set bits past the end
      //
      int bits_used_in_last_word = number_of_records - (used_words - 1) * BITS_PER_WORD;
      bit_mask_t temp = bit_mask_t{1} << bits_used_in_last_word;
      temp--;

      return put_word(temp, &(*mask)[used_words-1]);
    }

    return GDF_SUCCESS;
  }

  /**
   * determine which bit in a bitmap relates to a record
   * @param[in]  record_idx    The record index
   * @return which bit within the bitmap
   */
  template <typename T>
  CUDA_HOST_DEVICE_CALLABLE
  gdf_size_type is_valid(const bit_mask_t *valid, T record_idx) {
    static_assert(std::is_integral<T>::value, "Record index must be of an integ4al type");

    const gdf_size_type rec{which_word(record_idx)};
    const gdf_size_type bit{which_bit(record_idx)};

    return ((valid[rec] & (bit_mask_t{1} << bit)) != 0);
  }

  /**
   * determine which bit in a bitmap relates to a record
   * @param[in]  record_idx    The record index
   * @return which bit within the bitmap
   */
  template <typename T>
  CUDA_HOST_DEVICE_CALLABLE
  void set_bit_unsafe(bit_mask_t *valid, T record_idx) {
    static_assert(std::is_integral<T>::value, "Record index must be of an integ4al type");

    const gdf_size_type rec{which_word(record_idx)};
    const gdf_size_type bit{which_bit(record_idx)};

    valid[rec] = valid[rec] | (bit_mask_t{1} << bit);
  }

  /**
   * determine which bit in a bitmap relates to a record
   * @param[in]  record_idx    The record index
   * @return which bit within the bitmap
   */
  template <typename T>
  CUDA_HOST_DEVICE_CALLABLE
  void clear_bit_unsafe(bit_mask_t *valid, T record_idx) {
    static_assert(std::is_integral<T>::value, "Record index must be of an integ4al type");

    const gdf_size_type rec{which_word(record_idx)};
    const gdf_size_type bit{which_bit(record_idx)};

    valid[rec] = valid[rec] & (~(bit_mask_t{1} << bit));
  }

  namespace details {
    /**
     * determine the bitmap that contains a record
     * @param[in]  record_idx    The record index
     * @return the bitmap index
     */
    template <typename T>
    CUDA_HOST_DEVICE_CALLABLE
    constexpr gdf_size_type which_word(T record_idx) {
      return (record_idx / BITS_PER_WORD);
    }

    /**
     * determine which bit in a bitmap relates to a record
     * @param[in]  record_idx    The record index
     * @return which bit within the bitmap
     */
    template <typename T>
    CUDA_HOST_DEVICE_CALLABLE
    constexpr gdf_size_type which_bit(T record_idx) {
      return (record_idx % BITS_PER_WORD);
    }
  }
};

#endif
