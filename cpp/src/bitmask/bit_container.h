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

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE_CALLABLE __host__ __device__ __forceinline__
#define CUDA_DEVICE_CALLABLE __device__ __forceinline__
#define CUDA_LAUNCHABLE __global__
#else
#define CUDA_HOST_DEVICE_CALLABLE
#define CUDA_DEVICE_CALLABLE
#define CUDA_LAUNCHABLE
#endif


namespace bit_container {
  typedef uint32_t     bit_container_t;
  //typedef size_t       bit_size_t;
  typedef uint32_t       bit_size_t;

  const bit_size_t BITS_PER_WORD = 32;
  
  /**
   * determine the number of bit_container_t words are used
   * @param[in]  size    Number of bits in the bitmask
   * @return the number of words
   */
  CUDA_HOST_DEVICE_CALLABLE
  bit_size_t num_words(bit_size_t size) { 
    return (( size + ( BITS_PER_WORD - 1)) / BITS_PER_WORD ); 
  }

  /**
   * determine the bitmap that contains a record
   * @param[in]  record_idx    The record index
   * @return the bitmap index
   */
  template <typename T>
  CUDA_HOST_DEVICE_CALLABLE
  bit_size_t which_word(T record_idx) {
    return (record_idx / BITS_PER_WORD);
  }

  /**
   * determine which bit in a bitmap relates to a record
   * @param[in]  record_idx    The record index
   * @return which bit within the bitmap
   */
  template <typename T>
  CUDA_HOST_DEVICE_CALLABLE
  bit_size_t which_bit(T record_idx) {
    return (record_idx % BITS_PER_WORD);
  }

  /**
   * determine which bit in a bitmap relates to a record
   * @param[in]  record_idx    The record index
   * @return which bit within the bitmap
   */
  template <typename T>
  CUDA_HOST_DEVICE_CALLABLE
  bit_size_t is_valid(const bit_container_t *valid, T record_idx) {
    int rec = which_word(record_idx);
    int bit = which_bit(record_idx);

    return ((valid[rec] & (1U << bit)) != 0);
  }

  /**
   * determine which bit in a bitmap relates to a record
   * @param[in]  record_idx    The record index
   * @return which bit within the bitmap
   */
  template <typename T>
  CUDA_HOST_DEVICE_CALLABLE
  void set_bit_unsafe(bit_container_t *valid, T record_idx) {
    int rec = which_word(record_idx);
    int bit = which_bit(record_idx);

    valid[rec] = valid[rec] | (1U << bit);
  }

  /**
   * determine which bit in a bitmap relates to a record
   * @param[in]  record_idx    The record index
   * @return which bit within the bitmap
   */
  template <typename T>
  CUDA_HOST_DEVICE_CALLABLE
  void clear_bit_unsafe(const bit_container_t *valid, T record_idx) {
    int rec = which_word(record_idx);
    int bit = which_bit(record_idx);

    valid[rec] = valid[rec] & (~(1U << bit));
  }

};

#endif
