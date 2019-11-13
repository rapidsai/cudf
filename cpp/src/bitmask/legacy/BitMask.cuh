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

#ifndef _BIT_MASK_H_
#define _BIT_MASK_H_

#include "bit_mask.cuh"
#include <cudf/utilities/error.hpp>
#include <cudf/cudf.h>
#include <rmm/rmm.h>

using bit_mask_t = bit_mask::bit_mask_t;


/* ---------------------------------------------------------------------------- *
 * @brief  Class for managing bit containers on the device
 * ---------------------------------------------------------------------------- */
class BitMask {
public:
   __host__ __device__ BitMask(bit_mask_t *valid, int bitlength): valid_(valid), bitlength_(bitlength) {}

  /**
   * @brief Check to see if a record is Valid (aka not null)
   *
   * @param[in] record_idx   the record index to check
   *
   * @return  true if record is valid, false if record is null
   */
  template <typename T>
  __device__ bool is_valid(T record_idx) const {
    return bit_mask::is_valid(valid_, record_idx);
  }

  /**
   * @brief Set a bit (not thread-save)
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void set_bit_unsafe(T record_idx) {
    bit_mask::set_bit_unsafe(valid_, record_idx);
  }


  /**
   * @brief Clear a bit (not thread-safe)
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void clear_bit_unsafe(T record_idx) {
    bit_mask::clear_bit_unsafe(valid_, record_idx);
  }

  /**
   * @brief Set a bit in a thread-safe manner
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void set_bit(T record_idx) {
    bit_mask::set_bit_safe(valid_, record_idx);
  }


  /**
   * @brief Clear a bit in a thread-safe manner
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void clear_bit(T record_idx) {
    bit_mask::clear_bit_safe(valid_, record_idx);
  }

  /**
   * @brief Get the number of elements in this bit container
   *
   * @return the number of elements
   */
  __device__ cudf::size_type num_elements() const {
    return bit_mask::num_elements(bitlength_);
  }

  /**
   * @brief Get a reference to a specific element (device only)
   *
   * @param[in] element_idx
   *
   * @return reference to the specified element
   */
  __device__ bit_mask_t &get_element_device(cudf::size_type element_idx) {
    return valid_[element_idx];
  }

  /**
   * @brief Get a specific element (host only)
   *
   *  @param[in]  element_idx
   *  @param[out] element
   *
   *  @return GDF_SUCCESS on success, the CUDA error on failure
   */
  __host__ gdf_error get_element_host(cudf::size_type element_idx, bit_mask_t &element) const {
    return bit_mask::get_element(&element, valid_ + element_idx);
  }

  /**
   * @brief Set a specific element (host only)
   *
   * @param[in] element_idx
   *
   * @return the specified element
   */
  __host__ gdf_error set_element_host(cudf::size_type element_idx, const bit_mask_t &element) {
    return bit_mask::put_element(element, valid_ + element_idx);
  }

  /**
   * @brief Getter for valid
   *
   * @return pointer to valid bit container
   */
  __host__ __device__ bit_mask_t *get_valid() const {
    return valid_;
  }

  /**
   * @brief Get length
   *
   * @return length of bit container in bits
   */
  __host__ __device__ cudf::size_type length() const {
    return bitlength_;
  }

private:
  bit_mask_t      *valid_;
  cudf::size_type    bitlength_;
};

#endif
