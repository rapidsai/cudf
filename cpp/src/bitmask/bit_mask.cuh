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

#include "cudf.h"
#include "bit_mask.h"
#include "rmm/rmm.h"
#include "utilities/error_utils.h"
#include <cuda_runtime_api.h>

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
  __device__ bool IsValid(T record_idx) const {
    return bit_mask::IsValid(valid_, record_idx);
  }

  /**
   * @brief Set a bit (not thread-save)
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void SetBitUnsafe(T record_idx) {
    bit_mask::SetBitUnsafe(valid_, record_idx);
  }


  /**
   * @brief Clear a bit (not thread-safe)
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void ClearBitUnsafe(T record_idx) {
    bit_mask::ClearBitUnsafe(valid_, record_idx);
  }

  /**
   * @brief Set a bit in a thread-safe manner
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void SetBit(T record_idx) {
    bit_mask::SetBitSafe(valid_, record_idx);
  }


  /**
   * @brief Clear a bit in a thread-safe manner
   *
   * @param[in] record_idx    the record index
   */
  template <typename T>
  __device__ void ClearBit(T record_idx) {
    bit_mask::ClearBitSafe(valid_, record_idx);
  }

  /**
   * @brief Get the number of elements in this bit container
   *
   * @return the number of elements
   */
  __device__ gdf_size_type NumElements() const {
    return bit_mask::NumElements(bitlength_);
  }

  /**
   * @brief Get a reference to a specific element (device only)
   *
   * @param[in] element_idx
   *
   * @return reference to the specified element
   */
  __device__ bit_mask_t &GetElementDevice(gdf_size_type element_idx) {
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
  __host__ gdf_error GetElementHost(gdf_size_type element_idx, bit_mask_t &element) const {
    return bit_mask::GetElement(&element, valid_ + element_idx);
  }

  /**
   * @brief Set a specific element (host only)
   *
   * @param[in] element_idx
   *
   * @return the specified element
   */
  __host__ gdf_error SetElementHost(gdf_size_type element_idx, const bit_mask_t &element) {
    return bit_mask::PutElement(element, valid_ + element_idx);
  }

  /**
   * @brief Getter for valid
   *
   * @return pointer to valid bit container
   */
  __host__ __device__ bit_mask_t *GetValid() const {
    return valid_;
  }

  /**
   * @brief Get length
   *
   * @return length of bit container in bits
   */
  __host__ __device__ gdf_size_type Length() const {
    return bitlength_;
  }

private:
  bit_mask_t      *valid_;
  gdf_size_type    bitlength_;
};

#endif
