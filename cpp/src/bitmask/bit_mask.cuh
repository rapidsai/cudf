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

#ifndef CUDF_BIT_MASK_CUH
#define CUDF_BIT_MASK_CUH

#include <cuda_runtime.h>
#include <utilities/cudf_utils.h>
#include <types.hpp>
#include <utilities/bit_util.cuh>
#include <utilities/error_utils.hpp>
#include <utilities/integer_utils.hpp>

namespace bit_mask {
enum { bits_per_element = gdf::util::size_in_bits<bit_mask_t>() };

/**
 * @brief determine the number of bit_mask_t elements required for @p size bits
 *
 * @param[in]  size    Number of bits in the bitmask
 *
 * @return the number of elements
 */
CUDA_HOST_DEVICE_CALLABLE
constexpr gdf_size_type num_elements(gdf_size_type size) {
  return cudf::util::div_rounding_up_unsafe<gdf_size_type>(size,
                                                           bits_per_element);
}

/**
 *  @brief Copy bit mask data between host and device
 *
 *  @param[out] dst       the address of the destination
 *  @param[in] src        the address of the source
 *  @param[in] num_bits   the number of bits in the bit container
 *  @param[in] kind       the direction of the copy
 *
 *  @return GDF_SUCCESS on success, the CUDA error on failure
 */
inline gdf_error copy_bit_mask(bit_mask_t *dst, const bit_mask_t *src,
                               size_t num_bits, enum cudaMemcpyKind kind) {
  CUDA_TRY(
      cudaMemcpy(dst, src, num_elements(num_bits) * sizeof(bit_mask_t), kind));
  return GDF_SUCCESS;
}

/**
 *  @brief Deallocate device space for the valid bit mask
 *
 *  @param[in]  valid   The pointer to device space that we wish to deallocate
 *
 *  @return GDF_SUCCESS on success, the CUDA error on failure
 */
inline gdf_error destroy_bit_mask(bit_mask_t *valid) {
  RMM_TRY(RMM_FREE(valid, 0));
  return GDF_SUCCESS;
}

/**
 *  @brief Get a single element of bits from the device
 *
 *  @param[out]  element  address on host where the bits will be stored
 *  @param[out]  device_element  address on the device containing the bits to
 * fetch
 *
 *  @return GDF_SUCCESS on success, the CUDA error on failure
 */
inline gdf_error get_element(bit_mask_t *element,
                             const bit_mask_t *device_element) {
  CUDA_TRY(cudaMemcpy(element, device_element, sizeof(bit_mask_t),
                      cudaMemcpyDeviceToHost));
  return GDF_SUCCESS;
}

/**
 *  @brief Put a single element of bits to the device
 *
 *  @param[out]  element  address on host containing bits to store
 *  @param[out]  device_element address on the device where the bits will be
 * stored
 *
 *  @return GDF_SUCCESS on success, the CUDA error on failure
 */
inline gdf_error put_element(bit_mask_t element, bit_mask_t *device_element) {
  CUDA_TRY(cudaMemcpy(device_element, &element, sizeof(bit_mask_t),
                      cudaMemcpyHostToDevice));
  return GDF_SUCCESS;
}

/**
 *  @brief Allocate device space for the valid bit mask.
 *
 *  @param[out] mask                  address of the bit mask pointer
 *  @param[in]  number_of_records     number of records
 *  @param[in]  fill_value            Initialize all bits to fill_value if and
 * only if it is 0 or 1
 *  @param[in]  padding_bytes         The byte boundary allocation should be
 * padded to. Default: 64, meaning the allocation size is rounded up to next
 * multiple of 64 bytes.
 *
 *  @return GDF_SUCCESS on success, the RMM or CUDA error on error
 */
inline gdf_error create_bit_mask(bit_mask_t **mask,
                                 gdf_size_type number_of_records,
                                 int fill_value = -1,
                                 gdf_size_type padding_bytes = 64) {
  //
  //  To handle padding, we will round the number_of_records up to the next
  //  padding boundary, then identify how many element that equates to.  Then we
  //  can allocate the appropriate amount of storage.
  //
  gdf_size_type num_bytes =
      cudf::util::div_rounding_up_unsafe<gdf_size_type>(number_of_records, 8);

  gdf_size_type num_padding_blocks =
      cudf::util::div_rounding_up_unsafe<gdf_size_type>(num_bytes,
                                                        padding_bytes);
  gdf_size_type num_elements =
      bit_mask::num_elements(num_padding_blocks * 8 * padding_bytes);

  RMM_TRY(RMM_ALLOC(mask, sizeof(bit_mask_t) * num_elements, 0));

  if (fill_value == 0) {
    CUDA_TRY(cudaMemset(*mask, 0, sizeof(bit_mask_t) * num_elements));
  } else if (fill_value == 1) {
    //
    //  Value outside range of [0, num_rows) is undefined, so we will
    //  initialize in the simplest manner... we'll initialize all
    //  elements to 1.
    //
    CUDA_TRY(cudaMemset(*mask, 0xff, sizeof(bit_mask_t) * num_elements));
  }

  return GDF_SUCCESS;
}

/**
 *  @brief check to see if the specified bit is set to one
 *
 *  @param[in]  valid         The bit mask to update
 *  @param[in]  record_idx    The record index
 *
 *  @return which bit within the bit mask
 */
template <typename T>
CUDA_HOST_DEVICE_CALLABLE bool is_valid(const bit_mask_t *valid, T record_idx) {
  static_assert(std::is_integral<T>::value,
                "Record index must be of an integral type");

  return gdf::util::bit_is_set<bit_mask_t, T>(valid, record_idx);
}

/**
 *  @brief set the specified bit in the bit mask in an unsafe manner
 *
 *  This function sets the specified bit in an unsafe manner.  It assumes that
 *  that the calling code guarantees a thread-safe context.  That is, either
 *  the function is called from a block of serial code, or the data is
 * distributed among the threads such that no two threads could be updating a
 * bit in the same memory location concurrently.
 *
 *  @param[in,out]  valid         The bit mask to update
 *  @param[in]      record_idx    The record index
 *
 */
template <typename T>
CUDA_HOST_DEVICE_CALLABLE void set_bit_unsafe(bit_mask_t *valid, T record_idx) {
  static_assert(std::is_integral<T>::value,
                "Record index must be of an integral type");

  gdf::util::turn_bit_on(valid, record_idx);
}

/**
 *  @brief clear the specified bit in the bit mask in an unsafe manner.
 *
 *  This function clears the specified bit in an unsafe manner.  It assumes that
 *  that the calling code guarantees a thread-safe context.  That is, either
 *  the function is called from a block of serial code, or the data is
 * distributed among the threads such that no two threads could be updating a
 * bit in the same memory location concurrently.
 *
 *  @param[in,out]  valid         The bit mask to update
 *  @param[in]      record_idx    The record index
 *
 */
template <typename T>
CUDA_HOST_DEVICE_CALLABLE void clear_bit_unsafe(bit_mask_t *valid,
                                                T record_idx) {
  static_assert(std::is_integral<T>::value,
                "Record index must be of an integral type");

  return gdf::util::turn_bit_off(valid, record_idx);
}

/**
 *  @brief set the specified bit in the bit mask in an threadsafe manner
 *
 *  This function sets the specified bit in an threadsafe manner.  It uses
 *  atomic memory operations to guarantee that no update interferes with
 *  on another.
 *
 *  @param[in,out]  valid         The bit mask to update
 *  @param[in]      record_idx    The record index
 *
 */
template <typename T>
CUDA_DEVICE_CALLABLE void set_bit_safe(bit_mask_t *valid, T record_idx) {
  static_assert(std::is_integral<T>::value,
                "Record index must be of an integral type");

  const gdf_size_type rec{
      gdf::util::detail::bit_container_index<bit_mask_t, T>(record_idx)};
  const gdf_size_type bit{
      gdf::util::detail::intra_container_index<bit_mask_t, T>(record_idx)};

  atomicOr(&valid[rec], (bit_mask_t{1} << bit));
}

/**
 *  @brief clear the specified bit in the bit mask in an threadsafe manner
 *
 *  This function clear the specified bit in an threadsafe manner.  It uses
 *  atomic memory operations to guarantee that no update interferes with
 *  on another.
 *
 *  @param[in,out]  valid         The bit mask to update
 *  @param[in]      record_idx    The record index
 *
 */
template <typename T>
CUDA_DEVICE_CALLABLE void clear_bit_safe(bit_mask_t *valid, T record_idx) {
  static_assert(std::is_integral<T>::value,
                "Record index must be of an integral type");

  const gdf_size_type rec{
      gdf::util::detail::bit_container_index<bit_mask_t, T>(record_idx)};
  const gdf_size_type bit{
      gdf::util::detail::intra_container_index<bit_mask_t, T>(record_idx)};

  atomicAnd(&valid[rec], ~(bit_mask_t{1} << bit));
}

};  // namespace bit_mask

#endif
