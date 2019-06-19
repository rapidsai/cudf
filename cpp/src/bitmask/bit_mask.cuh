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
#include <cudf/types.hpp>
#include <utilities/bit_util.cuh>
#include <utilities/error_utils.hpp>
#include <utilities/integer_utils.hpp>

namespace bit_mask {
enum { bits_per_element = cudf::util::size_in_bits<bit_mask_t>() };

/**
 * @brief determine the number of bit_mask_t elements required for @p size bits
 *
 * @param[in]  size    Number of bits in the bitmask
 *
 * @return the number of elements
 */
CUDA_HOST_DEVICE_CALLABLE
constexpr gdf_size_type num_elements(gdf_size_type number_of_bits) {
  return cudf::util::div_rounding_up_safe<gdf_size_type>(number_of_bits, bits_per_element);
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
                               size_t number_of_bits, enum cudaMemcpyKind kind) {
  CUDA_TRY(
      cudaMemcpy(dst, src, num_elements(number_of_bits) * sizeof(bit_mask_t), kind));
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

namespace detail {

template <typename T>
constexpr inline T gcd(T u, T v) noexcept
{
    while (v != 0) {
        auto remainder = u % v;
        u = v;
        v = remainder;
    }
    return u;
}

template <typename T>
constexpr inline T lcm(T u, T v) noexcept
{
        return (u / gcd(u,v)) * v;
}

} // namespace detail

/**
 *  @brief Allocate device space for the valid bit mask.
 *
 *  @param[out] mask                  address of the bit mask pointer
 *  @param[in]  num_elements          number of elements in the bit mask
 *  @param[in]  fill_value            optional, should the memory be initialized to all 0 or 1s. All other
 *                                    values indicate un-initialized.  Default is uninitialized
 *  @param[in]  padding_boundary      optional, specifies the quantum, in bytes, of the amount of memory
 *                                    allocated (i.e. the actually-required allocation size is "padded" to
 *                                    a multiple of this value).
 *
 *  @return GDF_SUCCESS on success, the RMM or CUDA error on error
 */
inline gdf_error create_bit_mask(bit_mask_t **mask,
                                 gdf_size_type number_of_bits,
                                 int fill_value = -1,
                                 gdf_size_type padding_boundary = 64)
{
  // We assume RMM_ALLOC satisfies the allocation alignment for the beginning
  // of the allocated space; we ensure its end also has that allocation.
  //
  // TODO: The assumption may not be valid

  padding_boundary = detail::lcm<gdf_size_type>(sizeof(bit_mask_t), padding_boundary);
  gdf_size_type num_quanta_to_allocate =
      cudf::util::div_rounding_up_safe<gdf_size_type>(
          number_of_bits, CHAR_BIT * padding_boundary);

  RMM_TRY(RMM_ALLOC(mask, padding_boundary * num_quanta_to_allocate, 0));

  if (fill_value == 0) {
    CUDA_TRY(cudaMemset(*mask, 0, padding_boundary * num_quanta_to_allocate));
  } else if (fill_value == 1) {
    //
    //  Value outside range of [0, num_rows) is undefined, so we will
    //  initialize in the simplest manner... we'll initialize all
    //  elements to 1.
    //
    CUDA_TRY(cudaMemset(*mask, 0xff, padding_boundary * num_quanta_to_allocate));
  }

  return GDF_SUCCESS;
}

/**
 *  @brief check to see if the specified bit is set to one
 *
 *  Note that for performance reasons (this is often called in inner loops
 *  in CUDA device code) this function does not verify that @p valid is non-null.
 *  That should be checked in a wider scope, since it usually doesn't vary for 
 *  different threads of the same kernel.
 *
 *  @param[in]  valid         The bit mask to update
 *  @param[in]  bit_index     Index of the bit within the mask
 *
 *  @return which bit within the bit mask
 */
template <typename T>
CUDA_HOST_DEVICE_CALLABLE bool is_valid(bit_mask_t const * valid, T bit_index) {
  static_assert(std::is_integral<T>::value,
                "Record index must be of an integral type");

  return cudf::util::bit_is_set<bit_mask_t, T>(valid, bit_index);
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
 *  @param[in]      bit_index     Index of the bit within the mask
 *
 */
template <typename T>
CUDA_HOST_DEVICE_CALLABLE void set_bit_unsafe(bit_mask_t *valid, T bit_index) {
  static_assert(std::is_integral<T>::value,
                "Record index must be of an integral type");

  cudf::util::turn_bit_on(valid, bit_index);
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
 *  @param[in]      bit_index     Index of the bit within the mask
 *
 */
template <typename T>
CUDA_HOST_DEVICE_CALLABLE void clear_bit_unsafe(bit_mask_t *valid,
                                                T bit_index) {
  static_assert(std::is_integral<T>::value,
                "Record index must be of an integral type");

  return cudf::util::turn_bit_off(valid, bit_index);
}

/**
 *  @brief set the specified bit in the bit mask in an threadsafe manner
 *
 *  This function sets the specified bit in an threadsafe manner.  It uses
 *  atomic memory operations to guarantee that no update interferes with
 *  on another.
 *
 *  @param[in,out]  valid         The bit mask to update
 *  @param[in]      bit_index     Index of the bit within the mask
 *
 */
template <typename T>
CUDA_DEVICE_CALLABLE void set_bit_safe(bit_mask_t *valid, T bit_index) {
  static_assert(std::is_integral<T>::value,
                "Record index must be of an integral type");

  const gdf_size_type rec{
      cudf::util::detail::bit_container_index<bit_mask_t, T>(bit_index)};
  const gdf_size_type bit{
      cudf::util::detail::intra_container_index<bit_mask_t, T>(bit_index)};

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
 *  @param[in]      bit_index     Index of the bit within the mask
 *
 */
template <typename T>
CUDA_DEVICE_CALLABLE void clear_bit_safe(bit_mask_t *valid, T bit_index) {
  static_assert(std::is_integral<T>::value,
                "Record index must be of an integral type");

  const gdf_size_type rec{
      cudf::util::detail::bit_container_index<bit_mask_t, T>(bit_index)};
  const gdf_size_type bit{
      cudf::util::detail::intra_container_index<bit_mask_t, T>(bit_index)};

  atomicAnd(&valid[rec], ~(bit_mask_t{1} << bit));
}

}  // namespace bit_mask

#endif
