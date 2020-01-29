/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#pragma once

#include <cudf/types.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <vector>

namespace cudf {

namespace detail {

/**
 * @copydoc cudf::segmented_count_set_bits
 *
 * @param[in] stream Optional CUDA stream on which to execute kernels
 */
std::vector<size_type>
segmented_count_set_bits(bitmask_type const* bitmask,
                         std::vector<size_type> const& indices,
                         cudaStream_t stream = 0);

/**
 * @copydoc cudf::segmented_count_unset_bits
 *
 * @param[in] stream Optional CUDA stream on which to execute kernels
 */
std::vector<size_type>
segmented_count_unset_bits(bitmask_type const* bitmask,
                           std::vector<size_type> const& indices,
                           cudaStream_t stream = 0);

/**---------------------------------------------------------------------------*
 * @brief Concatenates `views[i]`'s bitmask from the bits
 * `[views[i].offset(), views[i].offset() + views[i].size())` for all elements
 * views[i] in views into an array
 *
 * @param views Vector of column views whose bitmask needs to be copied
 * @param dest_mask Pointer to array that contains the combined bitmask
 * of the column views
 * @param stream stream on which all memory allocations and copies
 * will be performed
 *---------------------------------------------------------------------------**/
void concatenate_masks(std::vector<column_view> const &views,
    bitmask_type * dest_mask,
    cudaStream_t stream);

/**
 * @brief Computes the required bytes necessary to represent the specified
 * number of bits with a given padding boundary. 
 * 
 * @note Does not throw when passed an invalid padding_boundary
 *
 * @note The Arrow specification for the null bitmask requires a 64B padding
 * boundary.
 *
 * @param number_of_bits The number of bits that need to be represented
 * @param padding_boundary The value returned will be rounded up to a multiple
 * of this value
 * @return std::size_t The necessary number of bytes
 **/
CUDA_HOST_DEVICE_CALLABLE std::size_t bitmask_allocation_size_bytes_nocheck(size_type number_of_bits,
                                          std::size_t padding_boundary = 64)
{ 
  auto necessary_bytes =
      cudf::util::div_rounding_up_safe<size_type>(number_of_bits, CHAR_BIT);

  auto padded_bytes =
      padding_boundary * cudf::util::div_rounding_up_safe<size_type>(
                             necessary_bytes, padding_boundary);
  return padded_bytes;
}

}  // namespace detail

}  // namespace cudf
