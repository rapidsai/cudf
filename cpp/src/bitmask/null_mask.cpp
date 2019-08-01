/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/null_mask.hpp>
#include <utilities/error_utils.hpp>
#include <utilities/integer_utils.hpp>

#include <rmm/device_buffer.hpp>

namespace cudf {

size_type state_null_count(mask_state state, size_type size) {
  switch (state) {
    case UNALLOCATED:
      return 0;
    case UNINITIALIZED:
      return UNKNOWN_NULL_COUNT;
    case ALL_NULL:
      return size;
    case ALL_VALID:
      return 0;
    default:
      CUDF_FAIL("Invalid null mask state.");
  }
}

// Computes required allocation size of a bitmask
std::size_t bitmask_allocation_size_bytes(size_type number_of_bits,
                                          std::size_t padding_boundary) {
  CUDF_EXPECTS(padding_boundary > 0, "Invalid padding boundary");
  auto neccessary_bytes =
      cudf::util::div_rounding_up_safe<size_type>(number_of_bits, CHAR_BIT);

  auto padded_bytes =
      padding_boundary * cudf::util::div_rounding_up_safe<size_type>(
                             neccessary_bytes, padding_boundary);
  return padded_bytes;
}

// Create a device_buffer for a null mask
rmm::device_buffer create_null_mask(size_type size, mask_state state,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource *mr) {
  size_type mask_size{0};

  if (state != UNALLOCATED) {
    mask_size = bitmask_allocation_size_bytes(size);
  }

  uint8_t fill_value = (state == ALL_VALID) ? 0xff : 0x00;

  rmm::device_buffer mask(mask_size, stream, mr);

  if (state != UNINITIALIZED) {
    CUDA_TRY(cudaMemsetAsync(static_cast<bitmask_type *>(mask.data()),
                             fill_value, mask_size, stream));
  }

  return mask;
}

}  // namespace cudf