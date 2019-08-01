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
#pragma once

#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utils/traits.hpp>
#include "column.hpp"

#include <rmm/mr/device_memory_resource.hpp>

#include <memory>

namespace cudf {

namespace {
struct size_of_helper {
  template <typename T>
  constexpr int operator()() const noexcept {
    return sizeof(T);
  }
};

/**
 * @brief Returns the size in bytes of elements of the specified `data_type`
 *
 * @note Only fixed-width types are supported
 *
 * @throws cudf::logic_error if `is_fixed_width(element_type) == false`
 *
 * TODO: This should go somewhere else
 */
constexpr inline std::size_t size_of(data_type element_type) {
  CUDF_EXPECTS(is_fixed_width(element_type), "Invalid element type.");
  return cudf::exp::type_dispatcher(element_type, size_of_helper{});
}

data_type verify_fixed_width_and_simple(data_type type) {
  CUDF_EXPECTS(cudf::is_fixed_width(type) and cudf::is_simple(type),
               "Invalid element type.");
  return type;
}

}  // namespace

// Allocate storage for a specified number of fixed-width elements
/*
column::column(data_type type, size_type size, mask_state state,
               cudaStream_t stream, rmm::mr::device_memory_resource *mr)
    : _type{verify_fixed_width_and_simple(type)},
      _size{size},
      _data{size * cudf::size_of(type), stream, mr},
      _null_mask{create_null_mask(size, state, stream, mr)} {
  switch (state) {
    case UNALLOCATED:
      _null_count = 0;
      break;
    case UNINITIALIZED:
      _null_count = UNKNOWN_NULL_COUNT;
      break;
    case ALL_NULL:
      _null_count = size;
      break;
    case ALL_VALID:
      _null_count = 0;
      break;
  }
}
*/

/**---------------------------------------------------------------------------*
 * @brief Construct a new column and allocate sufficient uninitialized storage
 * to hold `size` elements of the specified numeric `data_type` with an optional
 * null mask allocation.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a numeric type
 *
 * @param[in] type The desired numeric element type
 * @param[in] size The number of elements in the column
 * @param[in] state Optional, controls allocation/initialization of the
 * column's null mask. By default, no null mask is allocated.
 * @param[in] stream Optional stream on which all memory allocation and device
 * kernels will be issued.
 * @param[in] mr Optional resource that will be used for device memory
 * allocation of the column's `data` and `null_mask`.
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> make_numeric_column(
    data_type type, size_type size, mask_state state = UNALLOCATED,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
}  // namespace cudf
