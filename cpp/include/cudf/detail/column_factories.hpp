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
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include "../column/column_factories.hpp"

#include <rmm/thrust_rmm_allocator.h>

namespace cudf {
namespace experimental {
namespace detail {  

/**
 * @brief Construct column with sufficient uninitialized storage
 * to hold `size` elements of the specified numeric `data_type` with a
 * null mask.
 *
 * @note null_count is optional and will be computed if not provided.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a numeric type
 *
 * @param[in] type The desired numeric element type
 * @param[in] size The number of elements in the column
 * @param[in] null_mask Null mask to use for this column.
 * @param[in] null_count Optional number of nulls in the null_mask.
 * @param[in] stream Optional stream on which to issue all memory allocation and
 * device kernels
 * @param[in] mr Optional resource to use for device memory
 * allocation of the column's `data` and `null_mask`.
 */
template <typename B>
std::unique_ptr<column> make_numeric_column(
    data_type type, size_type size, B&& null_mask,
    size_type null_count = cudf::UNKNOWN_NULL_COUNT, cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
  CUDF_EXPECTS(is_numeric(type), "Invalid, non-numeric type.");
  return std::make_unique<column>(
      type, size, rmm::device_buffer{size * cudf::size_of(type), stream, mr},
      std::forward<B>(null_mask), null_count);
}

/**
 * @brief Construct column with sufficient uninitialized storage
 * to hold `size` elements of the specified timestamp `data_type` with a
 * null mask.
 *
 * @note null_count is optional and will be computed if not provided.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a timestamp type
 *
 * @param[in] type The desired timestamp element type
 * @param[in] size The number of elements in the column
 * @param[in] null_mask Null mask to use for this column.
 * @param[in] null_count Optional number of nulls in the null_mask.
 * @param[in] stream Optional stream on which to issue all memory allocation and
 * device kernels
 * @param[in] mr Optional resource to use for device memory
 * allocation of the column's `data` and `null_mask`.
 */
template <typename B>
std::unique_ptr<column> make_timestamp_column(
    data_type type, size_type size, B&& null_mask,
    size_type null_count = cudf::UNKNOWN_NULL_COUNT, cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
  CUDF_EXPECTS(is_timestamp(type), "Invalid, non-timestamp type.");
  return std::make_unique<column>(
      type, size, rmm::device_buffer{size * cudf::size_of(type), stream, mr},
      std::forward<B>(null_mask), null_count);
}

/**
 * @brief Construct column with sufficient uninitialized storage
 * to hold `size` elements of the specified fixed width `data_type` with a
 * null mask.
 *
 * @note null_count is optional and will be computed if not provided.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a fixed width type
 *
 * @param[in] type The desired fixed width element type
 * @param[in] size The number of elements in the column
 * @param[in] null_mask Null mask to use for this column.
 * @param[in] null_count Optional number of nulls in the null_mask.
 * @param[in] stream Optional stream on which to issue all memory allocation and device
 * kernels
 * @param[in] mr Optional resource to use for device memory
 * allocation of the column's `data` and `null_mask`.
 */
template <typename B>
std::unique_ptr<column> make_fixed_width_column(
    data_type type, size_type size,
    B&& null_mask,
    size_type null_count = cudf::UNKNOWN_NULL_COUNT, cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  CUDF_EXPECTS(is_fixed_width(type), "Invalid, non-fixed-width type.");
  if(is_timestamp(type)){
    return make_timestamp_column(type, size, std::forward<B>(null_mask), null_count, stream, mr);
  }
  return make_numeric_column(type, size, std::forward<B>(null_mask), null_count, stream, mr);  
}

}  // namespace detail
}  // namespace experimental
}  // namespace cudf
