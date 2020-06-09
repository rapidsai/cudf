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

#include <rmm/thrust_rmm_allocator.h>
#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf {
/**
 * @ingroup column_factories Factories
 * @{
 */

/**
 * @brief Creates an empty column of the specified @p type
 *
 * An empty column does not contain any elements or a validity mask.
 *
 * @param type The desired type
 * @return Empty column with desired type
 */
std::unique_ptr<column> make_empty_column(data_type type);

/**
 * @brief Construct column with sufficient uninitialized storage
 * to hold `size` elements of the specified numeric `data_type` with an optional
 * null mask.
 *
 * @note `null_count()` is determined by the requested null mask `state`
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a numeric type
 *
 * @param[in] type The desired numeric element type
 * @param[in] size The number of elements in the column
 * @param[in] state Optional, controls allocation/initialization of the
 * column's null mask. By default, no null mask is allocated.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> make_numeric_column(
  data_type type,
  size_type size,
  mask_state state                    = mask_state::UNALLOCATED,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

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
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 */
template <typename B>
std::unique_ptr<column> make_numeric_column(
  data_type type,
  size_type size,
  B&& null_mask,
  size_type null_count                = cudf::UNKNOWN_NULL_COUNT,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  CUDF_EXPECTS(is_numeric(type), "Invalid, non-numeric type.");
  return std::make_unique<column>(type,
                                  size,
                                  rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                  std::forward<B>(null_mask),
                                  null_count);
}

/**
 * @brief Construct column with sufficient uninitialized storage
 * to hold `size` elements of the specified timestamp `data_type` with an
 * optional null mask.
 *
 * @note `null_count()` is determined by the requested null mask `state`
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a timestamp type
 *
 * @param[in] type The desired timestamp element type
 * @param[in] size The number of elements in the column
 * @param[in] state Optional, controls allocation/initialization of the
 * column's null mask. By default, no null mask is allocated.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> make_timestamp_column(
  data_type type,
  size_type size,
  mask_state state                    = mask_state::UNALLOCATED,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

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
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 */
template <typename B>
std::unique_ptr<column> make_timestamp_column(
  data_type type,
  size_type size,
  B&& null_mask,
  size_type null_count                = cudf::UNKNOWN_NULL_COUNT,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  CUDF_EXPECTS(is_timestamp(type), "Invalid, non-timestamp type.");
  return std::make_unique<column>(type,
                                  size,
                                  rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                  std::forward<B>(null_mask),
                                  null_count);
}

/**
 * @brief Construct column with sufficient uninitialized storage
 * to hold `size` elements of the specified duration `data_type` with an
 * optional null mask.
 *
 * @note `null_count()` is determined by the requested null mask `state`
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a duration type
 *
 * @param[in] type The desired duration element type
 * @param[in] size The number of elements in the column
 * @param[in] state Optional, controls allocation/initialization of the
 * column's null mask. By default, no null mask is allocated.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> make_duration_column(
  data_type type,
  size_type size,
  mask_state state                    = mask_state::UNALLOCATED,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Construct column with sufficient uninitialized storage
 * to hold `size` elements of the specified duration `data_type` with a
 * null mask.
 *
 * @note null_count is optional and will be computed if not provided.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a duration type
 *
 * @param[in] type The desired duration element type
 * @param[in] size The number of elements in the column
 * @param[in] null_mask Null mask to use for this column.
 * @param[in] null_count Optional number of nulls in the null_mask.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 */
template <typename B>
std::unique_ptr<column> make_duration_column(
  data_type type,
  size_type size,
  B&& null_mask,
  size_type null_count                = cudf::UNKNOWN_NULL_COUNT,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  CUDF_EXPECTS(is_duration(type), "Invalid, non-duration type.");
  return std::make_unique<column>(type,
                                  size,
                                  rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                  std::forward<B>(null_mask),
                                  null_count);
}

/**
 * @brief Construct column with sufficient uninitialized storage
 * to hold `size` elements of the specified fixed width `data_type` with an optional
 * null mask.
 *
 * @note `null_count()` is determined by the requested null mask `state`
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a fixed width type
 *
 * @param[in] type The desired fixed width type
 * @param[in] size The number of elements in the column
 * @param[in] state Optional, controls allocation/initialization of the
 * column's null mask. By default, no null mask is allocated.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> make_fixed_width_column(
  data_type type,
  size_type size,
  mask_state state                    = mask_state::UNALLOCATED,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

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
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 */
template <typename B>
std::unique_ptr<column> make_fixed_width_column(
  data_type type,
  size_type size,
  B&& null_mask,
  size_type null_count                = cudf::UNKNOWN_NULL_COUNT,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  CUDF_EXPECTS(is_fixed_width(type), "Invalid, non-fixed-width type.");
  if (is_timestamp(type)) {
    return make_timestamp_column(type, size, std::forward<B>(null_mask), null_count, stream, mr);
  } else if (is_duration(type)) {
    return make_duration_column(type, size, std::forward<B>(null_mask), null_count, stream, mr);
  }
  return make_numeric_column(type, size, std::forward<B>(null_mask), null_count, stream, mr);
}

/**
 * @brief Construct STRING type column given a device vector of pointer/size pairs.
 * The total number of char bytes must not exceed the maximum size of size_type.
 * The string characters are expected to be UTF-8 encoded sequence of char
 * bytes. Use the strings_column_view class to perform strings operations on
 * this type of column.
 *
 * @note `null_count()` and `null_bitmask` are determined if a pair contains
 * a null string. That is, for each pair, if `.first` is null, that string
 * is considered null. Likewise, a string is considered empty (not null)
 * if `.first` is not null and `.second` is 0. Otherwise the `.first` member
 * must be a valid device address pointing to `.second` consecutive bytes.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param strings The vector of pointer/size pairs.
 *                Each pointer must be a device memory address or `nullptr`
 * (indicating a null string). The size must be the number of bytes.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used for allocation of the column's `null_mask` and children
 * columns' device memory.
 */
std::unique_ptr<column> make_strings_column(
  const rmm::device_vector<thrust::pair<const char*, size_type>>& strings,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Construct STRING type column given a device vector of string_view.
 * The total number of char bytes must not exceed the maximum size of size_type.
 * The string characters are expected to be UTF-8 encoded sequence of char
 * bytes. Use the strings_column_view class to perform strings operations on
 * this type of column.
 *
 * @note For each string_view, if `.data()` is `null_placeholder.data()`, that
 * string is considered null. Likewise, a string is considered empty (not null)
 * if `.data()` is not `null_placeholder.data()` and `.size_bytes()` is 0.
 * Otherwise the `.data()` must be a valid device address pointing to
 * `.size_bytes()` consecutive bytes. The `null_count()` for the output column
 * will be equal to the number of input `string_view`s that are null.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param string_views The vector of string_view.
 *                Each string_view must point to a device memory address or
 * `null_placeholder` (indicating a null string). The size must be the number of
 * bytes.
 * @param null_placeholder string_view indicating null string in given list of
 * string_views.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used for allocation of the column's `null_mask` and children
 * columns' device memory.
 */
std::unique_ptr<column> make_strings_column(
  const rmm::device_vector<string_view>& string_views,
  const string_view null_placeholder,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Construct STRING type column given a device vector of chars
 * encoded as UTF-8, a device vector of byte offsets identifying individual
 * strings within the char vector, and an optional null bitmask.
 *
 * `offsets.front()` must always be zero.
 *
 * The total number of char bytes must not exceed the maximum size of size_type.
 * Use the strings_column_view class to perform strings operations on this type
 * of column.
 * This function makes a deep copy of the strings, offsets, null_mask to create
 * a new column.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param strings The vector of chars in device memory.
 *                This char vector is expected to be UTF-8 encoded characters.
 * @param offsets The vector of byte offsets in device memory.
 *                The number of elements is one more than the total number
 *                of strings so the `offsets.back()` is the total
 *                number of bytes in the strings array.
 *                `offsets.front()` must always be 0 to point to the beginning
 *                of `strings`.
 * @param null_mask Device vector containing the null element indicator bitmask.
 *                  Arrow format for nulls is used for interpeting this bitmask.
 * @param null_count The number of null string entries. If equal to
 * `UNKNOWN_NULL_COUNT`, the null count will be computed dynamically on the
 * first invocation of `column::null_count()`
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used for allocation of the column's `null_mask` and children
 * columns' device memory.
 */
std::unique_ptr<column> make_strings_column(
  const rmm::device_vector<char>& strings,
  const rmm::device_vector<size_type>& offsets,
  const rmm::device_vector<bitmask_type>& null_mask = {},
  size_type null_count                              = cudf::UNKNOWN_NULL_COUNT,
  cudaStream_t stream                               = 0,
  rmm::mr::device_memory_resource* mr               = rmm::mr::get_default_resource());

/**
 * @brief Construct STRING type column given a host vector of chars
 * encoded as UTF-8, a host vector of byte offsets identifying individual
 * strings within the char vector, and an optional null bitmask.
 *
 * `offsets.front()` must always be zero.
 *
 * The total number of char bytes must not exceed the maximum size of size_type.
 * Use the strings_column_view class to perform strings operations on this type
 * of column.
 * This function makes a deep copy of the strings, offsets, null_mask to create
 * a new column.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param strings The contiguous array of chars in host memory.
 *                This char array is expected to be UTF-8 encoded characters.
 * @param offsets The array of byte offsets in host memory.
 *                The number of elements is one more than the total number
 *                of strings so the `offsets.back()` is the total
 *                number of bytes in the strings array.
 *                `offsets.front()` must always be 0 to point to the beginning
 *                of `strings`.
 * @param null_mask Host vector containing the null element indicator bitmask.
 *                  Arrow format for nulls is used for interpeting this bitmask.
 * @param null_count The number of null string entries. If equal to
 * `UNKNOWN_NULL_COUNT`, the null count will be computed dynamically on the
 * first invocation of `column::null_count()`
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used for allocation of the column's `null_mask` and children
 * columns' device memory.
 */
std::unique_ptr<column> make_strings_column(
  const std::vector<char>& strings,
  const std::vector<size_type>& offsets,
  const std::vector<bitmask_type>& null_mask = {},
  size_type null_count                       = cudf::UNKNOWN_NULL_COUNT,
  cudaStream_t stream                        = 0,
  rmm::mr::device_memory_resource* mr        = rmm::mr::get_default_resource());

/**
 * @brief Constructs a STRING type column given offsets column, chars columns,
 * and null mask and null count. The columns and mask are moved into the
 * resulting strings column.
 *
 * @param num_strings The number of strings the column represents.
 * @param offsets_column The column of offset values for this column.
 *                       The number of elements is one more than the total number
 *                       of strings so the offset[last] - offset[0] is the total
 *                       number of bytes in the strings vector.
 * @param chars_column The column of char bytes for all the strings for this column.
 *                     Individual strings are identified by the offsets and the
 *                     nullmask.
 * @param null_count The number of null string entries.
 * @param null_mask The bits specifying the null strings in device memory.
 *                  Arrow format for nulls is used for interpeting this bitmask.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used for allocation of the column's `null_mask` and children
 * columns' device memory.
 */
std::unique_ptr<column> make_strings_column(
  size_type num_strings,
  std::unique_ptr<column> offsets_column,
  std::unique_ptr<column> chars_column,
  size_type null_count,
  rmm::device_buffer&& null_mask,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Return a column with size elements that are all equal to the
 * given scalar.
 *
 * The output column will have the same type as `s.type()`
 * The output column will contain all null rows if `s.invalid()==false`
 * The output column will be empty if `size==0`.
 *
 * @param s The scalar to use for values in the column.
 * @param size The number of rows for the output column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
std::unique_ptr<column> make_column_from_scalar(
  scalar const& s,
  size_type size,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/** @} */  // end of group
}  // namespace cudf
