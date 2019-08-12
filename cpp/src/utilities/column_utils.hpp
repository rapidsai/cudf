/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Eyal Rozenberg <eyalroz@blazingdb.com>
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

/**
 * @file Small utility functions for working with @ref gdf_column structures,
 * which are raw C structures without any guarantees and operators defined for
 * them, and thus require a bit of care when handling. This care and due
 * diligence tends to be rather repetitive; hence this file.
 *
 * @note since the "right" way to work with these structures is pass references,
 * but the "C" way of doing the same involves pointers, many functions are
 * implemented for both.
 */
#ifndef UTILITIES_COLUMN_UTILS_HPP_
#define UTILITIES_COLUMN_UTILS_HPP_

#include <cudf/utilities/legacy/type_dispatcher.hpp>

#include <cudf/cudf.h>

#include <algorithm>

namespace cudf {

template<typename T>
inline T* get_data(const gdf_column& column) noexcept { return static_cast<T*>(column.data); }

template<typename T>
inline T* get_data(const gdf_column* column) noexcept { return get_data<T>(*column); }

namespace detail {

struct integral_check {
   template <typename T>
   constexpr bool operator()(void) const noexcept
   {
      return std::is_integral<T>::value;
   }
};

} // namespace detail

constexpr inline bool is_an_integer(gdf_dtype element_type) noexcept
{
   return cudf::type_dispatcher(element_type, detail::integral_check{});
}

constexpr inline bool is_integral(const gdf_column& column) noexcept { return is_an_integer(column.dtype); }
constexpr inline bool is_integral(const gdf_column* column) noexcept{ return is_integral(*column); }

constexpr inline bool is_nullable(const gdf_column& column) noexcept { return column.valid != nullptr; }

constexpr inline bool has_nulls(const gdf_column& column) noexcept
{
    return is_nullable(column) and column.null_count > 0;
}

namespace detail {

struct size_of_helper {
    template <typename T>
    constexpr int operator()() const noexcept { return sizeof(T); }
};

} // namespace detail

/**
 * @brief Returns the size in bytes of values of a column element type.
 */
constexpr inline std::size_t size_of(gdf_dtype element_type) {
    return type_dispatcher(element_type, detail::size_of_helper{});
}

/**
 * @brief Returns the size in bytes of each element of a column (a.k.a. the column's width)
 */
inline std::size_t byte_width(const gdf_column& col) noexcept { return size_of(col.dtype); }

// @TODO implement this using the type dispatcher
// std::size_t data_size_in_bytes(const gdf_column& col);

/**
 * @brief Ensure the input is in a valid state representing a proper column.
 * Specifically,
 * ensures all fields have valid (rather than junk, uninitialized or declared-invalid
 * values), and that they are consistent with each other.
 */
void validate(const gdf_column& column);
void validate(const gdf_column* column_ptr);

/**
 * @brief Ensures two (valid!) columns have the same type.
 *
 * @param validated_column_1 A column which would pass @ref validate() .
 * @param validated_column_2 A column which would pass @ref validate() .
 * @param ignore_extra_type_info For some column element types, a column carries some
 * qualifying information which applies to all elements (and thus not repeated for
 * each one). Generally, this information should not be ignored, so that for two columns
 * to have the same type, they must also share it. However, for potential practical
 * reasons (with this being a utility rather than an API function), we allow the extra
 * information to be ignored by setting this parameter to `true`.
 */
bool have_same_type(const gdf_column& validated_column_1, const gdf_column& validated_column_2, bool ignore_extra_type_info = false) noexcept;
bool have_same_type(const gdf_column* validated_column_ptr_1, const gdf_column* validated_column_ptr_2, bool ignore_extra_type_info = false) noexcept;

namespace detail {

bool extra_type_info_is_compatible(
    const gdf_dtype& common_dtype,
    const gdf_dtype_extra_info& lhs_extra_type_info,
    const gdf_dtype_extra_info& rhs_extra_type_info) noexcept;

} // namespace detail

} // namespace cudf

#endif // UTILITIES_COLUMN_UTILS_HPP_
