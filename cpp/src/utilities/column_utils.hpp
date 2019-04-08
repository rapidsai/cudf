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

#include <utilities/type_dispatcher.hpp>

#include <cudf.h>

#include <algorithm>

namespace cudf {

namespace detail {

// This should go up into more general utility code...
constexpr inline bool logical_xor(bool x, bool y) { return (x and (not y)) or ((not x) and y) ; }

} // namespace detail

template<typename T>
T* get_data(const gdf_column& column) { return static_cast<T*>(column.data); }

template<typename T>
T* get_data(const gdf_column* column) { return get_data<T>(*column); }

constexpr inline bool is_an_integer(gdf_dtype element_type)
{
    return
        element_type == GDF_INT8  or
        element_type == GDF_INT16 or
        element_type == GDF_INT32 or
        element_type == GDF_INT64;
}

constexpr inline bool is_integral(const gdf_column& column) { return is_an_integer(column.dtype); }
constexpr inline bool is_integral(const gdf_column* column) { return is_integral(*column); }

constexpr bool is_nullable(const gdf_column& column) { return column.valid != nullptr; }

namespace detail {

struct size_of_helper {
    template <typename T>
    constexpr int operator()() const { return sizeof(T); }
};

} // namespace detail

constexpr std::size_t inline size_of(gdf_dtype element_type) {
    return type_dispatcher(element_type, detail::size_of_helper{});
}

inline std::size_t width(const gdf_column& col) { return size_of(col.dtype); }

// @TODO implement this using the type dispatcher
// std::size_t data_size_in_bytes(const gdf_column& col);

gdf_error validate(const gdf_column& column);
gdf_error validate(const gdf_column* column_ptr);

bool have_matching_types(const gdf_column& validated_column_1, const gdf_column& validated_column_2);
bool have_matching_types(const gdf_column* validated_column_ptr_1, const gdf_column* validated_column_ptr_2);

namespace detail {

bool extra_type_info_is_compatible(
    const gdf_dtype& common_dtype,
    const gdf_dtype_extra_info& lhs_extra_type_info,
    const gdf_dtype_extra_info& rhs_extra_type_info) noexcept;

} // namespace detail

bool are_strictly_type_compatible(
    const gdf_column& lhs,
    const gdf_column& rhs) noexcept;

} // namespace cudf

#endif // UTILITIES_COLUMN_UTILS_HPP_
