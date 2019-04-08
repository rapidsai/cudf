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

#include "column_utils.hpp"

namespace cudf {

gdf_error validate(const gdf_column& column)
{
    if (column.data == nullptr and column.size > 0)          { return GDF_INVALID_API_CALL; }
    if (column.dtype == GDF_invalid)                         { return GDF_DTYPE_MISMATCH;   }
    if (column.dtype >= N_GDF_TYPES)                         { return GDF_DTYPE_MISMATCH;   }
    if (not is_nullable(column) and (column.null_count > 0)) { return GDF_VALIDITY_MISSING; }
    if (column.null_count > column.size)                     { return GDF_INVALID_API_CALL; }
    return GDF_SUCCESS;
}

gdf_error validate(const gdf_column* column_ptr)
{
    if (column_ptr == nullptr) { return GDF_INVALID_API_CALL; }
    return validate(*column_ptr);
}

bool have_matching_types(const gdf_column& validated_column_1, const gdf_column& validated_column_2)
{
    if (validated_column_1.dtype != validated_column_2.dtype) { return GDF_DTYPE_MISMATCH; }
    if (detail::logical_xor(is_nullable(validated_column_1), is_nullable(validated_column_2))) { return GDF_VALIDITY_MISSING; }
    return true;
}

bool have_matching_types(const gdf_column* validated_column_ptr_1, const gdf_column* validated_column_ptr_2)
{
    return have_matching_types(*validated_column_ptr_1, *validated_column_ptr_2);
}

namespace detail {

bool extra_type_info_is_compatible(
    const gdf_dtype& common_dtype,
    const gdf_dtype_extra_info& lhs_extra_type_info,
    const gdf_dtype_extra_info& rhs_extra_type_info) noexcept
{
    switch(common_dtype) {
    case GDF_CATEGORY:  return lhs_extra_type_info.category == rhs_extra_type_info.category;
    case GDF_TIMESTAMP: return lhs_extra_type_info.time_unit == rhs_extra_type_info.time_unit;
    default:            return true;
    }
}

} // namespace detail

bool are_strictly_type_compatible(
    const gdf_column& lhs,
    const gdf_column& rhs) noexcept
{
    if (lhs.dtype != rhs.dtype) { return false; }
    auto common_dtype = lhs.dtype;
    return detail::extra_type_info_is_compatible(common_dtype, lhs.dtype_info, rhs.dtype_info);
}


} // namespace cudf
