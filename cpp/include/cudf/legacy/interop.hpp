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

#include <cudf/types.h>
#include <cudf/types.hpp>

/**---------------------------------------------------------------------------*
 * @file interop.hpp
 * @brief Defines APIs for converting to/from "legacy" constructs such as
 * `gdf_column` and `gdf_dtype` into their corresponding new constructs such as
 * `column_view` and `data_type`.
 *
 * The purpose of these functions is to enable interoperability with APIs that
 * may not yet be updated to use the new constructs.
 *---------------------------------------------------------------------------**/

namespace cudf {
namespace legacy {

/**---------------------------------------------------------------------------*
 * @brief Converts a `gdf_dtype` to the corresponding `data_type`, if possible.
 *
 * @param dtype The `gdf_dtype` to convert
 * @return data_type The `data_type` corresponding to `dtype`.
 *---------------------------------------------------------------------------**/
data_type gdf_dtype_to_data_type(gdf_dtype dtype);

/**---------------------------------------------------------------------------*
 * @brief Converts a `data_type` to the corresponding `gdf_dtype`, if possible.
 *
 * @param type The `data_type` to convert
 * @return gdf_dtype The `gdf_dtype` corresponding to `type`.
 *---------------------------------------------------------------------------**/
gdf_dtype data_type_to_gdf_dtype(data_type type);

/**---------------------------------------------------------------------------*
 * @brief Constructs a `column_view` of the data referenced by a `gdf_column`.
 *
 * @param col The `gdf_column` to construct a view from
 * @return column_view A view of the same data contained in the `gdf_column`.
 *---------------------------------------------------------------------------**/
column_view gdf_column_to_view(gdf_column const& col);

/**---------------------------------------------------------------------------*
 * @brief Constructs a `mutable_column_view` of the data referenced by a
 * `gdf_column`.
 *
 * @param col The `gdf_column` to construct a view from
 * @return mutable_column_view A view of the same data contained in the
 * `gdf_column`.
 *---------------------------------------------------------------------------**/
mutable_column_view gdf_column_to_mutable_view(gdf_column* col);

/**---------------------------------------------------------------------------*
 * @brief Creates a `gdf_column` referencing the data contained in a
 * `mutable_column_view`.
 *
 * @note Conversion from a `column_view` to `gdf_column` is disallowed as it
 * would allow mutable access to the underlying data.
 *
 * @param view The view containing the data to wrap in a `gdf_column`
 * @return gdf_column The `gdf_column` referencing the data from `view`
 *---------------------------------------------------------------------------**/
gdf_column view_to_gdf_column(mutable_column_view view);
}  // namespace legacy
}  // namespace cudf