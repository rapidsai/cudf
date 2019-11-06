/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include "cudf/cudf.h"
#include "cudf/types.hpp"
#include "cudf/column/column.hpp"

namespace cudf {

/**
 * @brief Copies from a column of floating-point elements and replaces `-NaN` and `-0.0` with `+NaN` and `+0.0`, respectively.
 *
 * Converts floating point values from @p input using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in] Column of floating-point elements to copy and normalize
 * @param[in] device_memory_resource allocator for allocating output data 
 *
 * @returns new column with the modified data
 */
std::unique_ptr<column> normalize_nans_and_zeros( column_view const& input,                                                                                                    
                                                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

/**
 * @brief Modifies a column of floating-point elements to replace all `-NaN` and `-0.0` with `+NaN` and `+0.0`, respectively.
 * 
 * Converts floating point values from @p in_out using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in, out] Column of floating-point elements to normalize
 */
void normalize_nans_and_zeros(mutable_column_view& in_out);

} // namespace cudf

