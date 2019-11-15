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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/column/column.hpp>

namespace cudf
{
namespace strings
{

/**
 * @brief Returns a new numeric column by parsing float values from each string
 * in the provided strings column.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * Only characters [0-9] plus a prefix '-' and '+' and decimal '.' are recognized.
 * Additionally, scientific notation is also supported (e.g. "-1.78e+5").
 *
 * @throw cudf::logic_error if output_type is not float type.
 *
 * @param strings Strings instance for this operation.
 * @param output_type Type of float numeric column to return.
 * @param mr Resource for allocating device memory.
 * @return New column with floats converted from strings.
 */
std::unique_ptr<column> to_floats( strings_column_view const& strings,
                                   data_type output_type,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a new strings column converting the float values from the
 * provided column into strings.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * For each float, a string is created in base-10 decimal.
 * Negative numbers will include a '-' prefix.
 * Numbers producing more than 10 significant digits will produce a string that
 * includes scientific notation (e.g. "-1.78e+15").
 *
 * @throw cudf::logic_error if floats column is not float type.
 *
 * @param column Numeric column to convert.
 * @param mr Resource for allocating device memory.
 * @param stream Stream to use for any kernels in this function.
 * @return New strings column with floats as strings.
 */
std::unique_ptr<column> from_floats( column_view const& floats,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace strings
} // namespace cudf
