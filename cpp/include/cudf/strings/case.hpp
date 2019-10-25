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
 * @brief Returns a column of strings converting the strings in the provided
 * column to lowercase characters.
 *
 * Only uppercase characters are converted. All other characters are copied.
 * Case conversion may result in strings that are longer or shorter than the
 * original string in bytes.
 *
 * Any null entries create null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @return New column of strings with characters converted.
 */
std::unique_ptr<cudf::column> to_lower( strings_column_view strings,
                                        rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());


/**
 * @brief Returns a column of strings converting the strings in the provided
 * column to uppercase characters.
 *
 * Only lowercase characters are converted. All other characters are copied.
 * Case conversion may result in strings that are longer or shorter than the
 * original string in bytes.
 *
 * Any null entries create null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @return New column of strings with characters converted.
 */
std::unique_ptr<cudf::column> to_upper( strings_column_view strings,
                                        rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a column of strings converting the strings in the provided
 * column to uppercase characters if they are lowercase and lowercase if they
 * are uppercase.
 *
 * Only upper or lower case characters are converted. All other characters are copied.
 * Case conversion may result in strings that are longer or shorter than the
 * original string in bytes.
 *
 * Any null entries create null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @return New column of strings with characters converted.
 */
std::unique_ptr<cudf::column> swapcase( strings_column_view strings,
                                        rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace strings
} // namespace cudf
