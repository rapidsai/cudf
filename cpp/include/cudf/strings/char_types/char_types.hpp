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
 * @brief Character type values.
 * These types can be or'd to check for any combination of types.
 */
enum string_character_types {
    DECIMAL  =  1,
    NUMERIC  =  2,
    DIGIT    =  4,
    ALPHA    =  8,
    ALPHANUM = 15,
    SPACE    = 16,
    UPPER    = 32,
    LOWER    = 64
};

/**
 * @brief Returns a boolean column identifying strings entries in which all
 * characters are of the type specified.
 *
 * The output row entry will be set to false if the corresponding string element
 * is empty or has at least one character not of the specified type. If all
 * characters fit the type then true is set in that output row entry.
 *
 * Any null string results in a null entry for that row in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param types The character types to check in each string.
 * @param mr Resource for allocating device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<cudf::column> all_characters_of_type( strings_column_view strings,
                                                      string_character_types types,
                                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace strings
} // namespace cudf
