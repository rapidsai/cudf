/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cstdint>

namespace cudf {
namespace strings {
namespace detail {
// Type for the character flags table.
using character_flags_table_type = std::uint8_t;

/**
 * @brief Returns pointer to device memory that contains the static
 * characters flags table. On first call, this will copy the table into
 * device memory and is guaranteed to be thread-safe.
 *
 * This table is used to check the type of character like
 * alphanumeric, decimal, etc.
 *
 * @return Device memory pointer to character flags table.
 */
const character_flags_table_type* get_character_flags_table();

// Type for the character cases table.
using character_cases_table_type = uint16_t;

/**
 * @brief Returns pointer to device memory that contains the static
 * characters case table. On first call, this will copy the table into
 * device memory and is guaranteed to be thread-safe.
 *
 * This table is used to map upper and lower case characters with
 * their counterpart.
 *
 * @return Device memory pointer to character cases table.
 */
const character_cases_table_type* get_character_cases_table();

/**
 * @brief Returns pointer to device memory that contains the special
 * case mapping table. On first call, this will copy the table into
 * device memory and is guaranteed to be thread-safe.
 *
 * This table is used to handle special case character mappings that
 * don't trivially work with the normal character cases table.
 *
 * @return Device memory pointer to the special case mapping table
 */
const struct special_case_mapping* get_special_case_mapping_table();

}  // namespace detail
}  // namespace strings
}  // namespace cudf
