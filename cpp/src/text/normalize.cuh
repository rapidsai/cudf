/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "text/subword/detail/cp_data.h"

namespace nvtext {
namespace detail {

/**
 * @brief Bit used to filter out invalid code points.
 *
 * When normalizing characters to code point values, if this bit is set,
 * the code point should be filtered out before returning from the normalizer.
 */
constexpr uint32_t FILTER_BIT = 22;

/**
 * @brief Retrieve new code point from metadata value.
 *
 * @param metadata Value from the codepoint_metadata table.
 * @return The replacement character if appropriate.
 */
__device__ constexpr uint32_t get_first_cp(uint32_t metadata) { return metadata & NEW_CP_MASK; }

/**
 * @brief Retrieve token category from the metadata value.
 *
 * Category values are 0-5:
 * 0 - character should be padded
 * 1 - pad character if lower-case
 * 2 - character should be removed
 * 3 - remove character if lower-case
 * 4 - whitespace character -- always replace
 * 5 - uncategorized
 *
 * @param metadata Value from the codepoint_metadata table.
 * @return Category value.
 */
__device__ constexpr uint32_t extract_token_cat(uint32_t metadata)
{
  return (metadata >> TOKEN_CAT_SHIFT) & TOKEN_CAT_MASK;
}

/**
 * @brief Return true if category of metadata value specifies the character should be replaced.
 */
__device__ constexpr bool should_remove_cp(uint32_t metadata, bool lower_case)
{
  auto const cat = extract_token_cat(metadata);
  return (cat == TOKEN_CAT_REMOVE_CHAR) || (lower_case && (cat == TOKEN_CAT_REMOVE_CHAR_IF_LOWER));
}

/**
 * @brief Return true if category of metadata value specifies the character should be padded.
 */
__device__ constexpr bool should_add_spaces(uint32_t metadata, bool lower_case)
{
  auto const cat = extract_token_cat(metadata);
  return (cat == TOKEN_CAT_ADD_SPACE) || (lower_case && (cat == TOKEN_CAT_ADD_SPACE_IF_LOWER));
}

/**
 * @brief Return true if category of metadata value specifies the character should be replaced.
 */
__device__ constexpr bool always_replace(uint32_t metadata)
{
  return extract_token_cat(metadata) == TOKEN_CAT_ALWAYS_REPLACE;
}

/**
 * @brief Returns true if metadata value includes a multi-character transform bit equal to 1.
 */
__device__ constexpr bool is_multi_char_transform(uint32_t metadata)
{
  return (metadata >> MULTICHAR_SHIFT) & MULTICHAR_MASK;
}

/**
 * @brief Returns true if the byte passed in could be a valid head byte for
 * a utf8 character. That is, not binary `10xxxxxx`
 */
__device__ constexpr bool is_head_byte(unsigned char utf8_byte) { return (utf8_byte >> 6) != 2; }

}  // namespace detail
}  // namespace nvtext
