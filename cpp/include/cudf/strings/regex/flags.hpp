/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

/**
 * @addtogroup strings_regex
 * @{
 */

/**
 * @brief Regex flags.
 *
 * These types can be or'd to combine them.
 * The values are chosen to leave room for future flags
 * and to match the Python flag values.
 */
enum regex_flags : uint32_t {
  DEFAULT     = 0,    ///< default
  MULTILINE   = 8,    ///< the '^' and '$' honor new-line characters
  DOTALL      = 16,   ///< the '.' matching includes new-line characters
  ASCII       = 256,  ///< use only ASCII when matching built-in character classes
  EXT_NEWLINE = 512   ///< new-line matches extended characters
};

/**
 * @brief Returns true if the given flags contain MULTILINE.
 *
 * @param f Regex flags to check
 * @return true if `f` includes MULTILINE
 */
constexpr bool is_multiline(regex_flags const f)
{
  return (f & regex_flags::MULTILINE) == regex_flags::MULTILINE;
}

/**
 * @brief Returns true if the given flags contain DOTALL.
 *
 * @param f Regex flags to check
 * @return true if `f` includes DOTALL
 */
constexpr bool is_dotall(regex_flags const f)
{
  return (f & regex_flags::DOTALL) == regex_flags::DOTALL;
}

/**
 * @brief Returns true if the given flags contain ASCII.
 *
 * @param f Regex flags to check
 * @return true if `f` includes ASCII
 */
constexpr bool is_ascii(regex_flags const f)
{
  return (f & regex_flags::ASCII) == regex_flags::ASCII;
}

/**
 * @brief Returns true if the given flags contain EXT_NEWLINE
 *
 * @param f Regex flags to check
 * @return true if `f` includes EXT_NEWLINE
 */
constexpr bool is_ext_newline(regex_flags const f)
{
  return (f & regex_flags::EXT_NEWLINE) == regex_flags::EXT_NEWLINE;
}

/**
 * @brief Capture groups setting
 *
 * For processing a regex pattern containing capture groups.
 * These can be used to optimize the generated regex instructions
 * where the capture groups do not require extracting the groups.
 */
enum class capture_groups : uint32_t {
  EXTRACT,     ///< Capture groups processed normally for extract
  NON_CAPTURE  ///< Convert all capture groups to non-capture groups
};

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
