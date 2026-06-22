/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "udf_string.cuh"

#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace strings {
namespace udf {

/**
 * @brief Global variables for character-type flags and case conversion
 */
struct chars_tables {
  cudf::strings::detail::character_flags_table_type* flags_table;
  cudf::strings::detail::character_cases_table_type* cases_table;
  struct cudf::strings::detail::special_case_mapping* special_case_mapping_table;
};

namespace detail {

/**
 * @brief Utility for converting a single character
 *
 * There are special cases where the conversion may result in multiple characters.
 *
 * @param tables The char tables required for conversion
 * @param result String to append the converted character
 * @param code_point The code-point of the character to convert
 * @param flag The char-type flag of the character to convert
 */
__device__ inline void convert_char(chars_tables const tables,
                                    udf_string& result,
                                    uint32_t code_point,
                                    uint8_t flag)
{
  if (!cudf::strings::detail::IS_SPECIAL(flag)) {
    result.append(cudf::strings::detail::codepoint_to_utf8(tables.cases_table[code_point]));
    return;
  }

  // handle special case
  auto const map =
    tables
      .special_case_mapping_table[cudf::strings::detail::get_special_case_hash_index(code_point)];
  auto const output_count =
    cudf::strings::detail::IS_LOWER(flag) ? map.num_upper_chars : map.num_lower_chars;
  auto const* output_chars = cudf::strings::detail::IS_LOWER(flag) ? map.upper : map.lower;
  for (uint16_t idx = 0; idx < output_count; idx++) {
    result.append(cudf::strings::detail::codepoint_to_utf8(output_chars[idx]));
  }
}

/**
 * @brief Converts the given string to either upper or lower case
 *
 * @param tables The char tables required for conversion
 * @param d_str Input string to convert
 * @param case_flag Identifies upper/lower case conversion
 * @return New string containing the converted characters
 */
__device__ inline udf_string convert_case(
  chars_tables const tables,
  string_view d_str,
  cudf::strings::detail::character_flags_table_type case_flag)
{
  udf_string result;
  for (auto const chr : d_str) {
    auto const code_point = cudf::strings::detail::utf8_to_codepoint(chr);
    auto const flag       = code_point <= 0x00FFFF ? tables.flags_table[code_point] : 0;

    if ((flag & case_flag) || (cudf::strings::detail::IS_SPECIAL(flag) &&
                               !cudf::strings::detail::IS_UPPER_OR_LOWER(flag))) {
      convert_char(tables, result, code_point, flag);
    } else {
      result.append(chr);
    }
  }

  return result;
}

/**
 * @brief Utility for capitalize and title functions
 *
 * @tparam CapitalizeNextFn returns true if the next candidate character should be capitalized
 * @param tables The char tables required for conversion
 * @param d_str Input string to convert
 * @param next_fn Function for next character capitalized
 * @return New string containing the converted characters
 */
template <typename CapitalizeNextFn>
__device__ inline udf_string capitalize(chars_tables const tables,
                                        string_view d_str,
                                        CapitalizeNextFn next_fn)
{
  udf_string result;
  bool capitalize = true;
  for (auto const chr : d_str) {
    auto const code_point = cudf::strings::detail::utf8_to_codepoint(chr);
    auto const flag       = code_point <= 0x00FFFF ? tables.flags_table[code_point] : 0;
    auto const change_case =
      capitalize ? cudf::strings::detail::IS_LOWER(flag) : cudf::strings::detail::IS_UPPER(flag);
    if (change_case) {
      detail::convert_char(tables, result, code_point, flag);
    } else {
      result.append(chr);
    }
    capitalize = next_fn(flag);
  }
  return result;
}
}  // namespace detail

/**
 * @brief Converts the given string to lower case
 *
 * @param tables The char tables required for conversion
 * @param d_str Input string to convert
 * @return New string containing the converted characters
 */
__device__ inline udf_string to_lower(chars_tables const tables, string_view d_str)
{
  cudf::strings::detail::character_flags_table_type case_flag = cudf::strings::detail::IS_UPPER(
    cudf::strings::detail::ALL_FLAGS);  // convert only upper case characters
  return detail::convert_case(tables, d_str, case_flag);
}

/**
 * @brief Converts the given string to upper case
 *
 * @param tables The char tables required for conversion
 * @param d_str Input string to convert
 * @return New string containing the converted characters
 */
__device__ inline udf_string to_upper(chars_tables const tables, string_view d_str)
{
  cudf::strings::detail::character_flags_table_type case_flag = cudf::strings::detail::IS_LOWER(
    cudf::strings::detail::ALL_FLAGS);  // convert only lower case characters
  return detail::convert_case(tables, d_str, case_flag);
}

/**
 * @brief Converts the given string to lower/upper case
 *
 * All lower case characters are converted to upper case and
 * all upper case characters are converted to lower case.
 *
 * @param tables The char tables required for conversion
 * @param d_str Input string to convert
 * @return New string containing the converted characters
 */
__device__ inline udf_string swap_case(chars_tables const tables, string_view d_str)
{
  cudf::strings::detail::character_flags_table_type case_flag =
    cudf::strings::detail::IS_LOWER(cudf::strings::detail::ALL_FLAGS) |
    cudf::strings::detail::IS_UPPER(cudf::strings::detail::ALL_FLAGS);
  return detail::convert_case(tables, d_str, case_flag);
}

/**
 * @brief Capitalize the first character of the given string
 *
 * @param tables The char tables required for conversion
 * @param d_str Input string to convert
 * @return New string containing the converted characters
 */
__device__ inline udf_string capitalize(chars_tables const tables, string_view d_str)
{
  auto next_fn = [](cudf::strings::detail::character_flags_table_type) -> bool { return false; };
  return detail::capitalize(tables, d_str, next_fn);
}

/**
 * @brief Converts the given string to title case
 *
 * The first character after a non-character is converted to upper case.
 * All other characters are converted to lower case.
 *
 * @param tables The char tables required for conversion
 * @param d_str Input string to convert
 * @return New string containing the converted characters
 */
__device__ inline udf_string title(chars_tables const tables, string_view d_str)
{
  auto next_fn = [](cudf::strings::detail::character_flags_table_type flag) -> bool {
    return !cudf::strings::detail::IS_ALPHA(flag);
  };
  return detail::capitalize(tables, d_str, next_fn);
}

}  // namespace udf
}  // namespace strings
}  // namespace cudf
