
/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "../detail/strip_utils.cuh"
#include "dstring.cuh"

namespace cudf {
namespace strings {
namespace udf {

/**
 * @brief Strip characters from the beginning and/or end of the given string.
 *
 * The `d_to_strip` is interpretted as an array of characters to be removed.
 * If `d_to_strip` is an empty string, whitespace characters are stripped.
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{" aba ", 5};
 * auto d_to_strip = cudf::string_view{}; // empty string
 * auto result = strip(d_str, d_to_strip);
 * // result is "aba"
 * d_to_strip = cudf::string_view{" a", 2}; // space and 'a'
 * result = strip(d_str, d_to_strip);
 * // result is "b" ('a' or ' ' removed from the ends)
 * @endcode
 *
 * @param d_str String to strip characters from
 * @param d_to_strip Characters to remove
 * @param stype From where to strip the characters;
 *              Default `BOTH` indicates stripping characters from the
 *              beginning and the end of the input string `d_str`
 * @return New string with characters removed
 */
__device__ dstring strip(cudf::string_view const d_str,
                         cudf::string_view const d_to_strip,
                         strip_type stype = strip_type::BOTH)
{
  return dstring{cudf::strings::detail::strip(d_str, d_to_strip, stype)};
}

/**
 * @brief Strip characters from the beginning and/or end of the given string.
 *
 * The `d_to_strip` is interpretted as an array of characters to be removed.
 * If `d_to_strip` is an empty string, whitespace characters are stripped.
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{" abaa a ", 8};
 * auto result = strip(d_str, " ", 1);
 * // result is "abaa a"
 * result = strip(d_str, "a ", 2); // 'a' and space
 * // result is "b" ('a' or ' ' removed from the ends)
 * @endcode
 *
 * @param d_str String to strip characters from
 * @param d_to_strip Array of characters encoded in UTF-8 to remove
 * @param bytes Number of bytes to read from `d_to_strip`
 * @param stype From where to strip the characters;
 *              Default `BOTH` indicates stripping characters from the
 *              beginning and the end of the input string `d_str`
 * @return New string with characters removed
 */
__device__ dstring strip(cudf::string_view const d_str,
                         char const* d_to_strip,
                         cudf::size_type bytes,
                         strip_type stype = strip_type::BOTH)
{
  auto const sv = cudf::string_view{d_to_strip, bytes};
  return strip(d_str, sv, stype);
}

/**
 * @brief Strip characters from the beginning and/or end of the given string.
 *
 * The `d_to_strip` is interpretted as an array of characters to be removed.
 * If `d_to_strip` is an empty string, whitespace characters are stripped.
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{" abca a ", 8};
 * auto result = strip(d_str);
 * // result is "abca a"
 * result = strip(d_str, "a b"); // 'a', 'b', and space
 * // result is "c" ('a', ' ', or 'b' is removed from the ends)
 * @endcode
 *
 * @param d_str String to strip characters from
 * @param d_to_strip Null-terminated array of characters encoded in UTF-8
 *                   to remove
 * @param stype From where to strip the characters;
 *              Default `BOTH` indicates stripping characters from the
 *              beginning and the end of the input string `d_str`
 * @return New string with characters removed
 */
__device__ dstring strip(cudf::string_view const d_str,
                         char const* d_to_strip = "",
                         strip_type stype       = strip_type::BOTH)
{
  return strip(d_str, d_to_strip, detail::bytes_in_null_terminated_string(d_to_strip), stype);
}

/**
 * @brief Strip characters from the beginning of the given string.
 *
 * The `d_to_strip` is interpretted as an array of characters to be removed.
 * If `d_to_strip` is an empty string, whitespace characters are stripped.
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{" aba ", 5};
 * auto d_to_strip = cudf::string_view{}; // empty string
 * auto result = lstrip(d_str, d_to_strip);
 * // result is "aba "
 * d_to_strip = cudf::string_view{"a ", 2}; // space and 'a'
 * result = lstrip(d_str, d_to_strip);
 * // result is "ba " ('a' or ' ' removed from the beginning)
 * @endcode
 *
 * @param d_str String to strip characters from
 * @param d_to_strip Characters to remove
 * @return New string with characters removed
 */
__device__ dstring lstrip(cudf::string_view const d_str, cudf::string_view d_to_strip)
{
  return strip(d_str, d_to_strip, strip_type::LEFT);
}

/**
 * @brief Strip characters from the beginning of the given string.
 *
 * The `d_to_strip` is interpretted as an array of characters to be removed.
 * If `d_to_strip` is an empty string, whitespace characters are stripped.
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{" abaa a ", 8};
 * auto result = lstrip(d_str, " ", 1);
 * // result is "abaa a "
 * result = lstrip(d_str, "a ", 2); // 'a' and space
 * // result is "baa a " ('a' or ' ' removed from the beginning)
 * @endcode
 *
 * @param d_str String to strip characters from
 * @param d_to_strip Array of characters encoded in UTF-8 to remove
 * @param bytes Number of bytes to read from `d_to_strip`
 * @return New string with characters removed
 */
__device__ dstring lstrip(cudf::string_view const d_str,
                          char const* d_to_strip,
                          cudf::size_type bytes)
{
  auto const sv = cudf::string_view{d_to_strip, bytes};
  return strip(d_str, sv, strip_type::LEFT);
}

/**
 * @brief Strip characters from the beginning of the given string.
 *
 * The `d_to_strip` is interpretted as an array of characters to be removed.
 * If `d_to_strip` is an empty string, whitespace characters are stripped.
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{" abca a ", 8};
 * auto result = lstrip(d_str);
 * // result is "abca a "
 * result = lstrip(d_str, "a b"); // 'a', 'b', and space
 * // result is "ca a " ('a', ' ', or 'b' removed from the beginning)
 * @endcode
 *
 * @param d_str String to strip characters from
 * @param d_to_strip Null-terminated array of characters encoded in UTF-8
 *                   to remove
 * @return New string with characters removed
 */
__device__ dstring lstrip(cudf::string_view const d_str, char const* d_to_strip = "")
{
  return strip(
    d_str, d_to_strip, detail::bytes_in_null_terminated_string(d_to_strip), strip_type::LEFT);
}

/**
 * @brief Strip characters from the end of the given string.
 *
 * The `d_to_strip` is interpretted as an array of characters to be removed.
 * If `d_to_strip` is an empty string, whitespace characters are stripped.
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{" aba ", 5};
 * auto d_to_strip = cudf::string_view{}; // empty string
 * auto result = rstrip(d_str, d_to_strip);
 * // result is " aba"
 * d_to_strip = cudf::string_view{" a", 2}; // space and 'a'
 * result = rstrip(d_str, d_to_strip);
 * // result is " ab" ('a' or ' ' removed from the end)
 * @endcode
 *
 * @param d_str String to strip characters from
 * @param d_to_strip Characters to remove
 * @return New string with characters removed
 */
__device__ dstring rstrip(cudf::string_view const d_str, cudf::string_view d_to_strip)
{
  return strip(d_str, d_to_strip, strip_type::RIGHT);
}

/**
 * @brief Strip characters from the end of the given string.
 *
 * The `d_to_strip` is interpretted as an array of characters to be removed.
 * If `d_to_strip` is an empty string, whitespace characters are stripped.
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{" abaa a ", 8};
 * auto result = rstrip(d_str, " ", 1);
 * // result is " abaa a"
 * result = rstrip(d_str, " a", 2); // 'a' and space
 * // result is " ab" ('a' or ' ' removed from the end)
 * @endcode
 *
 * @param d_str String to strip characters from
 * @param d_to_strip Array of characters encoded in UTF-8 to remove
 * @param bytes Number of bytes to read from `d_to_strip`
 * @return New string with characters removed
 */
__device__ dstring rstrip(cudf::string_view const d_str,
                          char const* d_to_strip,
                          cudf::size_type bytes)
{
  auto const sv = cudf::string_view{d_to_strip, bytes};
  return strip(d_str, sv, strip_type::RIGHT);
}

/**
 * @brief Strip characters from the end of the given string.
 *
 * The `d_to_strip` is interpretted as an array of characters to be removed.
 * If `d_to_strip` is an empty string, whitespace characters are stripped.
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{" acba a ", 8};
 * auto result = rstrip(d_str);
 * // result is " acba a"
 * result = rstrip(d_str, "a b"); // 'a', 'b', and space
 * // result is " ac" ('a', ' ', or 'b' removed from the end)
 * @endcode
 *
 * @param d_str String to strip characters from
 * @param d_to_strip Null-terminated array of characters encoded in UTF-8
 *                   to remove
 * @return New string with characters removed
 */
__device__ dstring rstrip(cudf::string_view const d_str, char const* d_to_strip = "")
{
  return strip(
    d_str, d_to_strip, detail::bytes_in_null_terminated_string(d_to_strip), strip_type::RIGHT);
}

}  // namespace udf
}  // namespace strings
}  // namespace cudf
