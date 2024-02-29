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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/optional.h>

namespace cudf {

/**
 * @addtogroup json_object
 * @{
 * @file
 */

/**
 * @brief Settings for `get_json_object()`.
 */
class get_json_object_options {
  // allow single quotes to represent strings in JSON
  bool allow_single_quotes = false;

  // individual string values are returned with quotes stripped.
  bool strip_quotes_from_single_strings = true;

  // Whether to return nulls when an object does not contain the requested field.
  bool missing_fields_as_nulls = false;

  // Whether allow unescaped control characters in JSON Strings.
  bool allow_unescaped_control_chars = false;

  // Define the maximum JSON String length.
  int max_string_len = -1;

  // Define the maximum JSON number length.
  int max_num_len = -1;

  // Whether allow tailing useless sub-string
  bool allow_tailing_sub_string = false;

 public:
  /**
   * @brief Default constructor.
   */
  explicit get_json_object_options() = default;

  /**
   * @brief Returns true/false depending on whether single-quotes for representing strings
   * are allowed.
   *
   * @return true if single-quotes are allowed, false otherwise.
   */
  [[nodiscard]] CUDF_HOST_DEVICE inline bool get_allow_single_quotes() const
  {
    return allow_single_quotes;
  }

  /**
   * @brief Returns true/false depending on whether individually returned string values have
   * their quotes stripped.
   *
   * When set to true, if the return value for a given row is an individual string
   * (not an object, or an array of strings), strip the quotes from the string and return only the
   * contents of the string itself.  Example:
   *
   * @code{.pseudo}
   *
   * With strip_quotes_from_single_strings OFF:
   * Input  = {"a" : "b"}
   * Query  = $.a
   * Output = "b"
   *
   * With strip_quotes_from_single_strings ON:
   * Input  = {"a" : "b"}
   * Query  = $.a
   * Output = b
   *
   * @endcode
   *
   * @return true if individually returned string values have their quotes stripped.
   */
  [[nodiscard]] CUDF_HOST_DEVICE inline bool get_strip_quotes_from_single_strings() const
  {
    return strip_quotes_from_single_strings;
  }

  /**
   * @brief Whether a field not contained by an object is to be interpreted as null.
   *
   * When set to true, if an object is queried for a field it does not contain, a null is returned.
   *
   * @code{.pseudo}
   *
   * With missing_fields_as_nulls OFF:
   * Input  = {"a" : [{"x": "1", "y": "2"}, {"x": "3"}]}
   * Query  = $.a[*].y
   * Output = ["2"]
   *
   * With missing_fields_as_nulls ON:
   * Input  = {"a" : [{"x": "1", "y": "2"}, {"x": "3"}]}
   * Query  = $.a[*].y
   * Output = ["2", null]
   *
   * @endcode
   *
   * @return true if missing fields are interpreted as null.
   */
  [[nodiscard]] CUDF_HOST_DEVICE inline bool get_missing_fields_as_nulls() const
  {
    return missing_fields_as_nulls;
  }

  /**
   * @brief Returns true/false depending on whether unescaped characters for representing strings
   * are allowed.
   *
   * Unescaped control characters are ASCII characters with value less than 32,
   * including tab and line feed characters.
   *
   * If true, JSON is not conventional format.
   * e.g., how to represent carriage return and newline characters:
   *   if true, allow "\n\r" two control characters without escape directly
   *   if false, "\n\r" are not allowed, should use escape characters: "\\n\\r"
   *
   * @return true if unescaped characters are allowed, false otherwise.
   */
  [[nodiscard]] CUDF_HOST_DEVICE inline bool get_allow_unescaped_control_chars() const
  {
    return allow_unescaped_control_chars;
  }

  /**
   * @brief Returns maximum JSON String length, negative or zero means no limitation.
   *
   * By default, maximum JSON String length is negative one, means no limitation.
   * e.g.: The length of String "\\n" is 1, JSON parser does not count escape characters.
   *
   * @return integer value of allowed maximum JSON String length
   */
  [[nodiscard]] CUDF_HOST_DEVICE int get_max_string_len() const { return max_string_len; }

  /**
   * @brief Returns maximum JSON number length, negative or zero means no limitation.
   *
   * By default, maximum JSON number length is negative one, means no limitation.
   *
   * e.g.: The length of number -123.45e-67 is 7. if maximum JSON number length is 6,
   * then this number is a invalid number.
   *
   * @return integer value of allowed maximum JSON number length
   */
  [[nodiscard]] CUDF_HOST_DEVICE int get_max_num_len() const { return max_num_len; }

  /**
   * @brief Returns whether allow tailing useless sub-string in JSON.
   *
   * If true, e.g., the following invalid JSON is allowed, because prefix {'k' : 'v'} is valid.
   *   {'k' : 'v'}_extra_tail_sub_string
   *
   * @return true if alow tailing useless sub-string, false otherwise.
   */
  [[nodiscard]] CUDF_HOST_DEVICE int get_allow_tailing_sub_string() const
  {
    return allow_tailing_sub_string;
  }

  /**
   * @brief Set whether single-quotes for strings are allowed.
   *
   * @param _allow_single_quotes bool indicating desired behavior.
   */
  void set_allow_single_quotes(bool _allow_single_quotes)
  {
    allow_single_quotes = _allow_single_quotes;
  }

  /**
   * @brief Set whether individually returned string values have their quotes stripped.
   *
   * @param _strip_quotes_from_single_strings bool indicating desired behavior.
   */
  void set_strip_quotes_from_single_strings(bool _strip_quotes_from_single_strings)
  {
    strip_quotes_from_single_strings = _strip_quotes_from_single_strings;
  }

  /**
   * @brief Set whether missing fields are interpreted as null.
   *
   * @param _missing_fields_as_nulls bool indicating desired behavior.
   */
  void set_missing_fields_as_nulls(bool _missing_fields_as_nulls)
  {
    missing_fields_as_nulls = _missing_fields_as_nulls;
  }

  /**
   * @brief Set whether alow unescaped control characters.
   *
   * @param _allow_unescaped_control_chars bool indicating desired behavior.
   */
  void set_allow_unescaped_control_chars(bool _allow_unescaped_control_chars)
  {
    allow_unescaped_control_chars = _allow_unescaped_control_chars;
  }

  /**
   * @brief Set maximum JSON String length.
   *
   * @param _max_string_len integer indicating desired behavior.
   */
  void set_max_string_len(int _max_string_len) { max_string_len = _max_string_len; }

  /**
   * @brief Set maximum JSON number length.
   *
   * @param _max_num_len integer indicating desired behavior.
   */
  void set_max_num_len(int _max_num_len) { max_num_len = _max_num_len; }

  /**
   * @brief Set whether allow tailing useless sub-string.
   *
   * @param _allow_tailing_sub_string bool indicating desired behavior.
   */
  void set_allow_tailing_sub_string(bool _allow_tailing_sub_string)
  {
    allow_tailing_sub_string = _allow_tailing_sub_string;
  }
};

/**
 * @brief Apply a JSONPath string to all rows in an input strings column.
 *
 * Applies a JSONPath string to an incoming strings column where each row in the column
 * is a valid json string.  The output is returned by row as a strings column.
 *
 * https://tools.ietf.org/id/draft-goessner-dispatch-jsonpath-00.html
 * Implements only the operators: $ . [] *
 *
 * @throw std::invalid_argument if provided an invalid operator or an empty name
 *
 * @param col The input strings column. Each row must contain a valid json string
 * @param json_path The JSONPath string to be applied to each row
 * @param options Options for controlling the behavior of the function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Resource for allocating device memory
 * @return New strings column containing the retrieved json object strings
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  cudf::string_scalar const& json_path,
  get_json_object_options options     = get_json_object_options{},
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace cudf
