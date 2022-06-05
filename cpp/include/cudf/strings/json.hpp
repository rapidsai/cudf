/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <thrust/optional.h>

namespace cudf {
namespace strings {

/**
 * @addtogroup strings_json
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
 * @param col The input strings column. Each row must contain a valid json string
 * @param json_path The JSONPath string to be applied to each row
 * @param options Options for controlling the behavior of the function
 * @param mr Resource for allocating device memory.
 * @return New strings column containing the retrieved json object strings
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  cudf::string_scalar const& json_path,
  get_json_object_options options     = get_json_object_options{},
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
