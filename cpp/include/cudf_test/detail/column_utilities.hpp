/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>

namespace cudf {
namespace test {
namespace detail {

/**
 * @brief Formats a column view as a string
 *
 * @param col The column view
 * @param delimiter The delimiter to put between strings
 * @param indent Indentation for all output. See detail::to_strings for detailed
 * explanation.
 */
std::string to_string(cudf::column_view const& col,
                      std::string const& delimiter,
                      std::string const& indent = "");

/**
 * @brief Formats a null mask as a string
 *
 * @param null_mask The null mask buffer
 * @param null_mask_size Size of the null mask (in rows)
 * @param indent Indentation for all output. See detail::to_strings for detailed
 * explanation.
 */
std::string to_string(std::vector<bitmask_type> const& null_mask,
                      size_type null_mask_size,
                      std::string const& indent = "");

/**
 * @brief Convert column values to a host vector of strings
 *
 * Supports indentation of all output.  For example, if the displayed output of your column
 * would be
 *
 * @code{.pseudo}
 * "1,2,3,4,5"
 * @endcode
 * and the `indent` parameter was "   ", that indentation would be prepended to
 * result in the output
 * @code{.pseudo}
 * "   1,2,3,4,5"
 * @endcode
 *
 * The can be useful for displaying complex types. An example use case would be for
 * displaying the nesting of a LIST type column (via recursion).
 *
 *  List<List<int>>:
 *  Length : 3
 *  Offsets : 0, 2, 5, 6
 *  Children :
 *     List<int>:
 *     Length : 6
 *     Offsets : 0, 2, 4, 7, 8, 9, 11
 *     Children :
 *        1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10
 *
 * @param col The column view
 * @param indent Indentation for all output
 */
std::vector<std::string> to_strings(cudf::column_view const& col, std::string const& indent = "");

}  // namespace detail
}  // namespace test
}  // namespace cudf
