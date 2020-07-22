/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_split
 * @{
 */

/**
 * @brief Returns a list of columns by splitting each string using the
 * specified delimiter.
 *
 * The number of rows in the output columns will be the same as the
 * input column. The first column will contain the first tokens of
 * each string as a result of the split. Subsequent columns contain
 * the next token strings. Null entries are added for a row where
 * split results have been exhausted. The total number of columns
 * will equal the maximum number of splits encountered on any string
 * in the input column.
 *
 * Any null string entries return corresponding null output columns.
 *
 * @param strings_column Strings instance for this operation.
 * @param delimiter UTF-8 encoded string indicating the split points in each string.
 *        Default of empty string indicates split on whitespace.
 * @param maxsplit Maximum number of splits to perform.
 *        Default of -1 indicates all possible splits on each string.
 * @param mr Device memory resource used to allocate the returned table's device memory.
 * @return New table of strings columns.
 */
std::unique_ptr<table> split(strings_column_view const& strings_column,
                             string_scalar const& delimiter      = string_scalar(""),
                             size_type maxsplit                  = -1,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a list of columns by splitting each string using the
 * specified delimiter starting from the end of each string.
 *
 * The number of rows in the output columns will be the same as the
 * input column. The first column will contain the first tokens encountered
 * in each string as a result of the split. Subsequent columns contain
 * the next token strings. Null entries are added for a row where
 * split results have been exhausted. The total number of columns
 * will equal the maximum number of splits encountered on any string
 * in the input column.
 *
 * Any null string entries return corresponding null output columns.
 *
 * @param strings_column Strings instance for this operation.
 * @param delimiter UTF-8 encoded string indicating the split points in each string.
 *        Default of empty string indicates split on whitespace.
 * @param maxsplit Maximum number of splits to perform.
 *        Default of -1 indicates all possible splits on each string.
 * @param mr Device memory resource used to allocate the returned table's device memory.
 * @return New strings columns.
 */
std::unique_ptr<table> rsplit(
  strings_column_view const& strings_column,
  string_scalar const& delimiter      = string_scalar(""),
  size_type maxsplit                  = -1,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Splits individual strings elements into a list of strings.
 *
 * Each element generates an array of strings that are stored in an output
 * lists column.
 *
 * The number of elements in the output column will be the same as the number of
 * elements in the input column. Each individual list item will contain the
 * new strings for that row. The resulting number of strings in each row can vary
 * from 0 to `maxsplit + 1`.
 *
 * The `delimiter` is searched within each string from beginning to end
 * and splitting stops when either `maxsplit` or the end of the string is reached.
 *
 * If a delimiter is not whitespace and occurs adjacent to another delimiter,
 * an empty string is produced for that split occurrence. Likewise, a non-whitespace
 * delimiter produces an empty string if it appears at the beginning or the end
 * of a string.
 *
 * @code{.pseudo}
 * s = ["a_bc_def_g", "a__bc", "_ab_cd", "ab_cd_"]
 * s1 = split_record(s, "_")
 * s1 is a lists column of strings:
 *     [ ["a", "bc", "def", "g"],
 *       ["a", "", "bc"],
 *       ["", "ab", "cd"],
 *       ["ab", "cd", ""] ]
 * s2 = split_record(s, "_", 1)
 * s2 is a lists column of strings:
 *     [ ["a", "bc_def_g"],
 *       ["a", "_bc"],
 *       ["", "ab_cd"],
 *       ["ab", "cd_"] ]
 * @endcode
 *
 * A whitespace delimiter produces no empty strings.
 * @code{.pseudo}
 * s = ["a bc def", "a  bc", " ab cd", "ab cd "]
 * s1 = split_record(s, "")
 * s1 is a lists column of strings:
 *     [ ["a", "bc", "def"],
 *       ["a", "bc"],
 *       ["ab", "cd"],
 *       ["ab", "cd"] ]
 * s2 = split_record(s, "", 1)
 * s2 is a lists column of strings:
 *     [ ["a", "bc def"],
 *       ["a", "bc"],
 *       ["ab", "cd"],
 *       ["ab", "cd "] ]
 * @endcode
 *
 * A null string element will result in a null list item for that row.
 *
 * @throw cudf:logic_error if `delimiter` is invalid.
 *
 * @param strings A column of string elements to be splitted.
 * @param delimiter The string to identify split points in each string.
 *        Default of empty string indicates split on whitespace.
 * @param maxsplit Maximum number of splits to perform.
 *        Default of -1 indicates all possible splits on each string.
 * @param mr Device memory resource used to allocate the returned result's device memory.
 * @return Lists column of strings
 *         Each vector of the lists column holds splits from a single row
 *         element of the input column.
 */
std::unique_ptr<column> split_record(
  strings_column_view const& strings,
  string_scalar const& delimiter      = string_scalar(""),
  size_type maxsplit                  = -1,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Splits individual strings elements into a list of strings starting
 * from the end of each string.
 *
 * Each element generates an array of strings that are stored in an output
 * lists column.
 *
 * The number of elements in the output column will be the same as the number of
 * elements in the input column. Each individual list item will contain the
 * new strings for that row. The resulting number of strings in each row can vary
 * from 0 to `maxsplit + 1`.
 *
 * The `delimiter` is searched from end to beginning within each string
 * and splitting stops when either `maxsplit` or the beginning of the string
 * is reached.
 *
 * If a delimiter is not whitespace and occurs adjacent to another delimiter,
 * an empty string is produced for that split occurrence. Likewise, a non-whitespace
 * delimiter produces an empty string if it appears at the beginning or the end
 * of a string.
 *
 * Note that `rsplit_record` and `split_record` produce equivalent results for
 * the default `maxsplit` value.
 *
 * @code{.pseudo}
 * s = ["a_bc_def_g", "a__bc", "_ab_cd", "ab_cd_"]
 * s1 = rsplit_record(s, "_")
 * s1 is a lists column of strings:
 *     [ ["a", "bc", "def", "g"],
 *       ["a", "", "bc"],
 *       ["", "ab", "cd"],
 *       ["ab", "cd", ""] ]
 * s2 = rsplit_record(s, "_", 1)
 * s2 is a lists column of strings:
 *     [ ["a_bc_def", "g"],
 *       ["a_", "bc"],
 *       ["_ab", "cd"],
 *       ["ab_cd", ""] ]
 * @endcode
 *
 * A whitespace delimiter produces no empty strings.
 * @code{.pseudo}
 * s = ["a bc def", "a  bc", " ab cd", "ab cd "]
 * s1 = rsplit_record(s, "")
 * s1 is a lists column of strings:
 *     [ ["a", "bc", "def"],
 *       ["a", "bc"],
 *       ["ab", "cd"],
 *       ["ab", "cd"] ]
 * s2 = rsplit_record(s, "", 1)
 * s2 is a lists column of strings:
 *     [ ["a bc", "def"],
 *       ["a", "bc"],
 *       [" ab", "cd"],
 *       ["ab", "cd"] ]
 * @endcode
 *
 * A null string element will result in a null list item for that row.
 *
 * @throw cudf:logic_error if `delimiter` is invalid.
 *
 * @param strings A column of string elements to be splitted.
 * @param delimiter The string to identify split points in each string.
 *        Default of empty string indicates split on whitespace.
 * @param maxsplit Maximum number of splits to perform.
 *        Default of -1 indicates all possible splits on each string.
 * @param mr Device memory resource used to allocate the returned result's device memory.
 * @return Lists column of strings
 *         Each vector of the lists column holds splits from a single row
 *         element of the input column.
 */
std::unique_ptr<column> rsplit_record(
  strings_column_view const& strings,
  string_scalar const& delimiter      = string_scalar(""),
  size_type maxsplit                  = -1,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
