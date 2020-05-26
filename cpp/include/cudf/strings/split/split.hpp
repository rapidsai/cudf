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
 * @brief The result(s) of a `contiguous_(r)split_record`
 *
 * Each column_view resulting from a split operation performed by
 * contiguous_split_record will be returned wrapped in a
 * `contiguous_split_record_result`. The column data addresses stored in the
 * column_view objects are not owned by top level cudf::column objects. The
 * backing memory is instead owned by the `all_data` field and in one contiguous
 * block.
 *
 * The user is responsible for assuring that the `column_views` or any derived
 * objects do not outlive the memory owned by `all_data`
 */
struct contiguous_split_record_result {
  std::vector<column_view> column_views;
  std::unique_ptr<rmm::device_buffer> all_data;
};

/**
 * @brief Splits each element of the input column to a column of tokens storing
 * the resulting columns in a single contiguous block of memory.
 *
 * This function splits each element in the input column to a column of tokens.
 * The number of columns in the output vector will be the same as the number of
 * elements in the input column. The column length will coincide with the
 * number of tokens; the resulting columns wrapped in the returned object may
 * have different sizes.
 *
 * Splitting a null string element will result in an empty output column.
 *
 * @throws cudf:logic_error if `delimiter` is invalid.
 *
 * @param strings A column of string elements to be splitted.
 * @param delimiter UTF-8 encoded string indicating the split points in each
 *        string.
 *        Default of empty string indicates split on whitespace.
 * @param maxsplit Maximum number of splits to perform.
 *        Default of -1 indicates all possible splits on each string.
 * @param mr Device memory resource used to allocate the returned result's device memory.
 * @return contiguous_split_record_result New vector of strings column_view
 *         objects
 *         (each column_view element of the vector holds splits from a string
 *         element of the input column).
 */
contiguous_split_record_result contiguous_split_record(
  strings_column_view const& strings,
  string_scalar const& delimiter      = string_scalar(""),
  size_type maxsplit                  = -1,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Splits each element of the input column from the end to a column of
 * tokens storing the resulting columns in a single contiguous block of memory.
 *
 * This function splits each element in the input column to a column of tokens.
 * The number of columns in the output vector will be the same as the number of
 * elements in the input column. The column length will coincide with the
 * number of tokens; the resulting columns wrapped in the returned object may
 * have different sizes.
 *
 * Splitting a null string element will result in an empty output column.
 *
 * @throws cudf:logic_error if `delimiter` is invalid.
 *
 * @param strings A column of string elements to be splitted.
 * @param delimiter UTF-8 encoded string indicating the split points in each
 *        string.
 *        Default of empty string indicates split on whitespace.
 * @param maxsplit Maximum number of splits to perform.
 *        Default of -1 indicates all possible splits on each string.
 * @param mr Device memory resource used to allocate the returned result's device memory.
 * @return contiguous_split_record_result New vector of strings column_view
 *         objects
 *         (each column_view element of the vector holds splits from a string
 *         element of the input column).
 */
contiguous_split_record_result contiguous_rsplit_record(
  strings_column_view const& strings,
  string_scalar const& delimiter      = string_scalar(""),
  size_type maxsplit                  = -1,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
