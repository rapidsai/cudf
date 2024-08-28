/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_split
 * @{
 * @file strings/split/partition.hpp
 * @brief Strings partition APIs
 */

/**
 * @brief Returns a set of 3 columns by splitting each string using the
 * specified delimiter.
 *
 * The number of rows in the output columns will be the same as the
 * input column. The first column will contain the first tokens of
 * each string as a result of the split. The second column will contain
 * the delimiter. The third column will contain the remaining characters
 * of each string after the delimiter.
 *
 * Any null string entries return corresponding null output columns.
 *
 * @code{.pseudo}
 * Example:
 * s = ["ab_cd","def_g_h"]
 * r = partition(s,"_")
 * r[0] is ["ab","def"]
 * r[1] is ["_","_"]
 * r[2] is ["cd","g_h"]
 * @endcode
 *
 * @param input Strings instance for this operation
 * @param delimiter UTF-8 encoded string indicating where to split each string.
 *        Default of empty string indicates split on whitespace.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return New table of strings columns
 */
std::unique_ptr<table> partition(
  strings_column_view const& input,
  string_scalar const& delimiter     = string_scalar(""),
  rmm::cuda_stream_view stream       = cudf::get_default_stream(),
  cudf::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a set of 3 columns by splitting each string using the
 * specified delimiter starting from the end of each string.
 *
 * The number of rows in the output columns will be the same as the
 * input column. The first column will contain the characters of
 * each string before the last delimiter found. The second column will contain
 * the delimiter. The third column will contain the remaining characters
 * of each string after the delimiter.
 *
 * Any null string entries return corresponding null output columns.
 *
 * @code{.pseudo}
 * Example:
 * s = ["ab_cd","def_g_h"]
 * r = rpartition(s,"_")
 * r[0] is ["ab","def_g"]
 * r[1] is ["_","_"]
 * r[2] is ["cd","h"]
 * @endcode
 *
 * @param input Strings instance for this operation
 * @param delimiter UTF-8 encoded string indicating where to split each string.
 *        Default of empty string indicates split on whitespace.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return New strings columns
 */
std::unique_ptr<table> rpartition(
  strings_column_view const& input,
  string_scalar const& delimiter     = string_scalar(""),
  rmm::cuda_stream_view stream       = cudf::get_default_stream(),
  cudf::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
