/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace nvtext {
namespace detail {
/**
 * @copydoc nvtext::tokenize(strings_column_view const&,string_scalar
 * const&,rmm::mr::device_memory_resource*)
 *
 * @param strings Strings column tokenize.
 * @param delimiter UTF-8 characters used to separate each string into tokens.
 *                  The default of empty string will separate tokens using whitespace.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings columns of tokens.
 */
std::unique_ptr<cudf::column> tokenize(
  cudf::strings_column_view const& strings,
  cudf::string_scalar const& delimiter = cudf::string_scalar{""},
  rmm::cuda_stream_view stream         = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_current_device_resource());

/**
 * @copydoc nvtext::tokenize(strings_column_view const&,strings_column_view
 * const&,rmm::mr::device_memory_resource*)
 *
 * @param strings Strings column to tokenize.
 * @param delimiters Strings used to separate individual strings into tokens.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings columns of tokens.
 */
std::unique_ptr<cudf::column> tokenize(
  cudf::strings_column_view const& strings,
  cudf::strings_column_view const& delimiters,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc nvtext::count_tokens(strings_column_view const&, string_scalar
 * const&,rmm::mr::device_memory_resource*)
 *
 * @param strings Strings column to use for this operation.
 * @param delimiter Strings used to separate each string into tokens.
 *                  The default of empty string will separate tokens using whitespace.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New INT32 column of token counts.
 */
std::unique_ptr<cudf::column> count_tokens(
  cudf::strings_column_view const& strings,
  cudf::string_scalar const& delimiter = cudf::string_scalar{""},
  rmm::cuda_stream_view stream         = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_current_device_resource());

/**
 * @copydoc nvtext::count_tokens(strings_column_view const&,strings_column_view
 * const&,rmm::mr::device_memory_resource*)
 *
 * @param strings Strings column to use for this operation.
 * @param delimiters Strings used to separate each string into tokens.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New INT32 column of token counts.
 */
std::unique_ptr<cudf::column> count_tokens(
  cudf::strings_column_view const& strings,
  cudf::strings_column_view const& delimiters,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace nvtext
