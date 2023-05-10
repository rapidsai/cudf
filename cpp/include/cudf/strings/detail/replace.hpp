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

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief The type of algorithm to use for a replace operation.
 */
enum class replace_algorithm {
  AUTO,          ///< Automatically choose the algorithm based on heuristics
  ROW_PARALLEL,  ///< Row-level parallelism
  CHAR_PARALLEL  ///< Character-level parallelism
};

/**
 * @copydoc cudf::strings::replace(strings_column_view const&, string_scalar const&,
 * string_scalar const&, int32_t, rmm::mr::device_memory_resource*)
 *
 * @tparam    alg    Replacement algorithm to use
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
template <replace_algorithm alg = replace_algorithm::AUTO>
std::unique_ptr<column> replace(strings_column_view const& strings,
                                string_scalar const& target,
                                string_scalar const& repl,
                                int32_t maxrepl,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr);

/**
 * @copydoc cudf::strings::replace_slice(strings_column_view const&, string_scalar const&,
 * size_type. size_type, rmm::mr::device_memory_resource*)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> replace_slice(strings_column_view const& strings,
                                      string_scalar const& repl,
                                      size_type start,
                                      size_type stop,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr);

/**
 * @copydoc cudf::strings::replace(strings_column_view const&, strings_column_view const&,
 * strings_column_view const&, rmm::mr::device_memory_resource*)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> replace(strings_column_view const& strings,
                                strings_column_view const& targets,
                                strings_column_view const& repls,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr);

/**
 * @brief Replaces any null string entries with the given string.
 *
 * This returns a strings column with no null entries.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", nullptr, "goodbye"]
 * r = replace_nulls(s,"**")
 * r is now ["hello", "**", "goodbye"]
 * @endcode
 *
 * @param strings Strings column for this operation.
 * @param repl Replacement string for null entries. Default is empty string.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
std::unique_ptr<column> replace_nulls(strings_column_view const& strings,
                                      string_scalar const& repl,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace strings
}  // namespace cudf
