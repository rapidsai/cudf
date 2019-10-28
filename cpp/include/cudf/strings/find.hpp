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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/column/column.hpp>

namespace cudf
{
namespace strings
{

/**
 * @brief Returns a column of character position values where the target
 * string is first found in each string of the provided column.
 *
 * If the string is not found, -1 is returned for that row entry.
 *
 * The target string is searched within each string in the character
 * position range [start,stop). If the stop parameter is -1, then the
 * end of the string becomes the final position to include in the search.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @param strings Strings instance for this operation.
 * @param target Null-terminated, UTF-8 encoded host string to search for in each string.
 * @param start First character position to include in the search.
 * @param stop Last position (exclusive) to include in the search.
 *             Default of -1 will search to the end of the string.
 * @param mr Resource for allocating device memory.
 * @param stream Stream to use for any kernels in this function.
 * @return New INT32 column with character position values.
 */
std::unique_ptr<cudf::column> find( cudf::strings_column_view strings,
                                    const char* target,
                                    int32_t start=0, int32_t stop=-1,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                    cudaStream_t stream=0 );

/**
 * @brief Returns a column of character position values where the target
 * string is first found searching from the end of each string.
 *
 * If the string is not found, -1 is returned for that entry.
 *
 * The target string is searched within each string in the character
 * position range [start,stop). If the stop parameter is -1, then the
 * end of the string becomes the final position to include in the search.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @param strings Strings instance for this operation.
 * @param target Null-terminated, UTF-8 encoded host string to search for in each string.
 * @param start First position to include in the search.
 * @param stop Last position (exclusive) to include in the search.
 *             Default of -1 will search starting at the end of the string.
 * @param mr Resource for allocating device memory.
 * @param stream Stream to use for any kernels in this function.
 * @return New INT32 column with character position values.
 */
std::unique_ptr<cudf::column> rfind( cudf::strings_column_view strings,
                                     const char* target,
                                     int32_t start=0, int32_t stop=-1,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                     cudaStream_t stream=0 );
/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the target string was found within that string in the provided column.
 *
 * If a target string is not found, false is returned for that entry for that column.
 *
 * Any null string entries return corresponding null entries in the output columns.
 *
 * @param strings Strings instance for this operation.
 * @param target Null-terminated, UTF-8 encoded host string to search for in each string.
 * @param mr Resource for allocating device memory.
 * @param stream Stream to use for any kernels in this function.
 * @return New INT8 column.
 */
std::unique_ptr<cudf::column> contains( cudf::strings_column_view strings,
                                        const char* target,
                                        rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                        cudaStream_t stream=0 );

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the target string was found at the beginning of that string in the provided column.
 *
 * If a target string is not found at the beginning of the string, false is set for
 * that row entry in the output column.
 *
 * Any null string entries return corresponding null entries in the output columns.
 *
 * @param strings Strings instance for this operation.
 * @param target Null-terminated, UTF-8 encoded host string to search for in each string.
 * @param mr Resource for allocating device memory.
 * @param stream Stream to use for any kernels in this function.
 * @return New INT8 column.
 */
std::unique_ptr<cudf::column> starts_with( cudf::strings_column_view strings,
                                           const char* target,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                           cudaStream_t stream=0 );

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the target string was found at the end of that string in the provided column.
 *
 * If a target string is not found at the end of the string, false is set for
 * that row entry in the output column.
 *
 * Any null string entries return corresponding null entries in the output columns.
 *
 * @param strings Strings instance for this operation.
 * @param target Null-terminated, UTF-8 encoded host string to search for in each string.
 * @param mr Resource for allocating device memory.
 * @param stream Stream to use for any kernels in this function.
 * @return New INT8 column.
 */
std::unique_ptr<cudf::column> ends_with( cudf::strings_column_view strings,
                                         const char* target,
                                         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                         cudaStream_t stream=0 );


} // namespace strings
} // namespace cudf
