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

#include <cuda_runtime.h>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf
{
namespace strings
{
namespace detail
{

/**
 * @brief Creates a temporary string_view object from a host string.
 *
 * @param[in] str Null-terminated, encoded string in CPU memory.
 * @param[in] stream Stream to execute any device code against.
 * @return Device object pointer.
 */
std::unique_ptr<string_view, std::function<void(string_view*)>>
    string_from_host( const char* str, cudaStream_t stream=0 );

/**
 * @brief Creates a string_view vector from a strings column.
 * This is useful for doing some intermediate vector operations.
 *
 * @param strings Strings column instance.
 * @param stream Stream to execute any device code against.
 * @return string_view vector
 */
rmm::device_vector<string_view> create_string_vector_from_column(
    cudf::strings_column_view strings,
    cudaStream_t stream=0 );

/**
 * @brief Creates an offsets column from a string_view vector.
 * This can be used to recreate the offsets child of a new
 * strings column from an intermediate strings vector.
 *
 * @param strings Strings column
 * @param stream Stream to execute any device code against.
 * @param mr Memory resource to use.
 * @return Offsets column
 */
std::unique_ptr<cudf::column> offsets_from_string_vector(
    const rmm::device_vector<string_view>& strings,
    cudaStream_t stream=0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**
 * @brief Creates a chars column from a string_view vector.
 * This can be used to recreate the chars child of a new
 * strings column from an intermediate strings vector.
 *
 * @param strings Strings vector
 * @param d_offsets Offsets vector for placing strings into column's memory.
 * @param null_count Number of null strings.
 * @param stream Stream to execute any device code against.
 * @param mr Memory resource to use.
 * @return chars column
 */
std::unique_ptr<cudf::column> chars_from_string_vector(
    const rmm::device_vector<string_view>& strings,
    const int32_t* d_offsets, cudf::size_type null_count,
    cudaStream_t stream=0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**
 * @brief Create a chars column to be a child of a strings column.
 * This will return the properly sized column to be filled in by the caller.
 *
 * @param strings_count Number of strings in the column.
 * @param null_count Number of null string entries in the column.
 * @param total_bytes Number of bytes for the chars column.
 * @param mr Memory resource to use.
 * @param stream Stream to use for any kernel calls.
 * @return chars child column for strings column
 */
std::unique_ptr<column> create_chars_child_column( cudf::size_type strings_count,
    cudf::size_type null_count, cudf::size_type total_bytes,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @brief Create a strings column with no data for response to operations
 * invoked with an empty column.
 *
 * @param mr Memory resource to use.
 * @param stream Stream to use for any kernel calls.
 * @return Empty strings column
 */
std::unique_ptr<column> make_empty_strings_column(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

} // namespace detail
} // namespace strings
} // namespace cudf
