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
 * @brief Creates a strings array from a strings column.
 * This is useful for doing some intermediate array operations.
 *
 * @param strings Strings instance.
 * @param stream Stream to execute any device code against.
 * @return Strings array
 */
rmm::device_vector<string_view> create_string_array_from_column(
    cudf::strings_column_view strings,
    cudaStream_t stream=0 );

/**
 * @brief Creates an offsets column from a strings array.
 * This can be used to recreate the offsets child of a new
 * strings column from an intermediate strings array.
 *
 * @param strings Strings array
 * @param stream Stream to execute any device code against.
 * @param mr Memory resource to use.
 * @return Offsets column
 */
std::unique_ptr<cudf::column> offsets_from_string_array(
    const rmm::device_vector<string_view>& strings,
    cudaStream_t stream=0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**
 * @brief Creates a chars column from a strings array.
 * This can be used to recreate the chars child of a new
 * strings column from an intermediate strings array.
 *
 * @param strings Strings array
 * @param d_offsets Offsets array for placing strings into column's memory.
 * @param null_count Number of null strings.
 * @param stream Stream to execute any device code against.
 * @param mr Memory resource to use.
 * @return chars column
 */
std::unique_ptr<cudf::column> chars_from_string_array(
    const rmm::device_vector<string_view>& strings,
    const int32_t* d_offsets, cudf::size_type null_count,
    cudaStream_t stream=0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

} // namespace detail
} // namespace strings
} // namespace cudf
