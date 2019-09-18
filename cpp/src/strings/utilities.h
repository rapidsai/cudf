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
#include <cudf/strings/strings_column_handler.hpp>
#include <cudf/strings/string_view.cuh>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf 
{
namespace strings
{

/**
 * @brief Creates a temporary string_view object from a host string.
 * The host string is copied into device memory and the object
 * pointer can be used in device code.
 *
 * @param[in] str Null-terminated, encoded string in CPU memory.
 * @param[in] stream Stream to execute any device code against.
 * @return Device object pointer.
 */
std::unique_ptr<cudf::string_view, std::function<void(cudf::string_view*)>>
    string_from_host( const char* str, cudaStream_t stream=0 );

/**
 * 
 */
rmm::device_buffer create_string_array_from_column(
    strings_column_handler strings,
    cudaStream_t stream = (cudaStream_t)0 );

} // namespace strings
} // namespace cudf