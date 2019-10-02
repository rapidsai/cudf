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
#include <cudf/strings/string_view.cuh>
#include <cudf/column/column_view.hpp>

#include <cstring>

namespace cudf
{
namespace strings
{
namespace detail
{

/**
 * @brief This utility will copy the argument string's data into
 * the provided buffer.
 *
 * @param buffer Device buffer to copy to.
 * @param d_string String to copy.
 * @return Points to the end of the buffer after the copy.
 */
__device__ inline char* copy_string( char* buffer, const string_view& d_string )
{
    memcpy( buffer, d_string.data(), d_string.size_bytes() );
    return buffer + d_string.size_bytes();
}

} // namespace detail
} // namespace strings
} // namespace cudf
