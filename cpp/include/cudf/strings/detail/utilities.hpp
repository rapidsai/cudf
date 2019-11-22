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
#include <cudf/column/column.hpp>
#include <rmm/thrust_rmm_allocator.h>

namespace cudf
{
namespace strings
{
namespace detail
{

/**
 * @brief Create a chars column to be a child of a strings column.
 * This will return the properly sized column to be filled in by the caller.
 *
 * @param strings_count Number of strings in the column.
 * @param null_count Number of null string entries in the column.
 * @param bytes Number of bytes for the chars column.
 * @param mr Memory resource to use.
 * @param stream Stream to use for any kernel calls.
 * @return The chars child column for a strings column.
 */
std::unique_ptr<column> create_chars_child_column( size_type strings_count,
    size_type null_count, size_type bytes,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @brief Create a strings column with no strings.
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
