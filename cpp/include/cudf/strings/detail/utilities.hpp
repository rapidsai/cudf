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

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Create an offsets column to be a child of a strings column
 *
 * This will return the properly typed column to be filled in by the caller
 * given the number of bytes to address.
 *
 * @param chars_bytes Number of bytes for the chars in the strings column
 * @param count Number of elements for the offsets column.
 *              This is the number of rows in the parent strings column +1.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The offsets child column for a strings column
 */
std::unique_ptr<column> create_offsets_child_column(int64_t chars_bytes,
                                                    size_type count,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr);

/**
 * @brief Creates a string_view vector from a strings column.
 *
 * @param strings Strings column instance.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned vector's device memory.
 * @return Device vector of string_views
 */
rmm::device_uvector<string_view> create_string_vector_from_column(
  cudf::strings_column_view const strings,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Return the threshold size for a strings column to use int64 offsets
 *
 * A computed size above this threshold should using int64 offsets, otherwise
 * int32 offsets. By default this function will return std::numeric_limits<int32_t>::max().
 * This value can be overridden at runtime using the environment variable
 * LIBCUDF_LARGE_STRINGS_THRESHOLD.
 *
 * @return size in bytes
 */
int64_t get_offset64_threshold();

/**
 * @brief Checks if large strings is enabled
 *
 * This checks the setting in the environment variable LIBCUDF_LARGE_STRINGS_ENABLED.
 *
 * @return true if large strings are supported
 */
bool is_large_strings_enabled();

/**
 * @brief Return a normalized offset value from a strings offsets column
 *
 * The maximum value returned is `std::numeric_limits<int32_t>::max()`.
 *
 * @throw std::invalid_argument if `offsets` is neither INT32 nor INT64
 *
 * @param offsets Input column of type INT32 or INT64
 * @param index Row value to retrieve
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Value at `offsets[index]`
 */
int64_t get_offset_value(cudf::column_view const& offsets,
                         size_type index,
                         rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace strings
}  // namespace cudf
