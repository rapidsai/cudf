/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf/utilities/export.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace CUDF_EXPORT cudf {
namespace strings {

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
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

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

}  // namespace strings
}  // namespace CUDF_EXPORT cudf
