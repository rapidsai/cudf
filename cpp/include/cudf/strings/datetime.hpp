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
 * @brief Returns a new datetime column converting a string column into
 * timestamps using the provided format string.
 *
 * The format must include strptime format specifiers though only the
 * following are supported: %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z
 *
 * No checking is done for invalid formats or invalid timestamp units.
 *
 * Any null string entry will result in a null entry in the output column.
 *
 * @throw cudf::logic_error if timestamp_type is not a timestamp type.
 *
 * @param strings Strings instance for this operation.
 * @param timestamps Timestamp values to convert.
 * @param output_type The timestamp type used for the values in timestamps.
 * @param format The null-terminated CPU string specifying string output format.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New datetime column.
 */
std::unique_ptr<cudf::column> to_timestamps( strings_column_view strings,
                                             data_type timestamp_type,
                                             std::string format,
                                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                             cudaStream_t stream = 0 );


/**
 * @brief Returns a new strings column converting a datetime column into
 * strings using the provided format string.
 *
 * The format must include strftime format specifiers though only the
 * following are supported: %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z
 *
 * No checking is done for invalid formats or invalid timestamp units.
 *
 * Any null input entry will result in a corresponding null entry in the output column.
 *
 * @throw cudf::logic_error if timestamps column is not a timestamp type.
 *
 * @param timestamps Timestamp values to convert.
 * @param format The null-terminated CPU string specifying string output format.
 *        Default format is "%Y-%m-%dT%H:%M:%SZ".
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New strings column with formatted timestamps.
 */
std::unique_ptr<cudf::column> from_timestamps( column_view timestamps,
                                               std::string format = "%Y-%m-%dT%H:%M:%SZ",
                                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                               cudaStream_t stream = 0 );


} // namespace strings
} // namespace cudf
