/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace structs::detail {
/**
 * @brief Scan function for struct column type
 *
 * Called by cudf::scan() with only min and max aggregates.
 *
 * @tparam Op Either DeviceMin or DeviceMax operations
 *
 * @param input Input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New struct column
 */
template <typename Op>
std::unique_ptr<column> scan_inclusive(column_view const& input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

}  // namespace structs::detail
}  // namespace CUDF_EXPORT cudf
