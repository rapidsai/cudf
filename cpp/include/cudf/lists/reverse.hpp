/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace lists {
/**
 * @addtogroup lists_modify
 * @{
 * @file
 */

/**
 * @brief Reverse the element order within each list of the input column.
 *
 * Any null input row will result in a corresponding null row in the output column.
 *
 * @code{.pseudo}
 * Example:
 * s = [ [1, 2, 3], [], null, [4, 5, null] ]
 * r = reverse(s)
 * r is now [ [3, 2, 1], [], null, [null, 5, 4] ]
 * @endcode
 *
 * @param input Lists column for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New lists column with reversed lists
 */
std::unique_ptr<column> reverse(
  lists_column_view const& input,
  rmm::cuda_stream_view stream       = cudf::get_default_stream(),
  cudf::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group

}  // namespace lists
}  // namespace CUDF_EXPORT cudf
