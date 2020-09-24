/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/dictionary/dictionary_column_view.hpp>

namespace cudf {
namespace dictionary {
namespace detail {
/**
 * @brief Returns a single column by vertically concatenating the given vector of
 * dictionary columns.
 *
 * @throw cudf::logic_error if `columns.size()==0`
 * @throw cudf::logic_error if dictionary column keys are not all the same type.
 *
 * @param columns Vector of dictionary columns to concatenate.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New column with concatenated results.
 */
std::unique_ptr<column> concatenate(
  std::vector<column_view> const& columns,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
