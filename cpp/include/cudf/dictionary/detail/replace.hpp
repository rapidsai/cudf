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
#include <cudf/scalar/scalar.hpp>

namespace cudf {
namespace dictionary {
namespace detail {

/**
 * @brief Create a new dictionary column by replacing nulls with values
 * from a second dictionary.
 *
 * @throw cudf::logic_error if the keys type of both dictionaries do not match.
 * @throw cudf::logic_error if the column sizes do not match.
 *
 * @param input Column with nulls to replace.
 * @param replacement Column with values to use for replacing.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New dictionary column with null rows replaced.
 */
std::unique_ptr<column> replace_nulls(
  dictionary_column_view const& input,
  dictionary_column_view const& replacement,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @brief Create a new dictionary column by replacing nulls with a
 * specified scalar.
 *
 * @throw cudf::logic_error if the keys type does not match the replacement type.
 *
 * @param input Column with nulls to replace.
 * @param replacement Value to use for replacing.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New dictionary column with null rows replaced.
 */
std::unique_ptr<column> replace_nulls(
  dictionary_column_view const& input,
  scalar const& replacement,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
