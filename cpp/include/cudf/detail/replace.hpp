/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <memory>

// Forward declaration

namespace cudf {
namespace detail {
/**
 * @copydoc cudf::experimental::replace_nulls(column_view const&, column_view const&,
 * rmm::mr::device_memory_resource*)
 *
 * @param[in] stream Optional stream in which to perform allocations
 */
std::unique_ptr<column> replace_nulls(
  column_view const& input,
  cudf::column_view const& replacement,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::experimental::replace_nulls(column_view const&, scalar const&,
 * rmm::mr::device_memory_resource*)
 *
 * @param[in] stream Optional stream in which to perform allocations
 */
std::unique_ptr<column> replace_nulls(
  column_view const& input,
  scalar const& replacement,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::experimental::find_and_replace_all
 *
 * @param stream Optional CUDA stream to use for operations
 */
std::unique_ptr<column> find_and_replace_all(
  column_view const& input_col,
  column_view const& values_to_replace,
  column_view const& replacement_values,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);
}  // namespace detail
}  // namespace cudf
