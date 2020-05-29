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
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <vector>

namespace cudf {
namespace detail {
/**
 * @copydoc cudf::lower_bound
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> lower_bound(
  table_view const& t,
  table_view const& values,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t steam                  = 0);

/**
 * @copydoc cudf::upper_bound
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> upper_bound(
  table_view const& t,
  table_view const& values,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::contains(column_view const&, scalar const&,
 *                                       rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
bool contains(column_view const& col,
              scalar const& value,
              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
              cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::contains(column_view const&, column_view const&,
 *                                       rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> contains(
  column_view const& haystack,
  column_view const& needles,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

}  // namespace detail
}  // namespace cudf
