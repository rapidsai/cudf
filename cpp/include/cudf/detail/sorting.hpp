/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <vector>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> sorted_order(
  table_view input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_default_resource(),
  cudaStream_t stream                            = 0);

/**
 * @copydoc cudf::stable_sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> stable_sorted_order(
  table_view input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_default_resource(),
  cudaStream_t stream                            = 0);

/**
 * @copydoc cudf::sort_by_key
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> sort_by_key(
  table_view const& values,
  table_view const& keys,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_default_resource(),
  cudaStream_t stream                            = 0);

}  // namespace detail
}  // namespace cudf
