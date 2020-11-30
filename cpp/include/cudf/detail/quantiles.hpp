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

#include <cudf/quantiles.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

/** @copydoc cudf::quantile()
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> quantile(
  column_view const& input,
  std::vector<double> const& q,
  interpolation interp                = interpolation::LINEAR,
  column_view const& ordered_indices  = {},
  bool exact                          = true,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @copydoc cudf::quantiles()
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> quantiles(
  table_view const& input,
  std::vector<double> const& q,
  interpolation interp                           = interpolation::NEAREST,
  cudf::sorted is_input_sorted                   = sorted::NO,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace cudf
