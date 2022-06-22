/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf/tdigest/tdigest_column_view.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::quantile()
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> quantile(
  column_view const& input,
  std::vector<double> const& q,
  interpolation interp                = interpolation::LINEAR,
  column_view const& ordered_indices  = {},
  bool exact                          = true,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::quantiles()
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
  rmm::cuda_stream_view stream                   = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::percentile_approx(tdigest_column_view const&, column_view const&,
 * rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> percentile_approx(
  tdigest::tdigest_column_view const& input,
  column_view const& percentiles,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace cudf
