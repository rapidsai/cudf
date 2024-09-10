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

#include <cudf/timezone.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @copydoc cudf::make_timezone_transition_table(std::optional<std::string_view>, std::string_view,
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> make_timezone_transition_table(
  std::optional<std::string_view> tzif_dir,
  std::string_view timezone_name,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
