/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @copydoc cudf::get_json_object
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  std::string const& json_path,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace strings
}  // namespace cudf
