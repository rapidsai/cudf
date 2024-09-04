/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace spark_rapids_jni {

/**
 * @brief Type of instruction in a JSON path.
 */
enum class path_instruction_type : int8_t { WILDCARD, INDEX, NAMED };
/// path instruction type
// enum class path_instruction_type : int8_t { WILDCARD, INDEX, NAMED };
using json_path_t = std::vector<std::tuple<path_instruction_type, std::string, int32_t>>;

std::vector<std::unique_ptr<cudf::column>> get_json_object_multiple_paths2(
  cudf::column_view const& input,
  std::vector<json_path_t> const& json_paths,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni
}  // namespace CUDF_EXPORT cudf
