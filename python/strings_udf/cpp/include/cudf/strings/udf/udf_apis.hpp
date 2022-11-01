/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <rmm/device_buffer.hpp>

#include <memory>

namespace cudf {
namespace strings {
namespace udf {

/**
 * @brief Return a cudf::string_view array for the given strings column
 *
 * @param input Strings column to convert to a string_view array.
 * @throw cudf::logic_error if input is not a strings column.
 */
std::unique_ptr<rmm::device_buffer> to_string_view_array(cudf::column_view const input);

}  // namespace udf
}  // namespace strings
}  // namespace cudf
