/*
<<<<<<< HEAD
 * Copyright (c) 2021, BAIDU CORPORATION.
=======
 * Copyright (c) 2021, NVIDIA CORPORATION.
>>>>>>> 58f395b15524309b36bcc1480eb4d186764df7dd
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

<<<<<<< HEAD
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

=======
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

>>>>>>> 58f395b15524309b36bcc1480eb4d186764df7dd
namespace cudf {
namespace strings {
namespace detail {

/**
<<<<<<< HEAD
 * @copydoc cudf::strings::json_to_array
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::column> json_to_array(
  cudf::strings_column_view const& col,
=======
 * @copydoc cudf::strings::get_json_object
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  cudf::string_scalar const& json_path,
>>>>>>> 58f395b15524309b36bcc1480eb4d186764df7dd
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace strings
}  // namespace cudf
