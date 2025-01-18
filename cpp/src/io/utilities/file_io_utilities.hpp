/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cudf/io/datasource.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <string>

namespace cudf {
namespace io {
namespace detail {

// Call before any cuFile API calls to ensure the CUDA context is initialized.
void force_init_cuda_context();

}  // namespace detail
}  // namespace io
}  // namespace cudf
