/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cudf/detail/interop.hpp>

namespace cudf {
namespace detail {

// unique_ptr because that is what AllocateBuffer returns
std::unique_ptr<arrow::Buffer> allocate_arrow_buffer(int64_t const size, arrow::MemoryPool* ar_mr);

// shared_ptr because that is what AllocateBitmap returns
std::shared_ptr<arrow::Buffer> allocate_arrow_bitmap(int64_t const size, arrow::MemoryPool* ar_mr);

}  // namespace detail
}  // namespace cudf
