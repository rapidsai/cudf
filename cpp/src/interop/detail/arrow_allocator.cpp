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

#include <cudf/detail/interop.hpp>

namespace cudf {
namespace detail {

std::unique_ptr<arrow::Buffer> allocate_arrow_buffer(const int64_t size, arrow::MemoryPool* ar_mr)
{
  /*
  nvcc 11.0 generates Internal Compiler Error during codegen when arrow::AllocateBuffer
  and `ValueOrDie` are used inside a CUDA compilation unit.

  To work around this issue we compile an allocation shim in C++ and use
  that from our cuda sources
  */
  auto result = arrow::AllocateBuffer(size, ar_mr);
  CUDF_EXPECTS(result.ok(), "Failed to allocate Arrow buffer");
  return std::move(result).ValueOrDie();
}

std::shared_ptr<arrow::Buffer> allocate_arrow_bitmap(const int64_t size, arrow::MemoryPool* ar_mr)
{
  /*
  nvcc 11.0 generates Internal Compiler Error during codegen when arrow::AllocateBuffer
  and `ValueOrDie` are used inside a CUDA compilation unit.

  To work around this issue we compile an allocation shim in C++ and use
  that from our cuda sources
  */
  auto result = arrow::AllocateBitmap(size, ar_mr);
  CUDF_EXPECTS(result.ok(), "Failed to allocate Arrow bitmap");
  return std::move(result).ValueOrDie();
}

}  // namespace detail
}  // namespace cudf
