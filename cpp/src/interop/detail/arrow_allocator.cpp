/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <sys/mman.h>
#include <unistd.h>

#include <memory>

namespace cudf {
namespace detail {

/*
  Enable Transparent Huge Pages (THP) for large (>4MB) allocations.
  `buf` is returned untouched.
  Enabling THP can improve performance of device-host memory transfers
  significantly, see <https://github.com/rapidsai/cudf/pull/13914>.
*/
template <typename T>
T enable_hugepage(T&& buf)
{
  if (buf->size() < (1u << 22u)) {  // Smaller than 4 MB
    return std::move(buf);
  }

#ifdef MADV_HUGEPAGE
  const auto pagesize = sysconf(_SC_PAGESIZE);
  void* addr          = const_cast<uint8_t*>(buf->data());
  if (addr == nullptr) { return std::move(buf); }
  auto length{static_cast<std::size_t>(buf->size())};
  if (std::align(pagesize, pagesize, addr, length)) {
    // Intentionally not checking for errors that may be returned by older kernel versions;
    // optimistically tries enabling huge pages.
    madvise(addr, length, MADV_HUGEPAGE);
  }
#endif
  return std::move(buf);
}

std::unique_ptr<arrow::Buffer> allocate_arrow_buffer(int64_t const size, arrow::MemoryPool* ar_mr)
{
  /*
  nvcc 11.0 generates Internal Compiler Error during codegen when arrow::AllocateBuffer
  and `ValueOrDie` are used inside a CUDA compilation unit.

  To work around this issue we compile an allocation shim in C++ and use
  that from our cuda sources
  */
  arrow::Result<std::unique_ptr<arrow::Buffer>> result = arrow::AllocateBuffer(size, ar_mr);
  CUDF_EXPECTS(result.ok(), "Failed to allocate Arrow buffer");
  return enable_hugepage(std::move(result).ValueOrDie());
}

std::shared_ptr<arrow::Buffer> allocate_arrow_bitmap(int64_t const size, arrow::MemoryPool* ar_mr)
{
  /*
  nvcc 11.0 generates Internal Compiler Error during codegen when arrow::AllocateBuffer
  and `ValueOrDie` are used inside a CUDA compilation unit.

  To work around this issue we compile an allocation shim in C++ and use
  that from our cuda sources
  */
  arrow::Result<std::shared_ptr<arrow::Buffer>> result = arrow::AllocateBitmap(size, ar_mr);
  CUDF_EXPECTS(result.ok(), "Failed to allocate Arrow bitmap");
  return enable_hugepage(std::move(result).ValueOrDie());
}

}  // namespace detail
}  // namespace cudf
