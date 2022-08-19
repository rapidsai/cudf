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

#include <cinttypes>
#include <cstring>
#include <cuda.h>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace ipc {
inline void check_cu_status(CUresult res)
{
  if (res != CUDA_SUCCESS) {
    char const* msg;
    if (cuGetErrorString(res, &msg) != CUDA_SUCCESS) {
      CUDF_FAIL("Unable to get CU error explanation.");
    }
    throw logic_error(msg);
  }
}

/**
 * @brief This struct represents a pointer in exported IPC format.
 */
struct exported_ptr {
  cudaIpcMemHandle_t handle{0};
  int64_t offset{0};
  int64_t size{0};

  void serialize(std::string* p_bytes) const
  {
    std::string& bytes = *p_bytes;
    size_t this_size   = sizeof(handle) + sizeof(offset) + sizeof(size);
    size_t orig_size   = p_bytes->size();

    p_bytes->resize(orig_size + this_size);
    char* ptr = bytes.data() + orig_size;

    std::memcpy(ptr, &handle, sizeof(handle));
    ptr += sizeof(handle);

    std::memcpy(ptr, &offset, sizeof(offset));
    ptr += sizeof(offset);

    std::memcpy(ptr, &size, sizeof(size));
  }

  /**
   * @brief Construct an `exported_ptr` from buffer returned by `serialize` method.
   */
  static uint8_t const* from_buffer(uint8_t const* ptr, exported_ptr* out)
  {
    exported_ptr& dptr = *out;
    std::memcpy(&dptr.handle, ptr, sizeof(handle));
    ptr += sizeof(handle);
    std::memcpy(&dptr.offset, ptr, sizeof(offset));
    ptr += sizeof(offset);
    std::memcpy(&dptr.size, ptr, sizeof(size));
    ptr += sizeof(size);

    return ptr;
  }

  /**
   * @brief Construct an `exported_ptr` from a given pointer for exportation.
   */
  static exported_ptr from_data(uint8_t const* ptr, int64_t size)
  {
    /*
     * `cudaIpcGetMemHandle` returns the same handle for pointers having the same base
     * address (the address returned from `cudaMalloc`).  So `ptr` and `ptr + 4` have the
     * same `cudaIpcMemHandle_t`. As a result, we need to calculate the offset from base
     * address ourseleves and export it along with the handle.
     */
    CUdeviceptr pbase;
    size_t psize;
    check_cu_status(cuMemGetAddressRange(&pbase, &psize, reinterpret_cast<CUdeviceptr>(ptr)));
    auto base = reinterpret_cast<uint8_t const*>(pbase);

    cudaIpcMemHandle_t handle;
    auto non_const = const_cast<void*>(reinterpret_cast<void const*>(ptr));
    CUDF_CUDA_TRY(cudaIpcGetMemHandle(&handle, non_const));

    return exported_ptr{.handle = handle, .offset = ptr - base, .size = size};
  }
};

/**
 * @brief This class represents an pointer imported from IPC handle.
 */
class imported_ptr {
  // base pointer of the memory allocation
  uint8_t* base_ptr{nullptr};
  // offset in bytes
  int64_t offset{0};
  // range of this pointer
  int64_t size{0};

 public:
  imported_ptr() = default;
  explicit imported_ptr(exported_ptr const& handle) : offset{handle.offset}, size{handle.size}
  {
    CUDF_CUDA_TRY(
      cudaIpcOpenMemHandle((void**)&base_ptr, handle.handle, cudaIpcMemLazyEnablePeerAccess));
  }
  ~imported_ptr() noexcept(false)
  {
    if (base_ptr) { CUDF_CUDA_TRY(cudaIpcCloseMemHandle(base_ptr)); }
  }

  imported_ptr(imported_ptr const& that) = delete;
  imported_ptr(imported_ptr&& that) { std::swap(that.base_ptr, this->base_ptr); }
  imported_ptr& operator=(imported_ptr const& that) = delete;
  imported_ptr& operator=(imported_ptr&& that)
  {
    std::swap(that.base_ptr, this->base_ptr);
    return *this;
  }

  template <typename T>
  auto get() const
  {
    return reinterpret_cast<T const*>(base_ptr + offset);
  }
  template <typename T>
  auto get()
  {
    return reinterpret_cast<T*>(base_ptr + offset);
  }
};

/**
 * @brief This struct represents a column in exported IPC format.
 */
struct exported_column {
  exported_ptr data;
  exported_ptr mask;

  [[nodiscard]] bool has_nulls() const { return mask.size != 0; }

  void serialize(std::string* p_bytes) const
  {
    std::string& bytes = *p_bytes;
    size_t orig_size   = p_bytes->size();
    auto hn            = has_nulls();

    bytes.resize(orig_size + sizeof(hn));
    auto ptr = bytes.data() + orig_size;
    std::memcpy(ptr, &hn, sizeof(hn));

    data.serialize(p_bytes);
    if (has_nulls()) { mask.serialize(p_bytes); }
  }

  /**
   * @brief Construct an `exported_column` from the buffer created by `serialize` method.
   */
  static uint8_t const* from_buffer(uint8_t const* ptr, exported_column* out)
  {
    bool hn;
    std::memcpy(&hn, ptr, sizeof(hn));
    ptr += sizeof(hn);

    exported_column& column = *out;
    ptr                     = exported_ptr::from_buffer(ptr, &column.data);
    if (hn) { ptr = exported_ptr::from_buffer(ptr, &column.mask); }
    return ptr;
  }
};

constexpr int64_t magic_number() { return 0xf9f9f9; }
}  // namespace ipc
}  // namespace cudf
