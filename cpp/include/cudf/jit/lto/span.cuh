/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/jit/lto/export.cuh>
#include <cudf/jit/lto/types.cuh>

namespace CUDF_LTO_EXPORT cudf {
namespace lto {

template <typename T>
struct [[nodiscard]] span {
 private:
  T* _data     = nullptr;
  size_t _size = 0;

 public:
  __device__ T* data() const { return _data; }

  __device__ size_t size() const { return _size; }

  __device__ bool empty() const { return _size == 0; }

  __device__ T& operator[](size_t pos) const { return _data[pos]; }

  __device__ T* begin() const { return _data; }

  __device__ T* end() const { return _data + _size; }

  __device__ span<T const> as_const() const { return span<T const>{_data, _size}; }

  __device__ T& element(size_t idx) const { return _data[idx]; }

  __device__ void assign(size_t idx, T value) const { _data[idx] = value; }
};

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
