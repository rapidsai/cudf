

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/jit/lto/export.cuh>
#include <cudf/jit/lto/optional.cuh>
#include <cudf/jit/lto/types.cuh>

namespace CUDF_LTO_EXPORT cudf {
namespace lto {

__device__ constexpr bool bit_is_set(bitmask_type const* bitmask, size_t bit_index)
{
  constexpr auto bits_per_word = sizeof(bitmask_type) * 8;
  return bitmask[bit_index / bits_per_word] & (bitmask_type{1} << (bit_index % bits_per_word));
}

template <typename T>
struct [[nodiscard]] optional_span {
 private:
  T* _data                 = nullptr;
  size_t _size             = 0;
  bitmask_type* _null_mask = nullptr;

 public:
  __device__ T* data() const { return _data; }

  __device__ size_t size() const { return _size; }

  __device__ bool empty() const { return _size == 0; }

  __device__ T& operator[](size_t pos) const { return _data[pos]; }

  __device__ T* begin() const { return _data; }

  __device__ T* end() const { return _data + _size; }

  __device__ optional_span<T const> as_const() const
  {
    return optional_span<T const>{_data, _size, _null_mask};
  }

  __device__ bool nullable() const { return _null_mask != nullptr; }

  [[nodiscard]] __device__ bool is_valid_nocheck(size_t element_index) const
  {
    return bit_is_set(_null_mask, element_index);
  }

  __device__ bool is_valid(size_t element_index) const
  {
    return not nullable() or is_valid_nocheck(element_index);
  }

  __device__ bool is_null(size_t element_index) const { return !is_valid(element_index); }

  __device__ T& element(size_t idx) const { return _data[idx]; }

  __device__ optional<T> nullable_element(size_t idx) const;

  __device__ void assign(size_t idx, T value) const { _data[idx] = value; }
};

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
