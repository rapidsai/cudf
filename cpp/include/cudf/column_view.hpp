/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include "types.hpp"

namespace cudf {
struct column_view {
  column_view(void const* data, data_type type, size_type size,
              bitmask_view bitmask)
      : _data{data}, _type{type}, _size{size}, _bitmask{bitmask} {}

  template <typename T>
  __host__ __device__ T const* typed_data() const noexcept {
    return static_cast<T const*>(_data);
  }

  __host__ __device__ void const* data() const noexcept { return _data; }

  __device__ bool is_valid(size_type i) const noexcept {
    return _bitmask.is_valid(i);
  }

  __device__ bool is_null(size_type i) const noexcept {
    return _bitmask.is_null(i);
  }

  __host__ __device__ bool nullable() const noexcept {
    return _bitmask.nullable();
  }

  /*
    __host__ __device__ size_type null_count() const noexcept {
      return _null_count;
    }
    */

  __host__ __device__ size_type size() const noexcept { return _size; }

  __host__ __device__ data_type type() const noexcept { return _type; }

  __host__ __device__ bitmask_view const bitmask() const noexcept {
    return _bitmask;
  }

  __host__ __device__ column_view const* other() const noexcept {
    return _other;
  }

 private:
  void const* _data{nullptr};
  data_type _type{INVALID};
  cudf::size_type _size{0};
  bitmask_view _bitmask;
  // cudf::size_type _null_count{0};
  column_view* _other{nullptr};
};
}  // namespace cudf