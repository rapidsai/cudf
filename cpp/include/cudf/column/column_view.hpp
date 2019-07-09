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

#include <cudf/bitmask/bitmask_view.hpp>
#include <cudf/types.hpp>

namespace cudf {
struct column_view {
  column_view(void const* data, data_type type, size_type size,
              bitmask_view bitmask)
      : _data{data}, _type{type}, _size{size}, _bitmask{bitmask} {}

  template <typename T>
  T const* typed_data() const noexcept {
    return static_cast<T const*>(_data);
  }

  void const* data() const noexcept { return _data; }

  //__device__ bool is_valid(size_type i) const noexcept {
  //  return _bitmask.bit_is_set(i);
  //}

  //__device__ bool is_null(size_type i) const noexcept {
  //  return not is_valid(i);
  //}

  bool nullable() const noexcept { return nullptr != _bitmask.data(); }

  /*
    __host__ __device__ size_type null_count() const noexcept {
      return _null_count;
    }
    */

  size_type size() const noexcept { return _size; }

  data_type type() const noexcept { return _type; }

  bitmask_view const bitmask() const noexcept { return _bitmask; }

  column_view const* other() const noexcept { return _other; }

 private:
  void const* _data{nullptr};
  data_type _type{INVALID};
  cudf::size_type _size{0};
  bitmask_view _bitmask;
  // cudf::size_type _null_count{0};
  column_view** children{nullptr};
};
}  // namespace cudf