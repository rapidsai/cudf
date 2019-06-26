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
#include "types.hpp"

#include <rmm/device_buffer.hpp>

namespace cudf {

class bitmask_view {
 public:
  bitmask_view(bitmask_type const* mask, size_type size)
      : _mask{mask}, _size{size} {}

  __device__ bool is_valid(size_type i) const noexcept {
    // FIXME Implement
    return true;
  }

  __device__ bool is_null(size_type i) const noexcept {
    return not is_valid(i);
  }

  __host__ __device__ bool nullable() const noexcept {
    return nullptr != _mask;
  }

  __host__ __device__ bitmask_type const* data() const noexcept {
    return _mask;
  }

 private:
  bitmask_type const* _mask{nullptr};
  size_type _size{0};
};

class bitmask {
 public:
  operator bitmask_view() const noexcept {
    return bitmask_view{static_cast<bitmask_type const*>(data.data()), size};
  }

 private:
  rmm::device_buffer data{};
  size_type size{};
};

}  // namespace cudf