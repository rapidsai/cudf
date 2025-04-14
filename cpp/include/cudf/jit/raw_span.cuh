/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

namespace CUDF_EXPORT cudf {

template <typename T>
struct raw_span {
  using type = T;

 private:
  type* _data     = nullptr;
  size_type _size = 0;

 public:
  constexpr CUDF_HOST_DEVICE raw_span() {}
  constexpr CUDF_HOST_DEVICE raw_span(type* data, size_type size) : _data{data}, _size{size} {}

  CUDF_HOST_DEVICE type* data() const { return _data; }

  CUDF_HOST_DEVICE size_type size() const { return _size; }

  CUDF_HOST_DEVICE bool empty() const { return _size == 0; }

  CUDF_HOST_DEVICE type& operator[](size_type i) const { return _data[i]; }

  CUDF_HOST_DEVICE type* begin() const { return _data; }

  CUDF_HOST_DEVICE type* end() const { return _data + _size; }
};

}  // namespace CUDF_EXPORT cudf
