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

#pragma once

#include <cudf/utilities/span.hpp>

#include <io/utilities/hostdevice_vector.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

struct matrix_size {
  size_t num_rows;
  size_t num_columns;
};

constexpr size_t compute_offset(size_t row, size_t column, matrix_size size) noexcept
{
  return row * size.num_columns + column;
}

template <typename T>
class matrix_device_view {
 public:
  matrix_device_view(T* ptr, matrix_size size) : _data_ptr{ptr}, _size(size) {}

  constexpr device_span<T> operator[](size_t row)
  {
    return {_data_ptr + compute_offset(row, 0, _size), _size.num_columns};
  }
  constexpr auto size() { return _size; }

 private:
  T* _data_ptr;
  matrix_size _size;
};

template <typename T>
class hostdevice_matrix {
 public:
  hostdevice_matrix(size_t rows,
                    size_t columns,
                    rmm::cuda_stream_view stream = rmm::cuda_stream_default)
    : _size{rows, columns}, _data{rows * columns, stream}
  {
  }
  operator matrix_device_view<T>() { return {_data.device_ptr(), _size}; }
  operator matrix_device_view<T const>() const { return {_data.device_ptr(), _size}; }

  host_span<T> operator[](size_t row)
  {
    return {_data.host_ptr() + compute_offset(row, 0, _size), _size.num_columns};
  }

  host_span<T const> operator[](size_t row) const
  {
    return {_data.host_ptr() + compute_offset(row, 0, _size), _size.num_columns};
  }

  auto size() { return _size; }

  void host_to_device(rmm::cuda_stream_view stream, bool synchronize = false)
  {
    _data.host_to_device(stream, synchronize);
  }

  void device_to_host(rmm::cuda_stream_view stream, bool synchronize = false)
  {
    _data.device_to_host(stream, synchronize);
  }

 private:
  matrix_size _size;
  hostdevice_vector<T> _data;
};

}  // namespace detail
}  // namespace cudf
