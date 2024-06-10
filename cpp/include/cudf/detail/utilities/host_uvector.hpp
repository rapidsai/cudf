/*
 *  Copyright 2024 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <cstddef>

namespace cudf::detail {

template <typename T>
class host_uvector {
 public:
  using value_type      = T;
  using size_type       = std::size_t;
  using reference       = value_type&;
  using const_reference = value_type const&;
  using pointer         = value_type*;
  using const_pointer   = value_type const*;
  using iterator        = pointer;
  using const_iterator  = const_pointer;

  host_uvector(size_type size, rmm::host_async_resource_ref mr, rmm::cuda_stream_view stream)
    : _size{size}, _capacity{size}, _mr{mr}, _stream{stream}
  {
    if (_size != 0) {
      _data = static_cast<pointer>(mr.allocate_async(_size * sizeof(value_type), _stream));
    }
  }

  host_uvector(host_uvector const&) = delete;
  host_uvector(host_uvector&& other)
    : _data{other._data},
      _size{other._size},
      _capacity{other._capacity},
      _mr{other._mr},
      _stream{other._stream}
  {
    other._data     = nullptr;
    other._size     = 0;
    other._capacity = 0;
  }

  host_uvector& operator=(host_uvector const&) = delete;
  host_uvector& operator=(host_uvector&& other)
  {
    if (this != &other) {
      if (_data != nullptr) { _mr.deallocate_async(_data, _size * sizeof(value_type), _stream); }
      _data           = other._data;
      _size           = other._size;
      _capacity       = other._capacity;
      _mr             = other._mr;
      _stream         = other._stream;
      other._data     = nullptr;
      other._size     = 0;
      other._capacity = 0;
    }
    return *this;
  }

  ~host_uvector()
  {
    if (_data != nullptr) { _mr.deallocate_async(_data, _size * sizeof(value_type), _stream); }
  }

  void resize(size_type new_size)
  {
    if (new_size > _capacity) {
      auto new_data =
        static_cast<pointer>(_mr.allocate_async(new_size * sizeof(value_type), _stream));
      _stream.synchronize();
      if (_data != nullptr) {
        std::copy(_data, _data + _size, new_data);
        _mr.deallocate_async(_data, _size * sizeof(value_type), _stream);
      }
      _data     = new_data;
      _capacity = new_size;
    }
    _size = new_size;
  }

  void reserve(size_type new_capacity)
  {
    if (new_capacity > _capacity) {
      auto new_data =
        static_cast<pointer>(_mr.allocate_async(new_capacity * sizeof(value_type), _stream));
      _stream.synchronize();
      if (_data != nullptr) {
        std::copy(_data, _data + _size, new_data);
        _mr.deallocate_async(_data, _size * sizeof(value_type), _stream);
      }
      _data     = new_data;
      _capacity = new_capacity;
    }
  }

  void push_back(const_reference value)
  {
    if (_size == _capacity) { reserve(_capacity == 0 ? 2 : _capacity * 2); }
    _data[_size++] = value;
  }

  void clear() { _size = 0; }

  [[nodiscard]] size_type size() const { return _size; }
  [[nodiscard]] std::int64_t ssize() const { return _size; }
  [[nodiscard]] bool is_empty() const { return _size == 0; }
  [[nodiscard]] size_type capacity() const { return _capacity; }

  [[nodiscard]] reference operator[](size_type idx) { return _data[idx]; }
  [[nodiscard]] const_reference operator[](size_type idx) const { return _data[idx]; }

  [[nodiscard]] pointer data() { return _data; }
  [[nodiscard]] const_pointer data() const { return _data; }

  [[nodiscard]] reference front() { return _data[0]; }
  [[nodiscard]] const_reference front() const { return _data[0]; }

  [[nodiscard]] reference back() { return _data[_size - 1]; }
  [[nodiscard]] const_reference back() const { return _data[_size - 1]; }

  [[nodiscard]] iterator begin() { return _data; }
  [[nodiscard]] const_iterator begin() const { return _data; }

  [[nodiscard]] iterator end() { return _data + _size; }
  [[nodiscard]] const_iterator end() const { return _data + _size; }

  [[nodiscard]] rmm::host_async_resource_ref memory_resource() const { return _mr; }
  [[nodiscard]] rmm::cuda_stream_view stream() const { return _stream; }

 private:
  pointer _data{nullptr};
  size_type _size;
  size_type _capacity;
  rmm::host_async_resource_ref _mr;
  rmm::cuda_stream_view _stream;
};

}  // namespace cudf::detail
