/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cassert>
#include <memory>

namespace cudf {
namespace detail {
template <typename ColumnDeviceView, typename HostTableView>
class table_device_view_base {
 public:
  table_device_view_base()                              = delete;
  ~table_device_view_base()                             = default;
  table_device_view_base(table_device_view_base const&) = default;
  table_device_view_base(table_device_view_base&&)      = default;
  table_device_view_base& operator=(table_device_view_base const&) = default;
  table_device_view_base& operator=(table_device_view_base&&) = default;

  __device__ ColumnDeviceView* begin() const noexcept { return _columns; }

  __device__ ColumnDeviceView* end() const noexcept { return _columns + _num_columns; }

  __device__ ColumnDeviceView const& column(size_type column_index) const noexcept
  {
    assert(column_index >= 0);
    assert(column_index < _num_columns);
    return _columns[column_index];
  }

  __device__ ColumnDeviceView& column(size_type column_index) noexcept
  {
    assert(column_index >= 0);
    assert(column_index < _num_columns);
    return _columns[column_index];
  }

  __host__ __device__ size_type num_columns() const noexcept { return _num_columns; }

  __host__ __device__ size_type num_rows() const noexcept { return _num_rows; }

  void destroy();

 private:
  ColumnDeviceView* _columns{};  ///< Array of view objects in device memory
  size_type _num_rows{};
  size_type _num_columns{};
  cudaStream_t _stream{};

 protected:
  table_device_view_base(HostTableView source_view, cudaStream_t stream);

  rmm::device_buffer* _descendant_storage{};
};
}  // namespace detail

class table_device_view : public detail::table_device_view_base<column_device_view, table_view> {
 public:
  static auto create(table_view source_view, cudaStream_t stream = 0)
  {
    auto deleter = [](table_device_view* t) { t->destroy(); };
    return std::unique_ptr<table_device_view, decltype(deleter)>{
      new table_device_view(source_view, stream), deleter};
  }

 private:
  table_device_view(table_view source_view, cudaStream_t stream)
    : detail::table_device_view_base<column_device_view, table_view>(source_view, stream)
  {
  }
};

class mutable_table_device_view
  : public detail::table_device_view_base<mutable_column_device_view, mutable_table_view> {
 public:
  static auto create(mutable_table_view source_view, cudaStream_t stream = 0)
  {
    auto deleter = [](mutable_table_device_view* t) { t->destroy(); };
    return std::unique_ptr<mutable_table_device_view, decltype(deleter)>{
      new mutable_table_device_view(source_view, stream), deleter};
  }

 private:
  mutable_table_device_view(mutable_table_view source_view, cudaStream_t stream)
    : detail::table_device_view_base<mutable_column_device_view, mutable_table_view>(source_view,
                                                                                     stream)
  {
  }
};
}  // namespace cudf
