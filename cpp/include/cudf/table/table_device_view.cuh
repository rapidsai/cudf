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
#include <numeric>

/**
 * @file table_device_view.cuh
 * @brief Table device view class definitons
 */

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

template <typename ColumnDeviceView, typename HostTableView>
auto contiguous_copy_column_device_views(HostTableView source_view, rmm::cuda_stream_view stream)
{
  // First calculate the size of memory needed to hold the
  // table's ColumnDeviceViews. This is done by calling extent()
  // for each of the table's ColumnViews columns.
  std::size_t views_size_bytes = std::accumulate(
    source_view.begin(), source_view.end(), std::size_t{0}, [](std::size_t init, auto col) {
      return init + ColumnDeviceView::extent(col);
    });
  // pad the allocation for aligning the first pointer
  auto padded_views_size_bytes = views_size_bytes + std::size_t{alignof(ColumnDeviceView) - 1};
  // A buffer of CPU memory is allocated to hold the ColumnDeviceView
  // objects. Once filled, the CPU memory is then copied to device memory
  // and the pointer is set in the d_columns member.
  std::vector<int8_t> h_buffer(padded_views_size_bytes);
  // Each ColumnDeviceView instance may have child objects which may
  // require setting some internal device pointers before being copied
  // from CPU to device.
  // Allocate the device memory to be used in the result.
  // We need this pointer in order to pass it down when creating the
  // ColumnDeviceViews so the column can set the pointer(s) for any
  // of its child objects.
  auto _descendant_storage = new rmm::device_buffer(padded_views_size_bytes, stream);
  void* h_ptr              = h_buffer.data();
  void* d_ptr              = _descendant_storage->data();
  auto d_columns           = detail::child_columns_to_device_array<ColumnDeviceView>(
    source_view.begin(), source_view.end(), h_ptr, d_ptr);

  // align h_ptr also, because both h_ptr, d_ptr alignment will not be same!
  auto aligned_hptr = detail::align_ptr_for_type<ColumnDeviceView>(h_buffer.data());
  CUDA_TRY(cudaMemcpyAsync(d_columns, aligned_hptr, views_size_bytes, cudaMemcpyDefault, stream.value()));
  stream.synchronize();
  return std::make_tuple(_descendant_storage, d_columns);
}

}  // namespace cudf
