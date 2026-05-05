/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <memory>
#include <numeric>

namespace cudf {
namespace detail {
template <typename ColumnDeviceView, typename HostTableView>
void table_device_view_base<ColumnDeviceView, HostTableView>::destroy()
{
  delete _descendant_storage;
  delete this;
}

template <typename ColumnDeviceView, typename HostTableView>
table_device_view_base<ColumnDeviceView, HostTableView>::table_device_view_base(
  HostTableView source_view, rmm::cuda_stream_view stream)
  : _num_rows{source_view.num_rows()}, _num_columns{source_view.num_columns()}
{
  // The table's columns must be converted to ColumnDeviceView
  // objects and copied into device memory for the table_device_view's
  // _columns member.
  if (source_view.num_columns() > 0) {
    std::unique_ptr<rmm::device_buffer> descendant_storage_owner;
    std::tie(descendant_storage_owner, _columns) =
      contiguous_copy_column_device_views<ColumnDeviceView, HostTableView>(source_view, stream);
    _descendant_storage = descendant_storage_owner.release();
  }
}

// Explicit instantiation for a device table of immutable views
template class table_device_view_base<column_device_view, table_view>;

// Explicit instantiation for a device table of mutable views
template class table_device_view_base<mutable_column_device_view, mutable_table_view>;

}  // namespace detail

template <typename ColumnDeviceView, typename HostTableView>
std::pair<std::unique_ptr<rmm::device_buffer>, ColumnDeviceView*>
contiguous_copy_column_device_views(HostTableView source_view, rmm::cuda_stream_view stream)
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
  auto h_buffer = cudf::detail::make_pinned_vector_async<int8_t>(padded_views_size_bytes, stream);
  // Each ColumnDeviceView instance may have child objects which may
  // require setting some internal device pointers before being copied
  // from CPU to device.
  // Allocate the device memory to be used in the result.
  // We need this pointer in order to pass it down when creating the
  // ColumnDeviceViews so the column can set the pointer(s) for any
  // of its child objects.
  // align both h_ptr, d_ptr
  auto descendant_storage = std::make_unique<rmm::device_buffer>(padded_views_size_bytes, stream);
  void* h_ptr             = detail::align_ptr_for_type<ColumnDeviceView>(h_buffer.data());
  void* d_ptr    = detail::align_ptr_for_type<ColumnDeviceView>(descendant_storage->data());
  auto d_columns = detail::child_columns_to_device_array<ColumnDeviceView>(
    source_view.begin(), source_view.end(), h_ptr, d_ptr);

  auto const h_span = host_span<int8_t const>{h_buffer}.subspan(
    static_cast<int8_t const*>(h_ptr) - h_buffer.data(), views_size_bytes);
  auto const d_span = device_span<int8_t>{static_cast<int8_t*>(d_ptr), views_size_bytes};
  cudf::detail::cuda_memcpy(d_span, h_span, stream);
  return std::make_pair(std::move(descendant_storage), d_columns);
}

template std::pair<std::unique_ptr<rmm::device_buffer>, column_device_view*>
contiguous_copy_column_device_views<column_device_view, host_span<column_view const>>(
  host_span<column_view const> source_view, rmm::cuda_stream_view stream);

}  // namespace cudf
