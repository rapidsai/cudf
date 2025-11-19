/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

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
}  // namespace cudf
