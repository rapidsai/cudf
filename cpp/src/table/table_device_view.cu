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

#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/rmm.h>

#include <vector>
#include <algorithm>
#include <numeric>


namespace cudf {
namespace detail {

template <typename ColumnDeviceView, typename HostTableView>
void table_device_view_base<ColumnDeviceView, HostTableView>::destroy() {
  RMM_TRY(RMM_FREE(_columns, _stream));
  delete this;
}

template <typename ColumnDeviceView, typename HostTableView>
table_device_view_base<ColumnDeviceView, HostTableView>::table_device_view_base(
    HostTableView source_view, cudaStream_t stream)
    : _num_rows{source_view.num_rows()},
      _num_columns{source_view.num_columns()},
      _stream{stream} {

  // The table's columns must be converted to ColumnDeviceView
  // objects and copied into device memory for the table_device_view's
  // _columns member.
  if (source_view.num_columns() > 0) {
    //
    // First calculate the size of memory needed to hold the
    // table's ColumnDeviceViews. This is done by calling extent()
    // for each of the table's ColumnViews columns.
    size_type views_size_bytes =
        std::accumulate(source_view.begin(), source_view.end(), 0,
            [](size_type init, auto col) {
                return init + ColumnDeviceView::extent(col);
            });
    // A buffer of CPU memory is allocated to hold the ColumnDeviceView
    // objects. Once filled, the CPU memory is then copied to device memory
    // and the pointer is set in the _columns member.
    std::vector<int8_t> h_buffer(views_size_bytes);
    ColumnDeviceView* h_column = reinterpret_cast<ColumnDeviceView*>(h_buffer.data());
    // Each ColumnDeviceView instance may have child objects which may
    // require setting some internal device pointers before being copied
    // from CPU to device.
    // Allocate the device memory to be used in the result.
    // We need this pointer in order to pass it down when creating the
    // ColumnDeviceViews so the column can set the pointer(s) for any
    // of its child objects.
    RMM_TRY(RMM_ALLOC(&_columns, views_size_bytes, stream));
    ColumnDeviceView* d_column = _columns;
    // The beginning of the memory must be the fixed-sized ColumnDeviceView
    // objects in order for _columns to be used as an array. Therefore,
    // any child data is assigned to the end of this array (h_end/d_end).
    auto h_end = (int8_t*)(h_column + source_view.num_columns());
    auto d_end = (int8_t*)(d_column + source_view.num_columns());
    // Create the ColumnDeviceView from each column within the CPU memory
    // Any column child data should be copied into h_end and any
    // internal pointers should be set using d_end.
    for( auto itr=source_view.begin(); itr!=source_view.end(); ++itr )
    {
      auto col = *itr;
      // convert the ColumnView into ColumnDeviceView
      new(h_column) ColumnDeviceView(col,(ptrdiff_t)h_end,(ptrdiff_t)d_end);
      h_column++; // point to memory slot for the next ColumnDeviceView
      // update the pointers for holding ColumnDeviceView's child data
      auto col_child_data_size = (ColumnDeviceView::extent(col) - sizeof(ColumnDeviceView));
      h_end += col_child_data_size;
      d_end += col_child_data_size;
    }
    
    CUDA_TRY(cudaMemcpyAsync(_columns, h_buffer.data(),
                             views_size_bytes, cudaMemcpyDefault, stream));
    CUDA_TRY(cudaStreamSynchronize(stream));
  }
}

// Explicit instantiation for a device table of immutable views
template class table_device_view_base<column_device_view, table_view>;

// Explicit instantiation for a device table of mutable views
template class table_device_view_base<mutable_column_device_view,
                                      mutable_table_view>;

}  // namespace detail
}  // namespace cudf
