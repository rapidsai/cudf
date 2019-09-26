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
#include <utilities/error_utils.hpp>

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
  if (source_view.num_columns() > 0) {
    //
    size_type views_size_bytes =
        std::accumulate(source_view.begin(), source_view.end(), 0,
            [](size_type init, column_view col) {
                return init + ColumnDeviceView::extent(col);
            });
    
    //CUDA_TRY(cudaMemcpyAsync(_columns, &(*source_view.begin()),
    //                         views_size_bytes, cudaMemcpyDefault, stream));

    std::vector<int8_t> h_buffer(views_size_bytes);
    ColumnDeviceView* h_column = reinterpret_cast<ColumnDeviceView*>(h_buffer.data());
    int8_t* h_end = (int8_t*)(h_column + _num_columns);
    RMM_TRY(RMM_ALLOC(&_columns, views_size_bytes, stream));
    ColumnDeviceView* d_column = _columns;
    int8_t* d_end = (int8_t*)(d_column + _num_columns);
    for( size_type idx=0; idx < _num_columns; ++idx )
    {
      auto col = source_view.column(idx);
      new(h_column) ColumnDeviceView(col,(ptrdiff_t)h_end,(ptrdiff_t)d_end);
      h_column++;
      h_end += (ColumnDeviceView::extent(col));
      d_end += (ColumnDeviceView::extent(col));
    }
    
    CUDA_TRY(cudaMemcpyAsync(_columns, h_buffer.data(),
                             views_size_bytes, cudaMemcpyDefault, stream));
  }
}

// Explicit instantiation for a device table of immutable views
template class table_device_view_base<column_device_view, table_view>;

// Explicit instantiation for a device table of mutable views
template class table_device_view_base<mutable_column_device_view,
                                      mutable_table_view>;

}  // namespace detail
}  // namespace cudf
