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

#include <algorithm>

namespace cudf {

namespace detail {
template <typename ColumnDeviceView>
void table_device_view_base<ColumnDeviceView>::destroy() {
  RMM_TRY(RMM_FREE(_columns, _stream));
  delete this;
}
}  // namespace detail

auto table_device_view::create(table_view source_view, cudaStream_t stream) {
  size_type total_descendants =
      std::accumulate(source_view.begin(), source_view.end(), 0,
                      [](size_type init, column_view col) {
                        return init + count_descendants(col);
                      });
  CUDF_EXPECTS(0 == total_descendants,
               "Columns with descendants are not yet supported.");

  auto deleter = [](table_device_view* t) { t->destroy(); };

  return std::unique_ptr<table_device_view, decltype(deleter)>{
      new table_device_view(source_view, stream), deleter};
}

table_device_view::table_device_view(table_view source_view,
                                     cudaStream_t stream)
    : detail::table_device_view_base<column_device_view>(
          source_view.num_rows(), source_view.num_columns(), stream) {
  auto views_size_bytes =
      source_view.num_columns() * sizeof(column_device_view);
  RMM_TRY(RMM_ALLOC(_columns, views_size_bytes, stream));
  CUDA_TRY(cudaMemcpy(_columns, source_view.begin(), views_size_bytes));
}

auto mutable_table_device_view::create(mutable_table_view source_view,
                                       cudaStream_t stream) {
  size_type total_descendants =
      std::accumulate(source_view.begin(), source_view.end(), 0,
                      [](size_type init, column_view col) {
                        return init + count_descendants(col);
                      });
  CUDF_EXPECTS(0 == total_descendants,
               "Columns with descendants are not yet supported.");

  auto deleter = [](mutable_table_device_view* t) { t->destroy(); };

  return std::unique_ptr<mutable_table_device_view, decltype(deleter)>{
      new mutable_table_device_view(source_view, stream), deleter};
}

mutable_table_device_view::mutable_table_device_view(
    mutable_table_view source_view, cudaStream_t stream)
    : detail::table_device_view_base<mutable_column_device_view>(
          source_view.num_rows(), source_view.num_columns(), stream) {
  auto views_size_bytes =
      source_view.num_columns() * sizeof(mutable_column_device_view);
  RMM_TRY(RMM_ALLOC(_columns, views_size_bytes, stream));
  CUDA_TRY(cudaMemcpy(_columns, source_view.begin(), views_size_bytes));
}

}  // namespace cudf
