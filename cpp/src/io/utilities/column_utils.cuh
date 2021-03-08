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

#include <cudf/column/column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace io {

/**
 * @brief Create column_device_view pointers from leaf columns
 *
 * A device_uvector is created to store the leaves of parent columns. The
 * column descriptor array is updated to point to these leaf columns.
 *
 * @tparam ColumnDescriptor Struct describing properties of columns with
 * pointers to leaf and parent columns
 *
 * @param col_desc Column description array
 * @param parent_table_device_view Table device view containing parent columns
 * @param stream CUDA stream to use
 *
 * @return Device array containing leaf column device views
 */
template <typename ColumnDescriptor>
rmm::device_uvector<column_device_view> create_leaf_column_device_views(
  typename cudf::device_span<ColumnDescriptor> col_desc,
  const table_device_view &parent_table_device_view,
  rmm::cuda_stream_view stream)
{
  rmm::device_uvector<column_device_view> leaf_column_views(parent_table_device_view.num_columns(),
                                                            stream);
  auto leaf_columns = cudf::device_span<column_device_view>{leaf_column_views};

  auto iter = thrust::make_counting_iterator<size_type>(0);
  thrust::for_each(rmm::exec_policy(stream),
                   iter,
                   iter + parent_table_device_view.num_columns(),
                   [col_desc, parent_col_view = parent_table_device_view, leaf_columns] __device__(
                     size_type index) mutable {
                     column_device_view col = parent_col_view.column(index);

                     if (col.type().id() == type_id::LIST) {
                       col_desc[index].parent_column = parent_col_view.begin() + index;
                     } else {
                       col_desc[index].parent_column = nullptr;
                     }
                     // traverse till leaf column
                     while (col.type().id() == type_id::LIST) {
                       col = col.child(lists_column_view::child_column_index);
                     }
                     // Store leaf_column to device storage
                     column_device_view *leaf_col_ptr = leaf_columns.begin() + index;
                     *leaf_col_ptr                    = col;
                     col_desc[index].leaf_column      = leaf_col_ptr;
                   });

  return leaf_column_views;
}

}  // namespace io
}  // namespace cudf
