/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>

namespace cudf {

/**
 * @brief Destroy the `table_device_view` object.
 *
 * @note Does not free the table data, simply frees the device memory
 * allocated to hold the constituent column views.
 */
void table_device_view::destroy()
{
  delete this->descendant_storage<descendant_storage_type>();
  delete this;
}

/**
 * @brief Destroy the `table_device_view` object.
 *
 * @note Does not free the table data, simply frees the device memory
 * allocated to hold the constituent column views.
 */
void mutable_table_device_view::destroy()
{
  delete this->descendant_storage<descendant_storage_type>();
  delete this;
}

table_device_view::table_device_view(host_type source_view, rmm::cuda_stream_view stream)
  : base(static_cast<column_type*>(nullptr),
         source_view.num_rows(),
         source_view.num_columns(),
         static_cast<descendant_storage_type*>(nullptr))
{
  // The table's columns must be converted to ColumnDeviceView
  // objects and copied into device memory for the table_device_view's
  // _columns member.
  auto [descendant_storage_owner, columns] =
    contiguous_copy_column_device_views<column_type, host_type>(source_view, stream);
  _columns            = static_cast<column_type*>(columns);
  _descendant_storage = descendant_storage_owner.release();
}

mutable_table_device_view::mutable_table_device_view(host_type source_view,
                                                     rmm::cuda_stream_view stream)
  : base(static_cast<column_type*>(nullptr),
         source_view.num_rows(),
         source_view.num_columns(),
         static_cast<descendant_storage_type*>(nullptr))
{
  // The table's columns must be converted to ColumnDeviceView
  // objects and copied into device memory for the mutable_table_device_view's
  // _columns member.
  auto [descendant_storage_owner, columns] =
    contiguous_copy_column_device_views<column_type, host_type>(source_view, stream);
  _columns            = static_cast<column_type*>(columns);
  _descendant_storage = descendant_storage_owner.release();
}

}  // namespace cudf
