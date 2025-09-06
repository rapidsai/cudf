/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace io::parquet::experimental {
/**
 * @addtogroup io_readers
 * @{
 * @file
 */

/**
 * @brief Reads a table from parquet source, prepends an index column to it, deserializes the
 * roaring64 deletion vector and applies it to the read table
 *
 * @ingroup io_readers
 *
 * @param options Parquet reader options
 * @param serialized_roaring64 Host span of `portable` serialized roaring64 bitmap
 * @param row_group_offsets Host span of row index offsets for each row group
 * @param row_group_num_rows Number of rows in each row group
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned table
 *
 * @return Read table with a prepended index column filtered using the deletion vector, along with
 * its metadata
 */
table_with_metadata read_parquet_and_apply_deletion_vector(
  parquet_reader_options const& options,
  cudf::host_span<cuda::std::byte const> serialized_roaring64,
  cudf::host_span<size_t const> row_group_offsets,
  cudf::host_span<size_type const> row_group_num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/** @} */  // end of group

}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
