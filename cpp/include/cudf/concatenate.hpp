/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace cudf {
/**
 * @addtogroup copy_concatenate
 * @{
 * @file
 * @brief Concatenate columns APIs
 */

/**
 * @brief Concatenates `views[i]`'s bitmask from the bits
 * `[views[i].offset(), views[i].offset() + views[i].size())` for all elements
 * `views` into an `rmm::device_buffer`
 *
 * Returns an empty buffer if the column is not nullable.
 *
 * @param views Column views whose bitmasks will be concatenated
 * @param mr Device memory resource used for allocating the returned memory
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Bitmasks of all the column views in the views vector
 */
rmm::device_buffer concatenate_masks(
  host_span<column_view const> views,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Concatenates multiple columns into a single column
 *
 * @throws cudf::logic_error If types of the input columns mismatch
 * @throws std::overflow_error If the total number of output rows exceeds cudf::size_type
 *
 * @param columns_to_concat Column views to be concatenated into a single column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A single column having all the rows from the elements of `columns_to_concat` respectively
 * in the same order.
 */
std::unique_ptr<column> concatenate(
  host_span<column_view const> columns_to_concat,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Columns of `tables_to_concat` are concatenated vertically to return a
 * single table
 *
 * @code{.pseudo}
 * column_view c0 is {0,1,2,3}
 * column_view c1 is {4,5,6,7}
 * table_view t0{{c0, c0}};
 * table_view t1{{c1, c1}};
 * ...
 * auto t = concatenate({t0.view(), t1.view()});
 * column_view tc0 = (t->view()).column(0) is {0,1,2,3,4,5,6,7}
 * column_view tc1 = (t->view()).column(1) is {0,1,2,3,4,5,6,7}
 * @endcode
 *
 * @throws cudf::logic_error If number of columns mismatch
 * @throws std::overflow_error If the total number of output rows exceeds cudf::size_type
 *
 * @param tables_to_concat Table views to be concatenated into a single table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return A single table having all the rows from the elements of
 * `tables_to_concat` respectively in the same order.
 */
std::unique_ptr<table> concatenate(
  host_span<table_view const> tables_to_concat,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
