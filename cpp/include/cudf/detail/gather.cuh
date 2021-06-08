/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/detail/copy.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/lists/detail/gather.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <algorithm>

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>

namespace cudf {
namespace detail {

/**
 * @brief The operation to perform when a gather map index is out of bounds
 */
enum class gather_bitmask_op {
  DONT_CHECK,   // Don't check for out of bounds indices
  PASSTHROUGH,  // Preserve mask at rows with out of bounds indices
  NULLIFY,      // Nullify rows with out of bounds indices
};

// template <gather_bitmask_op Op, typename GatherMap>
// void gather_bitmask(table_device_view input,
//                    GatherMap gather_map_begin,
//                    bitmask_type** masks,
//                    size_type mask_count,
//                    size_type mask_size,
//                    size_type* valid_counts,
//                    rmm::cuda_stream_view stream);

template <typename MapIterator>
void gather_bitmask(table_view const& source,
                    MapIterator gather_map,
                    std::vector<std::unique_ptr<column>>& target,
                    gather_bitmask_op op,
                    rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource* mr);

/**
 * @brief Gathers the specified rows of a set of columns according to a gather map.
 *
 * Gathers the rows of the source columns according to `gather_map` such that row "i"
 * in the resulting table's columns will contain row "gather_map[i]" from the source columns.
 * The number of rows in the result table will be equal to the number of elements in
 * `gather_map`.
 *
 * A negative value `i` in the `gather_map` is interpreted as `i+n`, where
 * `n` is the number of rows in the `source_table`.
 *
 * tparam MapIterator Iterator type for the gather map
 * @param[in] source_table View into the table containing the input columns whose rows will be
 * gathered
 * @param[in] gather_map_begin Beginning of iterator range of integer indices that map the rows in
 * the source columns to rows in the destination columns
 * @param[in] gather_map_end End of iterator range of integer indices that map the rows in the
 * source columns to rows in the destination columns
 * @param[in] bounds_policy Policy to apply to account for possible out-of-bound indices
 * `DONT_CHECK` skips all bound checking for gather map values. `NULLIFY` coerces rows that
 * corresponds to out-of-bound indices in the gather map to be null elements. Callers should
 * use `DONT_CHECK` when they are certain that the gather_map contains only valid indices for
 * better performance. In case there are out-of-bound indices in the gather map, the behavior
 * is undefined. Defaults to `DONT_CHECK`.
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return cudf::table Result of the gather
 */

template <typename MapIterator>
std::unique_ptr<table> gather(
  table_view const& source_table,
  MapIterator gather_map_begin,
  MapIterator gather_map_end,
  out_of_bounds_policy bounds_policy  = out_of_bounds_policy::DONT_CHECK,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace cudf
