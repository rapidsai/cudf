/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "join_common_utils.cuh"
#include "join_common_utils.hpp"
#include "mixed_join_common_utils.cuh"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <thrust/iterator/discard_iterator.h>

namespace cudf {
namespace detail {

/**
 * @brief Computes the output size of joining the left table to the right table.
 *
 * This method probes the hash table with each row in the probe table using a
 * custom equality comparator that also checks that the conditional expression
 * evaluates to true between the left/right tables when a match is found
 * between probe and build rows.
 *
 * @tparam block_size The number of threads per block for this kernel
 * @tparam has_nulls Whether or not the inputs may contain nulls.
 *
 * @param[in] left_table The left table
 * @param[in] right_table The right table
 * @param[in] probe The table with which to probe the hash table for matches.
 * @param[in] build The table with which the hash table was built.
 * @param[in] hash_probe The hasher used for the probe table.
 * @param[in] equality_probe The equality comparator used when probing the hash table.
 * @param[in] join_type The type of join to be performed
 * @param[in] hash_table_view The hash table built from `build`.
 * @param[in] device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param[in] swap_tables If true, the kernel was launched with one thread per right row and
 * the kernel needs to internally loop over left rows. Otherwise, loop over right rows.
 * @param[out] output_size The resulting output size
 * @param[out] matches_per_row The number of matches in one pair of
 * equality/conditional tables for each row in the other pair of tables. If
 * swap_tables is true, matches_per_row corresponds to the right_table,
 * otherwise it corresponds to the left_table. Note that corresponding swap of
 * left/right tables to determine which is the build table and which is the
 * probe table has already happened on the host.
 */

template <bool has_nulls>
std::size_t launch_compute_mixed_join_output_size(
  cudf::table_device_view left_table,
  cudf::table_device_view right_table,
  cudf::table_device_view probe,
  cudf::table_device_view build,
  row_hash const hash_probe,
  row_equality const equality_probe,
  join_kind const join_type,
  cudf::detail::mixed_multimap_type::device_view hash_table_view,
  ast::detail::expression_device_view device_expression_data,
  bool const swap_tables,
  cudf::device_span<cudf::size_type> matches_per_row,
  detail::grid_1d const config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
}  // namespace detail
}  // namespace cudf
