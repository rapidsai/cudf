/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "join_common_utils.cuh"
#include "join_common_utils.hpp"
#include "mixed_join_common_utils.cuh"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuco/pair.cuh>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Performs a mixed join using hash lookup and expression evaluation.
 *
 * This method probes the hash table with each row in the probe table using a
 * custom equality comparator that also checks that the conditional expression
 * evaluates to true between the left/right tables when a match is found
 * between probe and build rows.
 *
 * @tparam has_nulls Whether or not the inputs may contain nulls.
 *
 * @param left_table The left table
 * @param right_table The right table
 * @param join_type The type of join to be performed
 * @param equality_probe The equality comparator used when probing the hash table
 * @param hash_table_storage The hash table storage for probing operations
 * @param input_pairs Array of hash-value/row-index pairs for probing
 * @param hash_indices Array of hash index pairs for efficient lookup
 * @param join_output_l The left result of the join operation
 * @param join_output_r The right result of the join operation
 * @param device_expression_data Container of device data required to evaluate the desired
 * expression
 * @param swap_tables If true, the kernel was launched with one thread per right row and
 * the kernel needs to internally loop over left rows. Otherwise, loop over right rows
 * @param config Grid configuration for kernel launch
 * @param shmem_size_per_block Shared memory size per block in bytes
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <bool has_nulls>
void launch_mixed_join(
  cudf::table_device_view left_table,
  cudf::table_device_view right_table,
  join_kind join_type,
  row_equality equality_probe,
  cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
  cuco::pair<hash_value_type, cudf::size_type> const* input_pairs,
  cuda::std::pair<uint32_t, uint32_t> const* hash_indices,
  cudf::size_type* join_output_l,
  cudf::size_type* join_output_r,
  cudf::ast::detail::expression_device_view device_expression_data,
  bool swap_tables,
  detail::grid_1d const& config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
