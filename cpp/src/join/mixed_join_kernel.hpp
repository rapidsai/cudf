/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "join/join_common_utils.hpp"
#include "join/mixed_join_common_utils.cuh"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/span.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Performs a join using the combination of a hash lookup to identify
 * equal rows between one pair of tables and the evaluation of an expression
 * containing an arbitrary expression.
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
 * @param[out] join_output_l The left result of the join operation
 * @param[out] join_output_r The right result of the join operation
 * @param[in] device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param[in] join_result_offsets The starting indices in join_output[l|r]
 * where the matches for each row begin. Equivalent to a prefix sum of
 * matches_per_row.
 * @param[in] swap_tables If true, the kernel was launched with one thread per right row and
 * the kernel needs to internally loop over left rows. Otherwise, loop over right rows.
 */
template <bool has_nulls>
void launch_mixed_join(table_device_view left_table,
                       table_device_view right_table,
                       table_device_view probe,
                       table_device_view build,
                       row_hash const hash_probe,
                       row_equality const equality_probe,
                       join_kind const join_type,
                       cudf::detail::mixed_multimap_type::device_view hash_table_view,
                       size_type* join_output_l,
                       size_type* join_output_r,
                       cudf::ast::detail::expression_device_view device_expression_data,
                       cudf::size_type const* join_result_offsets,
                       bool const swap_tables,
                       detail::grid_1d const config,
                       int64_t shmem_size_per_block,
                       rmm::cuda_stream_view stream);

}  // namespace detail

}  // namespace CUDF_EXPORT cudf
