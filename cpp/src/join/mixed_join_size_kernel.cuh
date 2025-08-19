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

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <thrust/reduce.h>

namespace cudf {
namespace detail {

/**
 * @brief Standalone count implementation for hash table probing
 *
 * This implementation provides essential count functionality for mixed joins
 * without requiring external dependencies.
 */
template <typename StorageRef, typename ProbingScheme, typename KeyEqual, typename ProbeKey>
__device__ __forceinline__ auto standalone_count(StorageRef const& storage_ref,
                                                 ProbingScheme const& probing_scheme,
                                                 KeyEqual const& key_equal,
                                                 ProbeKey const& probe_key) noexcept
{
  using size_type                   = typename StorageRef::size_type;
  static constexpr auto bucket_size = StorageRef::bucket_size;

  size_type count   = 0;
  auto const extent = storage_ref.extent();

  // Use the probing scheme's hash functions for double hashing
  auto const hasher    = probing_scheme.hash_function();
  auto const hash1_val = cuda::std::get<0>(hasher)(probe_key);
  auto const hash2_val = cuda::std::get<1>(hasher)(probe_key);

  // Double hashing logic: initial position and step size
  auto const init_idx = (hash1_val % (extent / bucket_size)) * bucket_size;
  auto const step     = ((hash2_val % (extent / bucket_size - 1)) + 1) * bucket_size;

  auto probe_idx = init_idx;

  while (true) {
    auto const bucket_slots = storage_ref[probe_idx];

    // Check for empty slots and key equality
    auto const first_slot_is_empty  = bucket_slots[0].second == cudf::detail::JoinNoneValue;
    auto const second_slot_is_empty = bucket_slots[1].second == cudf::detail::JoinNoneValue;
    auto const first_slot_equals =
      (not first_slot_is_empty and key_equal(probe_key, bucket_slots[0]));
    auto const second_slot_equals =
      (not second_slot_is_empty and key_equal(probe_key, bucket_slots[1]));

    count += (first_slot_equals + second_slot_equals);

    // Exit if we find an empty slot
    if (first_slot_is_empty or second_slot_is_empty) { return count; }

    // Move to next bucket using double hashing
    probe_idx = (probe_idx + step) % extent;

    // Detect full cycle completion
    if (probe_idx == init_idx) { return count; }
  }
}

template <bool has_nulls>
CUDF_KERNEL void __launch_bounds__(DEFAULT_JOIN_BLOCK_SIZE) compute_mixed_join_output_size(
  table_device_view left_table,
  table_device_view right_table,
  row_hash const hash_probe,
  row_equality const equality_probe,
  join_kind const join_type,
  cudf::detail::mixed_join_hash_table_ref_t<cuco::count_tag> hash_table_ref,
  ast::detail::expression_device_view device_expression_data,
  bool const swap_tables,
  cudf::device_span<cudf::size_type> matches_per_row)
{
  // The (required) extern storage of the shared memory array leads to
  // conflicting declarations between different templates. The easiest
  // workaround is to declare an arbitrary (here char) array type then cast it
  // after the fact to the appropriate type.
  extern __shared__ char raw_intermediate_storage[];
  auto intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    intermediate_storage + (threadIdx.x * device_expression_data.num_intermediates);

  auto const start_idx                 = cudf::detail::grid_1d::global_thread_id();
  auto const stride                    = cudf::detail::grid_1d::grid_stride();
  cudf::size_type const left_num_rows  = left_table.num_rows();
  cudf::size_type const right_num_rows = right_table.num_rows();
  auto const outer_num_rows            = (swap_tables ? right_num_rows : left_num_rows);

  auto const evaluator = cudf::ast::detail::expression_evaluator<has_nulls>{
    left_table, right_table, device_expression_data};
  auto const pair = pair_fn{hash_probe};

  // Figure out the number of elements for this key.
  // TODO: Address asymmetry in operator.
  auto count_equality = pair_expression_equality<has_nulls>{
    evaluator, thread_intermediate_storage, swap_tables, equality_probe};

  // Extract storage_ref and probing_scheme from hash_table_ref
  auto const storage_ref    = hash_table_ref.storage_ref();
  auto const probing_scheme = hash_table_ref.probing_scheme();

  for (auto outer_row_index = start_idx; outer_row_index < outer_num_rows;
       outer_row_index += stride) {
    auto query_pair = pair(outer_row_index);

    // Use our standalone count function instead of relying on static_multiset_ref
    auto const count = standalone_count(storage_ref, probing_scheme, count_equality, query_pair);
    if (join_type == join_kind::LEFT_JOIN || join_type == join_kind::FULL_JOIN) {
      // Non-matching rows are counted as 1 match for the outer joins
      matches_per_row[outer_row_index] = count == 0 ? 1 : count;
    } else {
      matches_per_row[outer_row_index] = count;
    }
  }
}

template <bool has_nulls>
std::size_t launch_compute_mixed_join_output_size(
  table_device_view left_table,
  table_device_view right_table,
  row_hash const hash_probe,
  row_equality const equality_probe,
  join_kind const join_type,
  cudf::detail::mixed_join_hash_table_ref_t<cuco::count_tag> hash_table_ref,
  ast::detail::expression_device_view device_expression_data,
  bool const swap_tables,
  cudf::device_span<cudf::size_type> matches_per_row,
  detail::grid_1d const config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  compute_mixed_join_output_size<has_nulls>
    <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
      left_table,
      right_table,
      hash_probe,
      equality_probe,
      join_type,
      hash_table_ref,
      device_expression_data,
      swap_tables,
      matches_per_row);
  return thrust::reduce(
    rmm::exec_policy_nosync(stream), matches_per_row.begin(), matches_per_row.end());
}

}  // namespace detail
}  // namespace cudf
