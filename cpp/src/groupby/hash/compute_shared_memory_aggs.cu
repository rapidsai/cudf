/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_shared_memory_aggs.hpp"
#include "global_memory_aggregator.cuh"
#include "helpers.cuh"
#include "shared_memory_aggregator.cuh"
#include "single_pass_functors.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cooperative_groups.h>
#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cudf::groupby::detail::hash {
namespace {
/// Shared memory data alignment
CUDF_HOST_DEVICE cudf::size_type constexpr ALIGNMENT = 16;

// Dictionary and nested value columns are rejected before this kernel is launched.
struct unsupported_shared_memory_type {};

template <cudf::type_id Id>
struct dispatch_shared_memory_type {
  using type = cuda::std::conditional_t<Id == cudf::type_id::DICTIONARY32 or
                                          Id == cudf::type_id::LIST or Id == cudf::type_id::STRUCT,
                                        unsupported_shared_memory_type,
                                        cudf::id_to_type<Id>>;
};

// Compound hash aggregations are decomposed into these simple aggregations before this kernel is
// launched. SUM_OVERFLOW is the only other simple hash aggregation and is explicitly rejected by
// is_shared_memory_compatible.
template <typename F, typename... Ts>
__device__ auto dispatch_shared_memory_aggregation(cudf::aggregation::Kind kind,
                                                   F&& f,
                                                   Ts&&... args)
{
  switch (kind) {
    case cudf::aggregation::SUM:
      return f.template operator()<cudf::aggregation::SUM>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::PRODUCT:
      return f.template operator()<cudf::aggregation::PRODUCT>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::MIN:
      return f.template operator()<cudf::aggregation::MIN>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::MAX:
      return f.template operator()<cudf::aggregation::MAX>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::COUNT_VALID:
      return f.template operator()<cudf::aggregation::COUNT_VALID>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::COUNT_ALL:
      return f.template operator()<cudf::aggregation::COUNT_ALL>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::SUM_OF_SQUARES:
      return f.template operator()<cudf::aggregation::SUM_OF_SQUARES>(
        cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::ARGMAX:
      return f.template operator()<cudf::aggregation::ARGMAX>(cuda::std::forward<Ts>(args)...);
    case cudf::aggregation::ARGMIN:
      return f.template operator()<cudf::aggregation::ARGMIN>(cuda::std::forward<Ts>(args)...);
    default: CUDF_UNREACHABLE("Unsupported shared memory aggregation.");
  }
}

template <typename Element>
struct dispatch_shared_memory_aggregation_fn {
  template <cudf::aggregation::Kind kind, typename F, typename... Ts>
  __device__ auto operator()(F&& f, Ts&&... args) const
  {
    return f.template operator()<Element, kind>(cuda::std::forward<Ts>(args)...);
  }
};

struct dispatch_shared_memory_source_fn {
  template <typename Element, typename F, typename... Ts>
  __device__ auto operator()(cudf::aggregation::Kind kind, F&& f, Ts&&... args) const
  {
    if constexpr (cuda::std::is_same_v<Element, unsupported_shared_memory_type>) {
      CUDF_UNREACHABLE("Unsupported shared memory aggregation type.");
    } else {
      return dispatch_shared_memory_aggregation(kind,
                                                dispatch_shared_memory_aggregation_fn<Element>{},
                                                cuda::std::forward<F>(f),
                                                cuda::std::forward<Ts>(args)...);
    }
  }
};

template <typename F, typename... Ts>
__device__ auto dispatch_shared_memory_type_and_aggregation(cudf::data_type type,
                                                            cudf::aggregation::Kind kind,
                                                            F&& f,
                                                            Ts&&... args)
{
  return cudf::type_dispatcher<dispatch_shared_memory_type>(type,
                                                            dispatch_shared_memory_source_fn{},
                                                            kind,
                                                            cuda::std::forward<F>(f),
                                                            cuda::std::forward<Ts>(args)...);
}

// Allocates shared memory required for output columns. Exits if there is insufficient memory to
// perform shared memory aggregation for the current output column.
__device__ void calculate_columns_to_aggregate(cudf::size_type& col_start,
                                               cudf::size_type& col_end,
                                               cudf::mutable_table_device_view output_values,
                                               cudf::size_type output_size,
                                               cudf::size_type* shmem_agg_res_offsets,
                                               cudf::size_type* shmem_agg_mask_offsets,
                                               cudf::size_type cardinality,
                                               cudf::size_type total_agg_size)
{
  col_start                       = col_end;
  cudf::size_type bytes_allocated = 0;

  auto const valid_col_size =
    cudf::util::round_up_safe(static_cast<cudf::size_type>(sizeof(bool) * cardinality), ALIGNMENT);

  while (bytes_allocated < total_agg_size && col_end < output_size) {
    auto const col_idx = col_end;
    auto const next_col_size =
      cudf::util::round_up_safe(cudf::type_dispatcher<cudf::dispatch_storage_type>(
                                  output_values.column(col_idx).type(), size_of_functor{}) *
                                  cardinality,
                                ALIGNMENT);
    auto const next_col_total_size = next_col_size + valid_col_size;

    if (bytes_allocated + next_col_total_size > total_agg_size) { break; }

    shmem_agg_res_offsets[col_end]  = bytes_allocated;
    shmem_agg_mask_offsets[col_end] = bytes_allocated + next_col_size;

    bytes_allocated += next_col_total_size;
    ++col_end;
  }
}

// Each block initialize its own shared memory aggregation results
__device__ void initialize_shmem_aggregations(cooperative_groups::thread_block const& block,
                                              cudf::size_type col_start,
                                              cudf::size_type col_end,
                                              cudf::mutable_table_device_view output_values,
                                              cuda::std::byte* shmem_agg_storage,
                                              cudf::size_type* shmem_agg_res_offsets,
                                              cudf::size_type* shmem_agg_mask_offsets,
                                              cudf::size_type cardinality,
                                              cudf::aggregation::Kind const* d_agg_kinds)
{
  for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
    for (auto idx = block.thread_rank(); idx < cardinality; idx += block.num_threads()) {
      auto target =
        reinterpret_cast<cuda::std::byte*>(shmem_agg_storage + shmem_agg_res_offsets[col_idx]);
      auto target_mask =
        reinterpret_cast<bool*>(shmem_agg_storage + shmem_agg_mask_offsets[col_idx]);
      dispatch_shared_memory_type_and_aggregation(output_values.column(col_idx).type(),
                                                  d_agg_kinds[col_idx],
                                                  initialize_shmem{},
                                                  target,
                                                  target_mask,
                                                  idx);
    }
  }
}

__device__ void compute_pre_aggregations(cudf::size_type col_start,
                                         cudf::size_type col_end,
                                         bitmask_type const* row_bitmask,
                                         cudf::table_device_view source,
                                         cudf::size_type num_input_rows,
                                         cudf::size_type* local_mapping_index,
                                         cuda::std::byte* shmem_agg_storage,
                                         cudf::size_type* shmem_agg_res_offsets,
                                         cudf::size_type* shmem_agg_mask_offsets,
                                         cudf::aggregation::Kind const* d_agg_kinds,
                                         cudf::size_type agg_location_offset)
{
  // Aggregates global memory sources to shared memory targets
  for (auto source_idx = cudf::detail::grid_1d::global_thread_id(); source_idx < num_input_rows;
       source_idx += cudf::detail::grid_1d::grid_stride()) {
    if (not row_bitmask or cudf::bit_is_set(row_bitmask, source_idx)) {
      auto const target_idx = local_mapping_index[source_idx] + agg_location_offset;
      for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
        auto const source_col = source.column(col_idx);

        cuda::std::byte* target =
          reinterpret_cast<cuda::std::byte*>(shmem_agg_storage + shmem_agg_res_offsets[col_idx]);
        bool* target_mask =
          reinterpret_cast<bool*>(shmem_agg_storage + shmem_agg_mask_offsets[col_idx]);

        dispatch_shared_memory_type_and_aggregation(source_col.type(),
                                                    d_agg_kinds[col_idx],
                                                    shmem_element_aggregator{},
                                                    target,
                                                    target_mask,
                                                    target_idx,
                                                    source_col,
                                                    source_idx);
      }
    }
  }
}

__device__ void compute_final_aggregations(cooperative_groups::thread_block const& block,
                                           cudf::size_type col_start,
                                           cudf::size_type col_end,
                                           cudf::table_device_view input_values,
                                           cudf::mutable_table_device_view target,
                                           cudf::size_type cardinality,
                                           cudf::size_type num_agg_locations,
                                           cudf::size_type* global_mapping_index,
                                           cuda::std::byte* shmem_agg_storage,
                                           cudf::size_type* agg_res_offsets,
                                           cudf::size_type* agg_mask_offsets,
                                           cudf::aggregation::Kind const* d_agg_kinds)
{
  // Aggregates shared memory sources to global memory targets
  for (auto idx = block.thread_rank(); idx < num_agg_locations; idx += block.num_threads()) {
    auto const target_idx =
      global_mapping_index[(block.group_index().x * GROUPBY_CARDINALITY_THRESHOLD) +
                           (idx % cardinality)];
    for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
      auto target_col = target.column(col_idx);

      cuda::std::byte* source =
        reinterpret_cast<cuda::std::byte*>(shmem_agg_storage + agg_res_offsets[col_idx]);
      bool* source_mask = reinterpret_cast<bool*>(shmem_agg_storage + agg_mask_offsets[col_idx]);

      dispatch_shared_memory_type_and_aggregation(input_values.column(col_idx).type(),
                                                  d_agg_kinds[col_idx],
                                                  gmem_element_aggregator{},
                                                  target_col,
                                                  target_idx,
                                                  input_values.column(col_idx),
                                                  source,
                                                  source_mask,
                                                  idx);
    }
  }
}

/* Takes the local_mapping_index and global_mapping_index to compute
 * pre (shared) and final (global) aggregates*/
CUDF_KERNEL void single_pass_shmem_aggs_kernel(cudf::size_type num_rows,
                                               bitmask_type const* row_bitmask,
                                               cudf::size_type* local_mapping_index,
                                               cudf::size_type* global_mapping_index,
                                               cudf::size_type* block_cardinality,
                                               cudf::table_device_view input_values,
                                               cudf::mutable_table_device_view output_values,
                                               cudf::aggregation::Kind const* d_agg_kinds,
                                               cudf::size_type total_agg_size,
                                               cudf::size_type offsets_size)
{
  auto const block       = cooperative_groups::this_thread_block();
  auto const cardinality = block_cardinality[block.group_index().x];
  if (cardinality > GROUPBY_CARDINALITY_THRESHOLD or cardinality == 0) { return; }

  auto constexpr min_shmem_agg_locations = 32;
  auto const multiplication_factor       = min_shmem_agg_locations / cardinality;
  auto const num_agg_locations           = cuda::std::max(multiplication_factor, 1) * cardinality;
  auto const agg_location_offset =
    multiplication_factor > 1 ? (block.thread_rank() % multiplication_factor) * cardinality : 0;

  auto const num_cols = output_values.num_columns();

  __shared__ cudf::size_type col_start;
  __shared__ cudf::size_type col_end;
  extern __shared__ cuda::std::byte shmem_agg_storage[];

  cudf::size_type* shmem_agg_res_offsets =
    reinterpret_cast<cudf::size_type*>(shmem_agg_storage + total_agg_size);
  cudf::size_type* shmem_agg_mask_offsets =
    reinterpret_cast<cudf::size_type*>(shmem_agg_storage + total_agg_size + offsets_size);

  if (block.thread_rank() == 0) {
    col_start = 0;
    col_end   = 0;
  }
  // Workaround: use __syncthreads() instead of block.sync() throughout this
  // kernel. cooperative_groups::thread_block::sync() does not properly fence
  // shared memory on sm_120 with CUDA 13.2, causing init stores to be
  // invisible to subsequent phases.
  __syncthreads();

  while (col_end < num_cols) {
    __syncthreads();
    if (block.thread_rank() == 0) {
      calculate_columns_to_aggregate(col_start,
                                     col_end,
                                     output_values,
                                     num_cols,
                                     shmem_agg_res_offsets,
                                     shmem_agg_mask_offsets,
                                     num_agg_locations,
                                     total_agg_size);
    }
    __syncthreads();

    initialize_shmem_aggregations(block,
                                  col_start,
                                  col_end,
                                  output_values,
                                  shmem_agg_storage,
                                  shmem_agg_res_offsets,
                                  shmem_agg_mask_offsets,
                                  num_agg_locations,
                                  d_agg_kinds);
    __syncthreads();

    compute_pre_aggregations(col_start,
                             col_end,
                             row_bitmask,
                             input_values,
                             num_rows,
                             local_mapping_index,
                             shmem_agg_storage,
                             shmem_agg_res_offsets,
                             shmem_agg_mask_offsets,
                             d_agg_kinds,
                             agg_location_offset);
    __syncthreads();

    compute_final_aggregations(block,
                               col_start,
                               col_end,
                               input_values,
                               output_values,
                               cardinality,
                               num_agg_locations,
                               global_mapping_index,
                               shmem_agg_storage,
                               shmem_agg_res_offsets,
                               shmem_agg_mask_offsets,
                               d_agg_kinds);
  }
}

CUDF_KERNEL void mapped_block_private_aggs_kernel(cudf::size_type num_rows,
                                                  cudf::size_type const* matching_keys,
                                                  cudf::size_type const* key_transform_map,
                                                  cudf::table_device_view input_values,
                                                  cudf::mutable_table_device_view target_values,
                                                  cudf::size_type num_output_rows,
                                                  cudf::size_type num_replicas,
                                                  cudf::aggregation::Kind const* d_agg_kinds)
{
  for (auto source_idx = cudf::detail::grid_1d::global_thread_id(); source_idx < num_rows;
       source_idx += cudf::detail::grid_1d::grid_stride()) {
    auto const sparse_target_idx = matching_keys[source_idx];
    if (sparse_target_idx == cudf::detail::CUDF_SIZE_TYPE_SENTINEL) { continue; }

    auto const replica_idx = static_cast<cudf::size_type>(blockIdx.x % num_replicas);
    auto const target_idx  = key_transform_map[sparse_target_idx];
    auto const private_idx = replica_idx * num_output_rows + target_idx;
    for (auto col_idx = 0; col_idx < input_values.num_columns(); ++col_idx) {
      auto const source_col = input_values.column(col_idx);
      dispatch_shared_memory_type_and_aggregation(source_col.type(),
                                                  d_agg_kinds[col_idx],
                                                  cudf::detail::element_aggregator{},
                                                  target_values.column(col_idx),
                                                  private_idx,
                                                  source_col,
                                                  source_idx);
    }
  }
}

struct block_private_reducer {
  template <typename Source, cudf::aggregation::Kind k>
  __device__ void operator()(cudf::column_device_view source_col,
                             cudf::mutable_column_device_view private_col,
                             cudf::mutable_column_device_view output_col,
                             cudf::size_type target_idx,
                             cudf::size_type num_output_rows,
                             cudf::size_type num_replicas) const
  {
    using Target       = cudf::device_storage_type_t<cudf::detail::target_type_t<Source, k>>;
    auto* private_data = reinterpret_cast<cuda::std::byte*>(private_col.template head<void>());

    for (auto replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
      auto const private_idx = replica_idx * num_output_rows + target_idx;
      if (private_col.nullable() && private_col.is_null(private_idx)) { continue; }

      // Non-nullable ARGMIN/ARGMAX intermediates use an out-of-range index as identity. Do not
      // feed that identity to the comparison loop in update_target_element_gmem.
      if constexpr (k == cudf::aggregation::ARGMIN) {
        if (private_col.template element<Target>(private_idx) == cudf::detail::ARGMIN_SENTINEL) {
          continue;
        }
      } else if constexpr (k == cudf::aggregation::ARGMAX) {
        if (private_col.template element<Target>(private_idx) == cudf::detail::ARGMAX_SENTINEL) {
          continue;
        }
      }

      if constexpr (!(k == cudf::aggregation::COUNT_VALID || k == cudf::aggregation::COUNT_ALL)) {
        if (output_col.nullable() && output_col.is_null(target_idx)) {
          output_col.set_valid(target_idx);
        }
      }
      update_target_element_gmem<Source, k>{}(
        output_col, target_idx, source_col, private_data, private_idx);
    }
  }
};

CUDF_KERNEL void reduce_block_private_aggs_kernel(cudf::size_type num_output_rows,
                                                  cudf::size_type num_replicas,
                                                  cudf::table_device_view input_values,
                                                  cudf::mutable_table_device_view private_values,
                                                  cudf::mutable_table_device_view output_values,
                                                  cudf::aggregation::Kind const* d_agg_kinds)
{
  auto const idx         = cudf::detail::grid_1d::global_thread_id();
  auto const num_columns = output_values.num_columns();
  if (idx >= num_output_rows * num_columns) { return; }

  auto const col_idx    = idx / num_output_rows;
  auto const target_idx = idx % num_output_rows;
  auto const source_col = input_values.column(col_idx);
  dispatch_shared_memory_type_and_aggregation(source_col.type(),
                                              d_agg_kinds[col_idx],
                                              block_private_reducer{},
                                              source_col,
                                              private_values.column(col_idx),
                                              output_values.column(col_idx),
                                              target_idx,
                                              num_output_rows,
                                              num_replicas);
}
}  // namespace

size_type get_available_shared_memory_size(cudf::size_type grid_size)
{
  auto const active_blocks_per_sm =
    cudf::util::div_rounding_up_safe(grid_size, cudf::detail::num_multiprocessors());

  size_t dynamic_shmem_size = 0;
  CUDF_CUDA_TRY(cudaOccupancyAvailableDynamicSMemPerBlock(
    &dynamic_shmem_size, single_pass_shmem_aggs_kernel, active_blocks_per_sm, GROUPBY_BLOCK_SIZE));
  return cudf::util::round_down_safe(static_cast<cudf::size_type>(0.5 * dynamic_shmem_size),
                                     ALIGNMENT);
}

int32_t max_active_blocks_shmem_aggs_kernel()
{
  int32_t max_active_blocks{-1};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, single_pass_shmem_aggs_kernel, GROUPBY_BLOCK_SIZE, 0));
  return max_active_blocks;
}

void compute_shared_memory_aggs(cudf::size_type grid_size,
                                size_type available_shmem_size,
                                cudf::size_type num_input_rows,
                                bitmask_type const* row_bitmask,
                                cudf::size_type* local_mapping_index,
                                cudf::size_type* global_mapping_index,
                                cudf::size_type* block_cardinality,
                                cudf::table_device_view input_values,
                                cudf::mutable_table_device_view output_values,
                                cudf::aggregation::Kind const* d_agg_kinds,
                                cuda::stream_ref stream)
{
  // For each aggregation, need one offset determining where the aggregation is
  // performed, another indicating the validity of the aggregation
  auto const offsets_size = compute_shmem_offsets_size(output_values.num_columns());
  // The rest of shmem is utilized for the actual arrays in shmem
  CUDF_EXPECTS(available_shmem_size > offsets_size * 2,
               "No enough space for shared memory aggregations");
  auto const shmem_agg_size = available_shmem_size - offsets_size * 2;
  single_pass_shmem_aggs_kernel<<<grid_size,
                                  GROUPBY_BLOCK_SIZE,
                                  available_shmem_size,
                                  stream.get()>>>(num_input_rows,
                                                  row_bitmask,
                                                  local_mapping_index,
                                                  global_mapping_index,
                                                  block_cardinality,
                                                  input_values,
                                                  output_values,
                                                  d_agg_kinds,
                                                  shmem_agg_size,
                                                  offsets_size);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void compute_block_private_aggs(cudf::size_type grid_size,
                                cudf::size_type block_size,
                                cudf::size_type num_replicas,
                                cudf::size_type num_input_rows,
                                cudf::size_type const* matching_keys,
                                cudf::size_type const* key_transform_map,
                                cudf::table_device_view input_values,
                                cudf::mutable_table_device_view output_values,
                                cudf::mutable_table_device_view private_values,
                                cudf::size_type num_output_rows,
                                cudf::aggregation::Kind const* d_agg_kinds,
                                cuda::stream_ref stream)
{
  CUDF_EXPECTS(grid_size > 0, "Invalid block-private grid size");
  CUDF_EXPECTS(block_size > 0 && block_size <= 1024, "Invalid block-private block size");
  CUDF_EXPECTS(num_replicas > 0 && num_replicas <= cudf::detail::num_multiprocessors(),
               "Invalid block-private replica count");
  CUDF_EXPECTS(num_output_rows <= std::numeric_limits<cudf::size_type>::max() / num_replicas,
               "Block-private output size exceeds the column size limit");

  mapped_block_private_aggs_kernel<<<grid_size, block_size, 0, stream.get()>>>(num_input_rows,
                                                                               matching_keys,
                                                                               key_transform_map,
                                                                               input_values,
                                                                               private_values,
                                                                               num_output_rows,
                                                                               num_replicas,
                                                                               d_agg_kinds);
  CUDF_CUDA_TRY(cudaGetLastError());

  auto const num_columns = output_values.num_columns();
  CUDF_EXPECTS(
    num_columns > 0 && num_output_rows <= std::numeric_limits<cudf::size_type>::max() / num_columns,
    "Block-private reduction size exceeds the launch size limit");
  auto const num_items        = num_output_rows * num_columns;
  auto const reduce_grid_size = cudf::util::div_rounding_up_safe(num_items, GROUPBY_BLOCK_SIZE);
  reduce_block_private_aggs_kernel<<<reduce_grid_size, GROUPBY_BLOCK_SIZE, 0, stream.get()>>>(
    num_output_rows, num_replicas, input_values, private_values, output_values, d_agg_kinds);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void compute_mapped_global_aggs(cudf::size_type grid_size,
                                cudf::size_type block_size,
                                cudf::size_type num_input_rows,
                                cudf::size_type const* matching_keys,
                                cudf::size_type const* key_transform_map,
                                cudf::table_device_view input_values,
                                cudf::mutable_table_device_view output_values,
                                cudf::size_type num_output_rows,
                                cudf::aggregation::Kind const* d_agg_kinds,
                                cuda::stream_ref stream)
{
  CUDF_EXPECTS(grid_size > 0, "Invalid mapped-global grid size");
  CUDF_EXPECTS(block_size > 0 && block_size <= 1024, "Invalid mapped-global block size");
  mapped_block_private_aggs_kernel<<<grid_size, block_size, 0, stream.get()>>>(num_input_rows,
                                                                               matching_keys,
                                                                               key_transform_map,
                                                                               input_values,
                                                                               output_values,
                                                                               num_output_rows,
                                                                               1,
                                                                               d_agg_kinds);
  CUDF_CUDA_TRY(cudaGetLastError());
}
}  // namespace cudf::groupby::detail::hash
