/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/hash/single_pass_functors.cuh"
#include "groupby/hash/single_pass_reductions.cuh"
#include "peer_aggregate.hpp"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/utilities/bit.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cudf::groupby {
namespace {

namespace hash                = detail::hash;
namespace cg                  = cooperative_groups;
constexpr int peer_block_size = 256;

template <typename T, aggregation::Kind K>
constexpr bool peer_type_supported()
{
  if constexpr (cuda::std::is_same_v<T, numeric::decimal128> && K == aggregation::SUM) {
    return true;  // This kind has a dedicated existing 128-bit atomic addition.
  }
  if constexpr (hash::is_reduction_supported<T>(K) &&
                (K == aggregation::SUM || K == aggregation::SUM_OF_SQUARES ||
                 K == aggregation::PRODUCT || K == aggregation::MIN || K == aggregation::MAX)) {
    using Result = hash::rep_type_t<cudf::detail::target_type_t<T, K>>;
    return sizeof(hash::rep_type_t<T>) <= sizeof(int64_t) && sizeof(Result) <= sizeof(int64_t) &&
           cudf::has_atomic_support<Result>();
  }
  return false;
}

// Materialize host-only eligibility as a constant before device-context instantiation.
template <typename T, aggregation::Kind K>
inline constexpr bool peer_type_supported_v = peer_type_supported<T, K>();

struct peer_supported_dispatch {
  template <typename T, aggregation::Kind K>
  bool operator()() const
  {
    return peer_type_supported<T, K>();
  }
};

template <typename ValueFn>
struct peer_values {
  ValueFn values;
  __device__ bool valid(size_type row) const { return !values.col_is_null(row); }
  __device__ auto value(size_type row) const { return values.compute(row); }
};

template <bool ValidOnly>
struct peer_count_values {
  column_device_view values;
  bool has_nulls;
  __device__ bool valid(size_type row) const
  {
    return !ValidOnly || !has_nulls || values.is_valid_nocheck(row);
  }
  __device__ size_type value(size_type) const { return 1; }
};

template <typename Values, typename Result>
struct peer_column {
  Values values;
  Result* output;
  bitmask_type* validity;
};

template <typename Column>
struct single_peer_column {
  Column column;
  __device__ Column const& operator[](size_type) const { return column; }
};

template <typename Column>
struct multiple_peer_columns {
  Column const* columns;
  __device__ Column const& operator[](size_type index) const { return columns[index]; }
};

// Public CG collectives shuffle this trivially-copyable representation without narrowing it.
// Unsigned word arithmetic keeps every carry and avoids signed overflow in the local reduction.
struct peer_sum128_value {
  uint64_t low;
  uint64_t high;
};
static_assert(sizeof(peer_sum128_value) == sizeof(__int128_t));
static_assert(cuda::std::is_trivially_copyable_v<peer_sum128_value>);

struct peer_sum128_op {
  __device__ peer_sum128_value operator()(peer_sum128_value lhs, peer_sum128_value rhs) const
  {
    auto const low = lhs.low + rhs.low;
    return {low, lhs.high + rhs.high + static_cast<uint64_t>(low < lhs.low)};
  }
};

template <typename T, typename Op>
__device__ T reduce_peer_value(cg::coalesced_group const& peers, T value, Op operation)
{
  if constexpr (cuda::std::is_same_v<T, __int128_t>) {
    static_assert(cuda::std::is_same_v<Op, cudf::DeviceSum>);
    auto const bits = static_cast<__uint128_t>(value);
    auto const packed =
      peer_sum128_value{static_cast<uint64_t>(bits), static_cast<uint64_t>(bits >> 64)};
    auto const result = cg::reduce(peers, packed, peer_sum128_op{});
    auto const output = (static_cast<__uint128_t>(result.high) << 64) | result.low;
    return cuda::std::bit_cast<__int128_t>(output);
  } else {
    return cg::reduce(peers, value, operation);
  }
}

template <typename Column, typename Op>
__device__ void aggregate_peer_row(Column const& column,
                                   size_type row,
                                   size_type target,
                                   Op operation)
{
  if (column.values.valid(row)) {
    auto const active = cg::coalesced_threads();
    auto const peers  = cg::labeled_partition(active, target);
    auto const value  = column.values.value(row);
    auto combine      = peers.size() > 1;
    if constexpr (cuda::std::is_floating_point_v<decltype(value)> &&
                  (cuda::std::is_same_v<Op, cudf::DeviceMin> ||
                   cuda::std::is_same_v<Op, cudf::DeviceMax>)) {
      if (combine) {
        // Existing MIN/MAX are order-sensitive for NaNs and signed-zero ties. Keep
        // every original atomic in those peer groups, without inventing a new policy.
        auto const exceptional = cuda::std::isnan(value) || value == 0;
        combine = cg::reduce(peers, static_cast<int>(exceptional), cg::plus<int>{}) == 0;
      }
    }
    auto const combined = combine ? reduce_peer_value(peers, value, operation) : value;
    if (!combine || peers.thread_rank() == 0) {
      if (column.validity != nullptr && !cudf::bit_is_set(column.validity, target)) {
        cudf::set_bit(column.validity, target);
      }
      if constexpr (cuda::std::is_same_v<cuda::std::remove_cv_t<decltype(combined)>, __int128_t>) {
        static_assert(cuda::std::is_same_v<Op, cudf::DeviceSum>);
        cudf::detail::atomic_add(column.output + target, combined);
      } else {
        cudf::detail::genericAtomicOperation(column.output + target, combined, operation);
      }
    }
  }
}

// A logical warp belongs to exactly one column, including when the last row tile is partial.
// Equal targets can be noncontiguous within the warp; labeled_partition combines all peers.
template <typename Columns, typename Op>
CUDF_KERNEL void __launch_bounds__(peer_block_size) aggregate_peers_kernel(size_type num_rows,
                                                                           size_type num_columns,
                                                                           size_type const* targets,
                                                                           Columns columns,
                                                                           Op operation)
{
  constexpr auto warp_size    = cudf::detail::warp_size;
  auto const tiles_per_column = (static_cast<int64_t>(num_rows) + warp_size - 1) / warp_size;
  auto const total_tiles      = tiles_per_column * num_columns;
  auto const global_thread    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto const lane             = threadIdx.x % warp_size;
  auto const tile_stride      = static_cast<int64_t>(gridDim.x) * (blockDim.x / warp_size);
  for (auto tile = global_thread / warp_size; tile < total_tiles; tile += tile_stride) {
    auto const row_wide = (tile % tiles_per_column) * warp_size + lane;
    if (row_wide < num_rows) {
      auto const row    = static_cast<size_type>(row_wide);
      auto const target = targets[row];
      if (target != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
        auto const& column = columns[static_cast<size_type>(tile / tiles_per_column)];
        aggregate_peer_row(column, row, target, operation);
      }
    }
    // Even excluded, null and tail lanes wait before advancing to another tile/column.
    // The loop bound is warp-uniform and all 32 lanes remain live at this barrier.
    __syncwarp();
  }
}

template <typename Columns, typename Op>
void launch_peer_kernel(size_type num_rows,
                        size_type num_columns,
                        size_type const* targets,
                        Columns columns,
                        Op operation,
                        cuda::stream_ref stream)
{
  if (num_rows == 0 || num_columns == 0) { return; }
  auto const tiles_per_column =
    (static_cast<int64_t>(num_rows) + cudf::detail::warp_size - 1) / cudf::detail::warp_size;
  auto const total_tiles         = tiles_per_column * num_columns;
  constexpr auto warps_per_block = peer_block_size / cudf::detail::warp_size;
  auto const needed_blocks       = (total_tiles + warps_per_block - 1) / warps_per_block;
  int active_blocks{};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks, aggregate_peers_kernel<Columns, Op>, peer_block_size, 0));
  auto const resident_blocks =
    static_cast<int64_t>(active_blocks) * cudf::detail::num_multiprocessors();
  auto const blocks = static_cast<int>(std::min(needed_blocks, resident_blocks));
  CUDF_EXPECTS(blocks > 0, "Peer aggregation kernel has no resident blocks.");
  aggregate_peers_kernel<<<blocks, peer_block_size, 0, stream.get()>>>(
    num_rows, num_columns, targets, columns, operation);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename Column, typename Op>
void launch_peer_columns(host_span<Column const> columns,
                         device_span<size_type const> targets,
                         Op operation,
                         cuda::stream_ref stream,
                         cudf::memory_resources mr)
{
  if (columns.size() == 1) {
    launch_peer_kernel(static_cast<size_type>(targets.size()),
                       1,
                       targets.data(),
                       single_peer_column<Column>{columns.front()},
                       operation,
                       stream);
  } else {
    cuda::device_buffer<Column> device_columns(
      stream, mr.get_temporary_mr(), columns.size(), cuda::no_init);
    // This synchronous helper keeps the pinned host descriptors alive until the transfer ends.
    cudf::detail::cuda_memcpy(
      device_span<Column>{device_columns.data(), device_columns.size()}, columns, stream);
    launch_peer_kernel(static_cast<size_type>(targets.size()),
                       static_cast<size_type>(columns.size()),
                       targets.data(),
                       multiple_peer_columns<Column>{device_columns.data()},
                       operation,
                       stream);
  }
}

template <aggregation::Kind K>
struct peer_dispatch {
  template <typename T>
    requires(peer_type_supported<T, K>())
  void operator()(table_view const& values,
                  mutable_table_view const& output,
                  size_type begin,
                  size_type end,
                  device_span<size_type const> targets,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr) const
  {
    using Source  = hash::rep_type_t<T>;
    using Result  = hash::rep_type_t<cudf::detail::target_type_t<T, K>>;
    using Op      = cudf::detail::corresponding_operator_t<K>;
    using ValueFn = hash::grouped_value_fn<Source, Result, K == aggregation::SUM_OF_SQUARES>;
    using Column  = peer_column<peer_values<ValueFn>, Result>;
    auto columns  = cudf::detail::make_empty_host_vector<Column>(end - begin, stream);
    using View =
      decltype(column_device_view::create(values.column(begin), stream, mr.get_temporary_mr()));
    std::vector<View> owners;
    owners.reserve(end - begin);
    for (auto index = begin; index < end; ++index) {
      auto const& input = values.column(index);
      owners.push_back(column_device_view::create(input, stream, mr.get_temporary_mr()));
      auto const& target = output.column(index);
      // Only the row-based helpers are used; they do not dereference the grouped-row map.
      ValueFn const row_values{nullptr,
                               detail::value_accessor<Source>{*owners.back()},
                               Op::template identity<Result>(),
                               input.has_nulls()};
      columns.push_back({{row_values}, target.template begin<Result>(), target.null_mask()});
    }
    launch_peer_columns(host_span<Column const>{columns}, targets, Op{}, stream, mr);
  }

  template <typename T>
    requires(!peer_type_supported<T, K>())
  void operator()(table_view const&,
                  mutable_table_view const&,
                  size_type,
                  size_type,
                  device_span<size_type const>,
                  cuda::stream_ref,
                  cudf::memory_resources) const
  {
    CUDF_FAIL("Unsupported type for peer aggregation");
  }
};

struct peer_kind_dispatch {
  template <aggregation::Kind K>
  void operator()(data_type type,
                  table_view const& values,
                  mutable_table_view const& output,
                  size_type begin,
                  size_type end,
                  device_span<size_type const> targets,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr) const
  {
    if constexpr (K == aggregation::SUM || K == aggregation::SUM_OF_SQUARES ||
                  K == aggregation::PRODUCT || K == aggregation::MIN || K == aggregation::MAX) {
      type_dispatcher(type, peer_dispatch<K>{}, values, output, begin, end, targets, stream, mr);
    } else {
      CUDF_FAIL("Unsupported kind for peer aggregation");
    }
  }
};

template <bool ValidOnly>
void compute_peer_counts(table_view const& values,
                         mutable_table_view const& output,
                         size_type begin,
                         size_type end,
                         device_span<size_type const> targets,
                         cuda::stream_ref stream,
                         cudf::memory_resources mr)
{
  using Column = peer_column<peer_count_values<ValidOnly>, size_type>;
  auto columns = cudf::detail::make_empty_host_vector<Column>(end - begin, stream);
  using View =
    decltype(column_device_view::create(values.column(begin), stream, mr.get_temporary_mr()));
  std::vector<View> owners;
  owners.reserve(end - begin);
  for (auto index = begin; index < end; ++index) {
    auto const& input = values.column(index);
    owners.push_back(column_device_view::create(input, stream, mr.get_temporary_mr()));
    columns.push_back({{*owners.back(), input.has_nulls()},
                       output.column(index).template begin<size_type>(),
                       nullptr});
  }
  // Counts merge partial counts with addition; DeviceCount is a per-element counting operator.
  launch_peer_columns(host_span<Column const>{columns}, targets, cudf::DeviceSum{}, stream, mr);
}

data_type decoded_type(column_view const& values)
{
  return cudf::is_dictionary(values.type()) ? dictionary_column_view(values).keys().type()
                                            : values.type();
}

// Only mixed-kind buckets use this descriptor. Homogeneous runs retain the existing
// typed descriptors and the single-column by-value kernel argument fast path.
template <typename T>
struct source_peer_column {
  detail::value_accessor<hash::rep_type_t<T>> input;
  void* output;
  bitmask_type* validity;
  aggregation::Kind kind;
  bool has_nulls;
};

template <typename Column, typename Op>
__device__ void aggregate_source_rows(size_type num_rows,
                                      size_type const* targets,
                                      Column column,
                                      Op operation,
                                      int local_block,
                                      int blocks_per_column)
{
  constexpr auto warp_size       = cudf::detail::warp_size;
  constexpr auto warps_per_block = peer_block_size / warp_size;
  auto const tiles               = (static_cast<int64_t>(num_rows) + warp_size - 1) / warp_size;
  auto const first  = static_cast<int64_t>(local_block) * warps_per_block + threadIdx.x / warp_size;
  auto const stride = static_cast<int64_t>(blocks_per_column) * warps_per_block;
  for (auto tile = first; tile < tiles; tile += stride) {
    auto const row_wide = tile * warp_size + threadIdx.x % warp_size;
    if (row_wide < num_rows) {
      auto const row    = static_cast<size_type>(row_wide);
      auto const target = targets[row];
      if (target != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
        aggregate_peer_row(column, row, target, operation);
      }
    }
    // The tile loop is warp-uniform, including invalid/excluded rows and its tail.
    __syncwarp();
  }
}

template <typename T, aggregation::Kind K>
__device__ void aggregate_source_column(size_type num_rows,
                                        size_type const* targets,
                                        source_peer_column<T> const& descriptor,
                                        int local_block,
                                        int blocks_per_column)
{
  if constexpr (K == aggregation::COUNT_ALL || K == aggregation::COUNT_VALID) {
    using Values = peer_count_values<K == aggregation::COUNT_VALID>;
    peer_column<Values, size_type> const column{{descriptor.input.col, descriptor.has_nulls},
                                                static_cast<size_type*>(descriptor.output),
                                                nullptr};
    aggregate_source_rows(
      num_rows, targets, column, cudf::DeviceSum{}, local_block, blocks_per_column);
  } else if constexpr (peer_type_supported_v<T, K>) {
    using Source  = hash::rep_type_t<T>;
    using Result  = hash::rep_type_t<cudf::detail::target_type_t<T, K>>;
    using Op      = cudf::detail::corresponding_operator_t<K>;
    using ValueFn = hash::grouped_value_fn<Source, Result, K == aggregation::SUM_OF_SQUARES>;
    ValueFn const values{
      nullptr, descriptor.input, Op::template identity<Result>(), descriptor.has_nulls};
    peer_column<peer_values<ValueFn>, Result> const column{
      {values}, static_cast<Result*>(descriptor.output), descriptor.validity};
    aggregate_source_rows(num_rows, targets, column, Op{}, local_block, blocks_per_column);
  }
}

template <typename T>
CUDF_KERNEL void __launch_bounds__(peer_block_size)
  aggregate_source_kernel(size_type num_rows,
                          size_type const* targets,
                          source_peer_column<T> const* columns,
                          int blocks_per_column)
{
  auto const column_index = blockIdx.x / blocks_per_column;
  auto const local_block  = static_cast<int>(blockIdx.x % blocks_per_column);
  auto const& column      = columns[column_index];
  // Every CTA selects one column and one compile-time operation before entering its row loop.
  // Different result promotions never share a typed output pointer or a peer collective.
  switch (column.kind) {
    case aggregation::SUM:
      aggregate_source_column<T, aggregation::SUM>(
        num_rows, targets, column, local_block, blocks_per_column);
      break;
    case aggregation::SUM_OF_SQUARES:
      aggregate_source_column<T, aggregation::SUM_OF_SQUARES>(
        num_rows, targets, column, local_block, blocks_per_column);
      break;
    case aggregation::PRODUCT:
      aggregate_source_column<T, aggregation::PRODUCT>(
        num_rows, targets, column, local_block, blocks_per_column);
      break;
    case aggregation::MIN:
      aggregate_source_column<T, aggregation::MIN>(
        num_rows, targets, column, local_block, blocks_per_column);
      break;
    case aggregation::MAX:
      aggregate_source_column<T, aggregation::MAX>(
        num_rows, targets, column, local_block, blocks_per_column);
      break;
    case aggregation::COUNT_VALID:
      aggregate_source_column<T, aggregation::COUNT_VALID>(
        num_rows, targets, column, local_block, blocks_per_column);
      break;
    case aggregation::COUNT_ALL:
      aggregate_source_column<T, aggregation::COUNT_ALL>(
        num_rows, targets, column, local_block, blocks_per_column);
      break;
    default: break;  // Host eligibility is identical to the ordinary peer path.
  }
}

struct source_fusion_dispatch {
  template <typename T>
  void operator()(table_view const& values,
                  mutable_table_view const& output,
                  host_span<aggregation::Kind const> kinds,
                  host_span<size_type const> indices,
                  device_span<size_type const> targets,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr) const
  {
    if (targets.empty()) { return; }
    auto columns =
      cudf::detail::make_empty_host_vector<source_peer_column<T>>(indices.size(), stream);
    using View = decltype(column_device_view::create(
      values.column(indices.front()), stream, mr.get_temporary_mr()));
    std::vector<View> owners;
    owners.reserve(indices.size());
    for (auto const index : indices) {
      auto const& input = values.column(index);
      owners.push_back(column_device_view::create(input, stream, mr.get_temporary_mr()));
      auto const& result = output.column(index);
      columns.push_back({detail::value_accessor<hash::rep_type_t<T>>{*owners.back()},
                         result.head(),
                         result.null_mask(),
                         kinds[index],
                         input.has_nulls()});
    }
    cuda::device_buffer<source_peer_column<T>> device_columns(
      stream, mr.get_temporary_mr(), columns.size(), cuda::no_init);
    // Keep both the pinned descriptors and their device-view owners alive through the copy.
    cudf::detail::cuda_memcpy(
      device_span<source_peer_column<T>>{device_columns.data(), device_columns.size()},
      host_span<source_peer_column<T> const>{columns},
      stream);
    int active_blocks{};
    CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, aggregate_source_kernel<T>, peer_block_size, 0));
    auto const resident_blocks =
      static_cast<int64_t>(active_blocks) * cudf::detail::num_multiprocessors();
    CUDF_EXPECTS(resident_blocks > 0, "Source fusion kernel has no resident blocks.");
    auto const needed_blocks =
      (static_cast<int64_t>(targets.size()) + peer_block_size - 1) / peer_block_size;
    auto const blocks_per_column = static_cast<int>(std::min(
      needed_blocks, std::max<int64_t>(1, resident_blocks / static_cast<int64_t>(indices.size()))));
    auto const blocks = static_cast<int>(static_cast<int64_t>(blocks_per_column) * indices.size());
    aggregate_source_kernel<T>
      <<<blocks, peer_block_size, 0, stream.get()>>>(static_cast<size_type>(targets.size()),
                                                     targets.data(),
                                                     device_columns.data(),
                                                     blocks_per_column);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
};

CUDF_KERNEL void __launch_bounds__(peer_block_size)
  aggregate_scalar_run_kernel(int64_t num_items,
                              int64_t index_offset,
                              hash::compute_single_pass_aggs_dense_output_fn aggregate)
{
  auto const first  = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (auto item = first; item < num_items; item += stride) {
    aggregate(index_offset + item);
  }
}

void compute_scalar_run(size_type begin,
                        size_type end,
                        device_span<size_type const> targets,
                        aggregation::Kind const* device_kinds,
                        table_device_view device_values,
                        mutable_table_device_view device_output,
                        cuda::stream_ref stream)
{
  auto const num_items = static_cast<int64_t>(end - begin) * static_cast<int64_t>(targets.size());
  if (num_items == 0) { return; }
  int active_blocks{};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks, aggregate_scalar_run_kernel, peer_block_size, 0));
  auto const resident_blocks =
    static_cast<int64_t>(active_blocks) * cudf::detail::num_multiprocessors();
  auto const needed_blocks = (num_items + peer_block_size - 1) / peer_block_size;
  auto const blocks        = static_cast<int>(std::min(needed_blocks, resident_blocks));
  CUDF_EXPECTS(blocks > 0, "Scalar aggregation kernel has no resident blocks.");
  aggregate_scalar_run_kernel<<<blocks, peer_block_size, 0, stream.get()>>>(
    num_items,
    static_cast<int64_t>(begin) * static_cast<int64_t>(targets.size()),
    hash::compute_single_pass_aggs_dense_output_fn{
      targets.data(), device_kinds, device_values, device_output});
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace

void compute_peer_aggs(table_view const& values,
                       host_span<aggregation::Kind const> kinds,
                       aggregation::Kind const* device_kinds,
                       device_span<size_type const> targets,
                       mutable_table_view const& output,
                       mutable_table_device_view device_output,
                       cuda::stream_ref stream,
                       cudf::memory_resources mr)
{
  CUDF_EXPECTS(values.num_columns() == static_cast<size_type>(kinds.size()) &&
                 output.num_columns() == values.num_columns(),
               "Invalid peer aggregation metadata");
  std::vector<bool> supported(values.num_columns());
  for (size_type i = 0; i < values.num_columns(); ++i) {
    supported[i] = kinds[i] == aggregation::COUNT_ALL || kinds[i] == aggregation::COUNT_VALID ||
                   cudf::detail::dispatch_type_and_aggregation(
                     decoded_type(values.column(i)), kinds[i], peer_supported_dispatch{});
  }
  std::vector<bool> fused(values.num_columns(), false);
  std::vector<bool> visited(values.num_columns(), false);
  for (size_type first = 0; first < values.num_columns(); ++first) {
    if (!supported[first] || visited[first]) { continue; }
    auto const type = decoded_type(values.column(first));
    std::vector<size_type> indices;
    bool mixed_kinds = false;
    for (size_type i = first; i < values.num_columns(); ++i) {
      if (supported[i] && decoded_type(values.column(i)) == type) {
        indices.push_back(i);
        visited[i]  = true;
        mixed_kinds = mixed_kinds || kinds[i] != kinds[first];
      }
    }
    if (!mixed_kinds) { continue; }
    type_dispatcher(type,
                    source_fusion_dispatch{},
                    values,
                    output,
                    kinds,
                    host_span<size_type const>{indices},
                    targets,
                    stream,
                    mr);
    for (auto const index : indices) {
      fused[index] = true;
    }
  }
  using DeviceValues = decltype(table_device_view::create(values, stream, mr.get_temporary_mr()));
  DeviceValues fallback_values;
  for (size_type begin = 0; begin < values.num_columns();) {
    if (fused[begin]) {
      ++begin;
      continue;
    }
    auto const kind  = kinds[begin];
    auto const type  = decoded_type(values.column(begin));
    auto const count = kind == aggregation::COUNT_ALL || kind == aggregation::COUNT_VALID;
    auto end         = begin + 1;
    if (!supported[begin]) {
      while (end < values.num_columns() && !supported[end]) {
        ++end;
      }
      if (!fallback_values) {
        fallback_values = table_device_view::create(values, stream, mr.get_temporary_mr());
      }
      compute_scalar_run(
        begin, end, targets, device_kinds, *fallback_values, device_output, stream);
    } else {
      while (end < values.num_columns() && !fused[end] && supported[end] && kinds[end] == kind &&
             (count || decoded_type(values.column(end)) == type)) {
        ++end;
      }
      if (kind == aggregation::COUNT_ALL) {
        compute_peer_counts<false>(values, output, begin, end, targets, stream, mr);
      } else if (kind == aggregation::COUNT_VALID) {
        compute_peer_counts<true>(values, output, begin, end, targets, stream, mr);
      } else {
        cudf::detail::aggregation_dispatcher(
          kind, peer_kind_dispatch{}, type, values, output, begin, end, targets, stream, mr);
      }
    }
    begin = end;
  }
}

}  // namespace cudf::groupby
