/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/hash/single_pass_reductions.cuh"
#include "warp_aggregate.hpp"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>

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

// All lanes in a logical warp process the same row tile. Equal targets can be
// noncontiguous within that tile; labeled_partition combines all of their peers.
template <typename Column, typename Op>
CUDF_KERNEL void __launch_bounds__(peer_block_size)
  aggregate_peers_kernel(size_type num_rows, size_type const* targets, Column column, Op operation)
{
  constexpr auto warp_size = cudf::detail::warp_size;
  auto const num_tiles     = (static_cast<int64_t>(num_rows) + warp_size - 1) / warp_size;
  auto const global_thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto const lane          = threadIdx.x % warp_size;
  auto const tile_stride   = static_cast<int64_t>(gridDim.x) * (blockDim.x / warp_size);
  for (auto tile = global_thread / warp_size; tile < num_tiles; tile += tile_stride) {
    auto const row_wide = tile * warp_size + lane;
    if (row_wide < num_rows) {
      auto const row    = static_cast<size_type>(row_wide);
      auto const target = targets[row];
      if (target != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
        aggregate_peer_row(column, row, target, operation);
      }
    }
    // Excluded, null and tail lanes also wait before advancing to another tile.
    __syncwarp();
  }
}

template <typename Column, typename Op>
void launch_peer_kernel(device_span<size_type const> targets,
                        Column column,
                        Op operation,
                        cuda::stream_ref stream)
{
  if (targets.empty()) { return; }
  auto const needed_blocks =
    (static_cast<int64_t>(targets.size()) + peer_block_size - 1) / peer_block_size;
  int active_blocks{};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks, aggregate_peers_kernel<Column, Op>, peer_block_size, 0));
  auto const resident_blocks =
    static_cast<int64_t>(active_blocks) * cudf::detail::num_multiprocessors();
  auto const blocks = static_cast<int>(std::min(needed_blocks, resident_blocks));
  CUDF_EXPECTS(blocks > 0, "Peer aggregation kernel has no resident blocks.");
  aggregate_peers_kernel<<<blocks, peer_block_size, 0, stream.get()>>>(
    static_cast<size_type>(targets.size()), targets.data(), column, operation);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <aggregation::Kind K>
struct peer_dispatch {
  template <typename T>
    requires(peer_type_supported<T, K>())
  void operator()(column_view const& input,
                  mutable_column_view const& output,
                  device_span<size_type const> targets,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr) const
  {
    using Source      = hash::rep_type_t<T>;
    using Result      = hash::rep_type_t<cudf::detail::target_type_t<T, K>>;
    using Op          = cudf::detail::corresponding_operator_t<K>;
    using ValueFn     = hash::grouped_value_fn<Source, Result, K == aggregation::SUM_OF_SQUARES>;
    auto const values = column_device_view::create(input, stream, mr.get_temporary_mr());
    // These row-based helpers never dereference the grouped-row map.
    ValueFn const row_values{nullptr,
                             detail::value_accessor<Source>{*values},
                             Op::template identity<Result>(),
                             input.has_nulls()};
    peer_column<peer_values<ValueFn>, Result> const column{
      {row_values}, output.template begin<Result>(), output.null_mask()};
    launch_peer_kernel(targets, column, Op{}, stream);
  }

  template <typename T>
    requires(!peer_type_supported<T, K>())
  void operator()(column_view const&,
                  mutable_column_view const&,
                  device_span<size_type const>,
                  cuda::stream_ref,
                  cudf::memory_resources) const
  {
    CUDF_FAIL("Unsupported type for peer aggregation");
  }
};

struct peer_kind_dispatch {
  template <aggregation::Kind K>
  void operator()(column_view const& input,
                  mutable_column_view const& output,
                  device_span<size_type const> targets,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr) const
  {
    if constexpr (K == aggregation::COUNT_ALL || K == aggregation::COUNT_VALID) {
      using Column      = peer_column<peer_count_values<K == aggregation::COUNT_VALID>, size_type>;
      auto const values = column_device_view::create(input, stream, mr.get_temporary_mr());
      Column const column{
        {*values, input.has_nulls()}, output.template begin<size_type>(), nullptr};
      // Counts combine row contributions with addition, including COUNT_ALL on null rows.
      launch_peer_kernel(targets, column, cudf::DeviceSum{}, stream);
    } else if constexpr (K == aggregation::SUM || K == aggregation::SUM_OF_SQUARES ||
                         K == aggregation::PRODUCT || K == aggregation::MIN ||
                         K == aggregation::MAX) {
      auto const type = cudf::is_dictionary(input.type())
                          ? dictionary_column_view(input).keys().type()
                          : input.type();
      type_dispatcher(type, peer_dispatch<K>{}, input, output, targets, stream, mr);
    } else {
      CUDF_FAIL("Unsupported kind for peer aggregation");
    }
  }
};

}  // namespace

void compute_warp_aggregates(table_view const& values,
                             host_span<aggregation::Kind const> kinds,
                             device_span<size_type const> targets,
                             mutable_table_view const& output,
                             cuda::stream_ref stream,
                             cudf::memory_resources mr)
{
  CUDF_EXPECTS(values.num_columns() == static_cast<size_type>(kinds.size()) &&
                 output.num_columns() == values.num_columns(),
               "Invalid peer aggregation metadata");
  for (size_type i = 0; i < values.num_columns(); ++i) {
    cudf::detail::aggregation_dispatcher(
      kinds[i], peer_kind_dispatch{}, values.column(i), output.column(i), targets, stream, mr);
  }
}

}  // namespace cudf::groupby
