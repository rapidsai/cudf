/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/hash/single_pass_reductions.cuh"
#include "warp_aggregate.hpp"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include <cstdint>

namespace cudf::groupby {
namespace {

namespace hash           = detail::hash;
namespace cg             = cooperative_groups;
constexpr int block_size = 256;

template <typename T, aggregation::Kind K>
constexpr bool is_warp_aggregation_supported()
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
struct aggregation_values {
  ValueFn values;
  __device__ bool valid(size_type row) const { return !values.col_is_null(row); }
  __device__ auto value(size_type row) const { return values.compute(row); }
};

template <bool ValidOnly>
struct count_values {
  column_device_view values;
  bool has_nulls;
  __device__ bool valid(size_type row) const
  {
    return !ValidOnly || !has_nulls || values.is_valid_nocheck(row);
  }
  __device__ size_type value(size_type) const { return 1; }
};

template <typename Values, typename Result>
struct aggregation_column {
  Values values;
  Result* output;
  bitmask_type* validity;
};

// Public CG collectives shuffle this trivially-copyable representation without narrowing it.
// Unsigned word arithmetic keeps every carry and avoids signed overflow in the local reduction.
struct int128_words {
  uint64_t low;
  uint64_t high;
};
static_assert(sizeof(int128_words) == sizeof(__int128_t));
static_assert(cuda::std::is_trivially_copyable_v<int128_words>);

struct int128_sum {
  __device__ int128_words operator()(int128_words lhs, int128_words rhs) const
  {
    auto const low = lhs.low + rhs.low;
    return {low, lhs.high + rhs.high + static_cast<uint64_t>(low < lhs.low)};
  }
};

template <typename T, typename Op>
__device__ T reduce_group_value(cg::coalesced_group const& thread_group, T value, Op operation)
{
  if constexpr (cuda::std::is_same_v<T, __int128_t>) {
    static_assert(cuda::std::is_same_v<Op, cudf::DeviceSum>);
    auto const bits = static_cast<__uint128_t>(value);
    auto const packed =
      int128_words{static_cast<uint64_t>(bits), static_cast<uint64_t>(bits >> 64)};
    auto const result = cg::reduce(thread_group, packed, int128_sum{});
    auto const output = (static_cast<__uint128_t>(result.high) << 64) | result.low;
    return cuda::std::bit_cast<__int128_t>(output);
  } else {
    return cg::reduce(thread_group, value, operation);
  }
}

template <typename Column, typename Op>
__device__ void aggregate_row(Column const& column, size_type row, size_type target, Op operation)
{
  if (column.values.valid(row)) {
    auto const active       = cg::coalesced_threads();
    auto const thread_group = cg::labeled_partition(active, target);
    auto const value        = column.values.value(row);
    auto combine            = thread_group.size() > 1;
    if constexpr (cuda::std::is_floating_point_v<decltype(value)> &&
                  (cuda::std::is_same_v<Op, cudf::DeviceMin> ||
                   cuda::std::is_same_v<Op, cudf::DeviceMax>)) {
      if (combine) {
        // Existing MIN/MAX are order-sensitive for NaNs and signed-zero ties. Keep
        // every original atomic in those thread groups.
        auto const exceptional = cuda::std::isnan(value) || value == 0;
        combine = cg::reduce(thread_group, static_cast<int>(exceptional), cg::plus<int>{}) == 0;
      }
    }
    auto const combined = combine ? reduce_group_value(thread_group, value, operation) : value;
    if (!combine || thread_group.thread_rank() == 0) {
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

// Equal targets can be noncontiguous within a warp; labeled_partition groups lanes with the same
// target.
template <typename Column, typename Op>
CUDF_KERNEL void __launch_bounds__(block_size)
  aggregate_warp_kernel(size_type num_rows, size_type const* targets, Column column, Op operation)
{
  auto const row = cudf::detail::grid_1d::global_thread_id();
  if (row >= num_rows) { return; }
  auto const target = targets[row];
  if (target != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
    aggregate_row(column, static_cast<size_type>(row), target, operation);
  }
}

template <typename Column, typename Op>
void launch_warp_kernel(device_span<size_type const> targets,
                        Column column,
                        Op operation,
                        cuda::stream_ref stream)
{
  auto const num_rows = static_cast<size_type>(targets.size());
  cudf::detail::grid_1d const grid{num_rows, block_size};
  aggregate_warp_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
    num_rows, targets.data(), column, operation);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <aggregation::Kind K>
struct aggregation_dispatch {
  template <typename T>
    requires(is_warp_aggregation_supported<T, K>())
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
    aggregation_column<aggregation_values<ValueFn>, Result> const column{
      {row_values}, output.template begin<Result>(), output.null_mask()};
    launch_warp_kernel(targets, column, Op{}, stream);
  }

  template <typename T>
    requires(!is_warp_aggregation_supported<T, K>())
  void operator()(column_view const&,
                  mutable_column_view const&,
                  device_span<size_type const>,
                  cuda::stream_ref,
                  cudf::memory_resources) const
  {
    CUDF_FAIL("Unsupported type for warp aggregation");
  }
};

struct aggregation_kind_dispatch {
  template <aggregation::Kind K>
  void operator()(column_view const& input,
                  mutable_column_view const& output,
                  device_span<size_type const> targets,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr) const
  {
    if constexpr (K == aggregation::COUNT_ALL || K == aggregation::COUNT_VALID) {
      using Column = aggregation_column<count_values<K == aggregation::COUNT_VALID>, size_type>;
      auto const values = column_device_view::create(input, stream, mr.get_temporary_mr());
      Column const column{
        {*values, input.has_nulls()}, output.template begin<size_type>(), nullptr};
      // Counts combine row contributions with addition, including COUNT_ALL on null rows.
      launch_warp_kernel(targets, column, cudf::DeviceSum{}, stream);
    } else if constexpr (K == aggregation::SUM || K == aggregation::SUM_OF_SQUARES ||
                         K == aggregation::PRODUCT || K == aggregation::MIN ||
                         K == aggregation::MAX) {
      auto const type = cudf::is_dictionary(input.type())
                          ? dictionary_column_view(input).keys().type()
                          : input.type();
      type_dispatcher(type, aggregation_dispatch<K>{}, input, output, targets, stream, mr);
    } else {
      CUDF_FAIL("Unsupported kind for warp aggregation");
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
  for (size_type i = 0; i < values.num_columns(); ++i) {
    cudf::detail::aggregation_dispatcher(kinds[i],
                                         aggregation_kind_dispatch{},
                                         values.column(i),
                                         output.column(i),
                                         targets,
                                         stream,
                                         mr);
  }
}

}  // namespace cudf::groupby
