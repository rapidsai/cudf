/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/execution>
#include <cuda/stream>

#include <type_traits>

namespace cudf {
namespace reduction {
namespace detail {
/**
 * @brief Make an execution environment to pass to CUB reductions.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used for internal allocations in the reduction.
 *
 * @returns A new execution environment.
 */
inline auto make_env(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto stream_property   = cuda::std::execution::prop{cuda::get_stream_t{}, stream};
  auto resource_property = cuda::std::execution::prop{cuda::mr::get_memory_resource_t{}, mr};
  return cuda::std::execution::env{stream_property, resource_property};
}

/**
 * @brief A custom tuning policy for the block reductions for one-pass variance computation.
 * @tparam State type of the state being reduced over.
 * @tparam BinaryOp type of the binary operator used in the reduction.
 *
 * @note This policy preserves CUB's architecture-specific tuning, changing only the
 * collective used to combine expensive intermediate states for variance.
 */
template <typename State, typename BinaryOp>
struct variance_reduce_policy {
  __host__ __device__ constexpr cub::ReducePolicy operator()(cuda::compute_capability cc) const
  {
    // TODO: Update to public CUB facilities using policy transformers once that exists.
    auto policy = cub::detail::reduce::
      policy_selector_from_types<State, std::make_unsigned_t<size_type>, BinaryOp>{}(cc);
    // Combining the states for variance is quite expensive. By using raking block reduce,
    // we use significantly less collective FP64 arithmetic than CUB's default policy.
    policy.multi_tile.reduce_algorithm  = cub::BLOCK_REDUCE_RAKING;
    policy.single_tile.reduce_algorithm = cub::BLOCK_REDUCE_RAKING;
    return policy;
  }
};

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
