/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/hashing/detail/hashing.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

#include "murmurhash3_x86_32_jit_launch.hpp"

namespace cudf {
namespace hashing {
namespace detail {

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  return murmurhash3_x86_32_jit(input, seed, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::murmurhash3_x86_32(input, seed, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
