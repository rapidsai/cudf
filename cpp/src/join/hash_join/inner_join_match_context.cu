/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "match_context_impl.cuh"

#include <cudf/detail/nvtx/ranges.hpp>

#include <rmm/device_uvector.hpp>

namespace cudf::detail {

template <typename Hasher>
cudf::join_match_context hash_join<Hasher>::inner_join_match_context(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"hash_join::inner_join_match_context"};
  return this->template join_match_context_impl<join_kind::INNER_JOIN>(probe, stream, mr);
}

template cudf::join_match_context hash_join<hash_join_hasher>::inner_join_match_context(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const;

}  // namespace cudf::detail
