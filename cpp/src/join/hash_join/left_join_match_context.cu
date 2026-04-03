/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join/join.hpp>

namespace cudf::detail {

template <typename Hasher>
cudf::join_match_context hash_join<Hasher>::left_join_match_context(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"hash_join::left_join_match_context"};
  return cudf::join_match_context{probe,
                                  make_join_match_counts(_build,
                                                         _preprocessed_build,
                                                         _hash_table,
                                                         _is_empty,
                                                         _has_nulls,
                                                         _nulls_equal,
                                                         join_kind::LEFT_JOIN,
                                                         probe,
                                                         stream,
                                                         mr)};
}

template cudf::join_match_context hash_join<hash_join_hasher>::left_join_match_context(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const;

}  // namespace cudf::detail
