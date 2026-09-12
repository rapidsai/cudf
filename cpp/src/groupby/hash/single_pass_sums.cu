/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "single_pass_reductions.cuh"

namespace cudf::groupby::detail::hash::single_pass {

template std::unique_ptr<column> compute_reduction<aggregation::SUM>(reduction_context const& ctx);
template std::unique_ptr<column> compute_reduction<aggregation::SUM_OF_SQUARES>(
  reduction_context const& ctx);

std::vector<std::unique_ptr<column>> compute_fused_sums(reduction_context const& ctx,
                                                        host_span<aggregation::Kind const> kinds,
                                                        std::span<int8_t const> is_intermediate)
{
  return type_dispatcher(ctx.values_type, fused_sums_fn{}, ctx, kinds, is_intermediate);
}

}  // namespace cudf::groupby::detail::hash::single_pass
