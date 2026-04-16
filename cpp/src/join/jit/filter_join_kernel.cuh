/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/types.hpp>

#include <jit/span.cuh>

namespace cudf::join::jit {

/**
 * @brief JIT kernel for filtering join indices based on predicate
 *
 * @tparam has_user_data Whether the predicate function requires user data
 * @tparam is_null_aware Whether the expression needs input validity as part of its computation
 * @tparam InputAccessors Variadic template for input column accessors
 * @param left_indices Device span of left table indices
 * @param right_indices Device span of right table indices
 * @param left_tables Device view of left table columns
 * @param right_tables Device view of right table columns
 * @param predicate_results Output array for predicate evaluation results
 * @param user_data Optional user data for predicate function
 */
template <bool has_user_data, null_aware is_null_aware, typename... InputAccessors>
CUDF_KERNEL void filter_join_kernel(cudf::jit::device_span<cudf::size_type const> left_indices,
                                    cudf::jit::device_span<cudf::size_type const> right_indices,
                                    cudf::column_device_view_core const* left_tables,
                                    cudf::column_device_view_core const* right_tables,
                                    bool* predicate_results,
                                    void* user_data);

}  // namespace cudf::join::jit
