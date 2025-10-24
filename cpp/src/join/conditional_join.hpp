/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "join_common_utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <optional>

namespace cudf {
namespace detail {

/**
 * @brief Computes the join operation between two tables and returns the
 * output indices of left and right table as a combined table
 *
 * @param left  Table of left columns to join
 * @param right Table of right  columns to join
 * tables have been flipped, meaning the output indices should also be flipped
 * @param JoinKind The type of join to be performed
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Join output indices vector pair
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_join(table_view const& left,
                 table_view const& right,
                 ast::expression const& binary_predicate,
                 join_kind JoinKind,
                 std::optional<std::size_t> output_size,
                 rmm::cuda_stream_view stream,
                 rmm::device_async_resource_ref mr);

/**
 * @brief Computes the size of a join operation between two tables without
 * materializing the result and returns the total size value.
 *
 * @param left  Table of left columns to join
 * @param right Table of right  columns to join
 * tables have been flipped, meaning the output indices should also be flipped
 * @param JoinKind The type of join to be performed
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Join output indices vector pair
 */
std::size_t compute_conditional_join_output_size(table_view const& left,
                                                 table_view const& right,
                                                 ast::expression const& binary_predicate,
                                                 join_kind JoinKind,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
