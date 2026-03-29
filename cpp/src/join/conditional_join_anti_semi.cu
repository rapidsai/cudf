/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/conditional_join.hpp"
#include "join/conditional_join_kernels.cuh"
#include "join/join_common_utils.hpp"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <optional>

namespace cudf {
namespace detail {
namespace {
constexpr int DEFAULT_CACHE_SIZE = 128;
}

std::unique_ptr<rmm::device_uvector<size_type>> conditional_join_anti_semi(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  join_kind join_type,
  std::optional<std::size_t> output_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (right.num_rows() == 0) {
    switch (join_type) {
      case join_kind::LEFT_ANTI_JOIN: return get_trivial_left_join_indices(left, stream, mr).first;
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left.num_rows() == 0) {
    switch (join_type) {
      case join_kind::LEFT_ANTI_JOIN: [[fallthrough]];
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  }

  auto const has_nulls = binary_predicate.may_evaluate_null(left, right, stream);

  auto const parser =
    ast::detail::expression_parser{binary_predicate, left, right, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a Boolean output.");

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  detail::grid_1d const config(left.num_rows(), DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  // TODO: Remove the output_size parameter. It is not needed because the
  // output size is bounded by the size of the left table.
  std::size_t join_size;
  if (output_size.has_value()) {
    join_size = *output_size;
  } else {
    // Allocate storage for the counter used to get the size of the join output
    cudf::detail::device_scalar<std::size_t> size(0, stream, mr);
    if (has_nulls) {
      compute_conditional_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, true>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_table, *right_table, join_type, parser.device_expression_data, false, size.data());
    } else {
      compute_conditional_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, false>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_table, *right_table, join_type, parser.device_expression_data, false, size.data());
    }
    join_size = size.value(stream);
  }

  cudf::detail::device_scalar<std::size_t> write_index(0, stream);

  auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const& join_output_l = left_indices->data();

  if (has_nulls) {
    conditional_join_anti_semi<DEFAULT_JOIN_BLOCK_SIZE, DEFAULT_CACHE_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        join_type,
        join_output_l,
        write_index.data(),
        parser.device_expression_data,
        join_size);
  } else {
    conditional_join_anti_semi<DEFAULT_JOIN_BLOCK_SIZE, DEFAULT_CACHE_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        join_type,
        join_output_l,
        write_index.data(),
        parser.device_expression_data,
        join_size);
  }
  return left_indices;
}

}  // namespace detail

std::unique_ptr<rmm::device_uvector<size_type>> conditional_left_semi_join(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  std::optional<std::size_t> output_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::conditional_join_anti_semi(
    left, right, binary_predicate, join_kind::LEFT_SEMI_JOIN, output_size, stream, mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> conditional_left_anti_join(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  std::optional<std::size_t> output_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::conditional_join_anti_semi(
    left, right, binary_predicate, join_kind::LEFT_ANTI_JOIN, output_size, stream, mr);
}

std::size_t conditional_left_semi_join_size(table_view const& left,
                                            table_view const& right,
                                            ast::expression const& binary_predicate,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_conditional_join_output_size(
    left, right, binary_predicate, join_kind::LEFT_SEMI_JOIN, stream, mr);
}

std::size_t conditional_left_anti_join_size(table_view const& left,
                                            table_view const& right,
                                            ast::expression const& binary_predicate,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_conditional_join_output_size(
    left, right, binary_predicate, join_kind::LEFT_ANTI_JOIN, stream, mr);
}

}  // namespace cudf
