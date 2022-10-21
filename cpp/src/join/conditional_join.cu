/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <join/conditional_join.hpp>
#include <join/conditional_join_kernels.cuh>
#include <join/join_common_utils.cuh>
#include <join/join_common_utils.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <optional>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_join(table_view const& left,
                 table_view const& right,
                 ast::expression const& binary_predicate,
                 join_kind join_type,
                 std::optional<std::size_t> output_size,
                 rmm::cuda_stream_view stream,
                 rmm::mr::device_memory_resource* mr)
{
  // We can immediately filter out cases where the right table is empty. In
  // some cases, we return all the rows of the left table with a corresponding
  // null index for the right table; in others, we return an empty output.
  auto right_num_rows{right.num_rows()};
  auto left_num_rows{left.num_rows()};
  if (right_num_rows == 0) {
    switch (join_type) {
      // Left, left anti, and full all return all the row indices from left
      // with a corresponding NULL from the right.
      case join_kind::LEFT_JOIN:
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::FULL_JOIN: return get_trivial_left_join_indices(left, stream);
      // Inner and left semi joins return empty output because no matches can exist.
      case join_kind::INNER_JOIN:
      case join_kind::LEFT_SEMI_JOIN:
        return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                         std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left_num_rows == 0) {
    switch (join_type) {
      // Left, left anti, left semi, and inner joins all return empty sets.
      case join_kind::LEFT_JOIN:
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::INNER_JOIN:
      case join_kind::LEFT_SEMI_JOIN:
        return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                         std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
      // Full joins need to return the trivial complement.
      case join_kind::FULL_JOIN: {
        auto ret_flipped = get_trivial_left_join_indices(right, stream);
        return std::pair(std::move(ret_flipped.second), std::move(ret_flipped.first));
      }
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  }

  // If evaluating the expression may produce null outputs we create a nullable
  // output column and follow the null-supporting expression evaluation code
  // path.
  auto const has_nulls = binary_predicate.may_evaluate_null(left, right, stream);

  auto const parser =
    ast::detail::expression_parser{binary_predicate, left, right, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.");

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  // For inner joins we support optimizing the join by launching one thread for
  // whichever table is larger rather than always using the left table.
  auto swap_tables = (join_type == join_kind::INNER_JOIN) && (right_num_rows > left_num_rows);
  detail::grid_1d const config(swap_tables ? right_num_rows : left_num_rows,
                               DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;
  join_kind const kernel_join_type =
    join_type == join_kind::FULL_JOIN ? join_kind::LEFT_JOIN : join_type;

  // If the join size was not provided as an input, compute it here.
  std::size_t join_size;
  if (output_size.has_value()) {
    join_size = *output_size;
  } else {
    // Allocate storage for the counter used to get the size of the join output
    rmm::device_scalar<std::size_t> size(0, stream, mr);
    if (has_nulls) {
      compute_conditional_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, true>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_table,
          *right_table,
          kernel_join_type,
          parser.device_expression_data,
          swap_tables,
          size.data());
    } else {
      compute_conditional_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, false>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_table,
          *right_table,
          kernel_join_type,
          parser.device_expression_data,
          swap_tables,
          size.data());
    }
    join_size = size.value(stream);
  }

  // The initial early exit clauses guarantee that we will not reach this point
  // unless both the left and right tables are non-empty. Under that
  // constraint, neither left nor full joins can return an empty result since
  // at minimum we are guaranteed null matches for all non-matching rows. In
  // all other cases (inner, left semi, and left anti joins) if we reach this
  // point we can safely return an empty result.
  if (join_size == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  rmm::device_scalar<size_type> write_index(0, stream);

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const& join_output_l = left_indices->data();
  auto const& join_output_r = right_indices->data();
  if (has_nulls) {
    conditional_join<DEFAULT_JOIN_BLOCK_SIZE, DEFAULT_JOIN_CACHE_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        kernel_join_type,
        join_output_l,
        join_output_r,
        write_index.data(),
        parser.device_expression_data,
        join_size,
        swap_tables);
  } else {
    conditional_join<DEFAULT_JOIN_BLOCK_SIZE, DEFAULT_JOIN_CACHE_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        kernel_join_type,
        join_output_l,
        join_output_r,
        write_index.data(),
        parser.device_expression_data,
        join_size,
        swap_tables);
  }

  auto join_indices = std::pair(std::move(left_indices), std::move(right_indices));

  // For full joins, get the indices in the right table that were not joined to
  // by any row in the left table.
  if (join_type == join_kind::FULL_JOIN) {
    auto complement_indices = detail::get_left_join_indices_complement(
      join_indices.second, left_num_rows, right_num_rows, stream, mr);
    join_indices = detail::concatenate_vector_pairs(join_indices, complement_indices, stream);
  }
  return join_indices;
}

std::size_t compute_conditional_join_output_size(table_view const& left,
                                                 table_view const& right,
                                                 ast::expression const& binary_predicate,
                                                 join_kind join_type,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  // Until we add logic to handle the number of non-matches in the right table,
  // full joins are not supported in this function. Note that this does not
  // prevent actually performing full joins since we do that by calculating the
  // left join and then concatenating the complementary right indices.
  CUDF_EXPECTS(join_type != join_kind::FULL_JOIN,
               "Size estimation is not available for full joins.");

  // We can immediately filter out cases where one table is empty. In
  // some cases, we return all the rows of the other table with a corresponding
  // null index for the empty table; in others, we return an empty output.
  auto right_num_rows{right.num_rows()};
  auto left_num_rows{left.num_rows()};
  if (right_num_rows == 0) {
    switch (join_type) {
      // Left, left anti, and full all return all the row indices from left
      // with a corresponding NULL from the right.
      case join_kind::LEFT_JOIN:
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::FULL_JOIN: return left_num_rows;
      // Inner and left semi joins return empty output because no matches can exist.
      case join_kind::INNER_JOIN:
      case join_kind::LEFT_SEMI_JOIN: return 0;
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left_num_rows == 0) {
    switch (join_type) {
      // Left, left anti, left semi, and inner joins all return empty sets.
      case join_kind::LEFT_JOIN:
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::INNER_JOIN:
      case join_kind::LEFT_SEMI_JOIN: return 0;
      // Full joins need to return the trivial complement.
      case join_kind::FULL_JOIN: return right_num_rows;
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  }

  // Prepare output column. Whether or not the output column is nullable is
  // determined by whether any of the columns in the input table are nullable.
  // If none of the input columns actually contain nulls, we can still use the
  // non-nullable version of the expression evaluation code path for
  // performance, so we capture that information as well.
  auto const has_nulls = binary_predicate.may_evaluate_null(left, right, stream);

  auto const parser =
    ast::detail::expression_parser{binary_predicate, left, right, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.");

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  // For inner joins we support optimizing the join by launching one thread for
  // whichever table is larger rather than always using the left table.
  auto swap_tables = (join_type == join_kind::INNER_JOIN) && (right_num_rows > left_num_rows);
  detail::grid_1d const config(swap_tables ? right_num_rows : left_num_rows,
                               DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  // Allocate storage for the counter used to get the size of the join output
  rmm::device_scalar<std::size_t> size(0, stream, mr);

  // Determine number of output rows without actually building the output to simply
  // find what the size of the output will be.
  if (has_nulls) {
    compute_conditional_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        join_type,
        parser.device_expression_data,
        swap_tables,
        size.data());
  } else {
    compute_conditional_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        join_type,
        parser.device_expression_data,
        swap_tables,
        size.data());
  }
  return size.value(stream);
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_inner_join(table_view const& left,
                       table_view const& right,
                       ast::expression const& binary_predicate,
                       std::optional<std::size_t> output_size,
                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::conditional_join(left,
                                  right,
                                  binary_predicate,
                                  detail::join_kind::INNER_JOIN,
                                  output_size,
                                  cudf::get_default_stream(),
                                  mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_left_join(table_view const& left,
                      table_view const& right,
                      ast::expression const& binary_predicate,
                      std::optional<std::size_t> output_size,
                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::conditional_join(left,
                                  right,
                                  binary_predicate,
                                  detail::join_kind::LEFT_JOIN,
                                  output_size,
                                  cudf::get_default_stream(),
                                  mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_full_join(table_view const& left,
                      table_view const& right,
                      ast::expression const& binary_predicate,
                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::conditional_join(left,
                                  right,
                                  binary_predicate,
                                  detail::join_kind::FULL_JOIN,
                                  {},
                                  cudf::get_default_stream(),
                                  mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> conditional_left_semi_join(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  std::optional<std::size_t> output_size,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return std::move(detail::conditional_join(left,
                                            right,
                                            binary_predicate,
                                            detail::join_kind::LEFT_SEMI_JOIN,
                                            output_size,
                                            cudf::get_default_stream(),
                                            mr)
                     .first);
}

std::unique_ptr<rmm::device_uvector<size_type>> conditional_left_anti_join(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  std::optional<std::size_t> output_size,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return std::move(detail::conditional_join(left,
                                            right,
                                            binary_predicate,
                                            detail::join_kind::LEFT_ANTI_JOIN,
                                            output_size,
                                            cudf::get_default_stream(),
                                            mr)
                     .first);
}

std::size_t conditional_inner_join_size(table_view const& left,
                                        table_view const& right,
                                        ast::expression const& binary_predicate,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_conditional_join_output_size(
    left, right, binary_predicate, detail::join_kind::INNER_JOIN, cudf::get_default_stream(), mr);
}

std::size_t conditional_left_join_size(table_view const& left,
                                       table_view const& right,
                                       ast::expression const& binary_predicate,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_conditional_join_output_size(
    left, right, binary_predicate, detail::join_kind::LEFT_JOIN, cudf::get_default_stream(), mr);
}

std::size_t conditional_left_semi_join_size(table_view const& left,
                                            table_view const& right,
                                            ast::expression const& binary_predicate,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return std::move(detail::compute_conditional_join_output_size(left,
                                                                right,
                                                                binary_predicate,
                                                                detail::join_kind::LEFT_SEMI_JOIN,
                                                                cudf::get_default_stream(),
                                                                mr));
}

std::size_t conditional_left_anti_join_size(table_view const& left,
                                            table_view const& right,
                                            ast::expression const& binary_predicate,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return std::move(detail::compute_conditional_join_output_size(left,
                                                                right,
                                                                binary_predicate,
                                                                detail::join_kind::LEFT_ANTI_JOIN,
                                                                cudf::get_default_stream(),
                                                                mr));
}

}  // namespace cudf
