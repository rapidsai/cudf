/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <join/hash_join.cuh>
#include <join/join_common_utils.cuh>
#include <join/join_common_utils.hpp>
#include <join/mixed_join_kernels.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <optional>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_join(table_view const& left,
           table_view const& right,
           std::vector<cudf::size_type> const& left_on,
           std::vector<cudf::size_type> const& right_on,
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
        return std::make_pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
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
        return std::make_pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                              std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
      // Full joins need to return the trivial complement.
      case join_kind::FULL_JOIN: {
        auto ret_flipped = get_trivial_left_join_indices(right, stream);
        return std::make_pair(std::move(ret_flipped.second), std::move(ret_flipped.first));
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

  // TODO: The non-conditional join impls start with a dictionary matching,
  // figure out what that is and what it's needed for (and if conditional joins
  // need to do the same).
  auto swap_tables = (join_type == join_kind::INNER_JOIN) && (right_num_rows > left_num_rows);
  CUDF_EXPECTS(left_on.size() > 0 && right_on.size() > 0,
               "Need to equality join on at least one column.");
  auto probe      = swap_tables ? right.select(right_on) : left.select(left_on);
  auto build      = swap_tables ? left.select(right_on) : right.select(left_on);
  auto build_view = table_device_view::create(build, stream);
  auto probe_view = table_device_view::create(probe, stream);
  // TODO: Introducing this as an output is a fundamental change because
  // calling code that wants to pass the size calculation to multiple join
  // calculations as an optimization must now also pass this vector. Since this
  // vector's nature changes depending on whether or not swapping occurs, we're
  // effectively exposing an implementation detail. I don't know if there's any
  // way to avoid this.
  auto matches_per_row = std::make_unique<rmm::device_uvector<size_type>>(
    swap_tables ? right.num_rows() : left.num_rows(), stream, mr);

  // Don't use multimap_type because we want a CG size of 1.
  mixed_multimap_type hash_table{compute_hash_table_size(build.num_rows()),
                                 std::numeric_limits<hash_value_type>::max(),
                                 cudf::detail::JoinNoneValue,
                                 stream.value()};

  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  // TODO: Decide how to handle null equality when mixing AST operators with hash joins.
  if ((build.num_columns() == 0) && (build.num_rows() != 0)) {
    build_join_hash_table(build, hash_table, null_equality::EQUAL, stream);
  }
  // TODO: Should this be a pair_equality?
  // row_equality equality{cudf::nullate::YES{}, *probe_view, *build_view, null_equality::EQUAL};

  // row_equality equality{*probe_view, *build_view, compare_nulls == null_equality::EQUAL};
  auto hash_table_view = hash_table.get_device_view();

  /*
     I need to compute the output size. Unlike with hash joins, I think I'll have to use a
     custom kernel (like in conditional joins) because the expression_evaluator uses shared
     memory. There's no way to achieve that using thrust. Therefore, rather than writing a
     probe function like we do in hash joins, I think we need to pass a device view of the
     container to kernels for both size calculation and the actual join. So ultimately this
     code is going to end up looking a lot closer to the conditional joins than the hash joins,
     it's just that we're going to build the hash table on the host side and add a probe of
     the device view and iterate over the results to determine what indices to test against.

     The multimap does not yet support a device function iterator over the results, so we'll
     need to dynamically allocate a device-side array for the output indices and insert into
     that. Eventually we should add an iterator equivalent of static_multimap::retrieve in cuco.
  */

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  // For inner joins we support optimizing the join by launching one thread for
  // whichever table is larger rather than always using the left table.
  detail::grid_1d config(swap_tables ? right_num_rows : left_num_rows, DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;
  join_kind kernel_join_type = join_type == join_kind::FULL_JOIN ? join_kind::LEFT_JOIN : join_type;

  // If the join size was not provided as an input, compute it here.
  std::size_t join_size;
  if (output_size.has_value()) {
    join_size = *output_size;
  } else {
    // Allocate storage for the counter used to get the size of the join output
    rmm::device_scalar<std::size_t> size(0, stream, mr);
    CHECK_CUDA(stream.value());
    if (has_nulls) {
      compute_mixed_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, true>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_table,
          *right_table,
          *probe_view,
          *build_view,
          kernel_join_type,
          hash_table_view,
          parser.device_expression_data,
          swap_tables,
          size.data(),
          matches_per_row->data());
    } else {
      compute_mixed_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, false>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_table,
          *right_table,
          *probe_view,
          *build_view,
          kernel_join_type,
          hash_table_view,
          parser.device_expression_data,
          swap_tables,
          size.data(),
          matches_per_row->data());
    }
    CHECK_CUDA(stream.value());
    join_size = size.value(stream);
  }

  // The initial early exit clauses guarantee that we will not reach this point
  // unless both the left and right tables are non-empty. Under that
  // constraint, neither left nor full joins can return an empty result since
  // at minimum we are guaranteed null matches for all non-matching rows. In
  // all other cases (inner, left semi, and left anti joins) if we reach this
  // point we can safely return an empty result.
  if (join_size == 0) {
    return std::make_pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                          std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  rmm::device_scalar<size_type> write_index(0, stream);

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const& join_output_l = left_indices->data();
  auto const& join_output_r = right_indices->data();
  if (has_nulls) {
    mixed_join<DEFAULT_JOIN_BLOCK_SIZE, DEFAULT_JOIN_CACHE_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        *probe_view,
        *build_view,
        kernel_join_type,
        hash_table_view,
        join_output_l,
        join_output_r,
        write_index.data(),
        parser.device_expression_data,
        join_size,
        swap_tables);
  } else {
    mixed_join<DEFAULT_JOIN_BLOCK_SIZE, DEFAULT_JOIN_CACHE_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        *probe_view,
        *build_view,
        kernel_join_type,
        hash_table_view,
        join_output_l,
        join_output_r,
        write_index.data(),
        parser.device_expression_data,
        join_size,
        swap_tables);
  }

  CHECK_CUDA(stream.value());

  auto join_indices = std::make_pair(std::move(left_indices), std::move(right_indices));

  // For full joins, get the indices in the right table that were not joined to
  // by any row in the left table.
  if (join_type == join_kind::FULL_JOIN) {
    auto complement_indices = detail::get_left_join_indices_complement(
      join_indices.second, left_num_rows, right_num_rows, stream, mr);
    join_indices = detail::concatenate_vector_pairs(join_indices, complement_indices, stream);
  }
  return join_indices;
}

std::size_t compute_mixed_join_output_size(table_view const& left,
                                           table_view const& right,
                                           std::vector<cudf::size_type> const& left_on,
                                           std::vector<cudf::size_type> const& right_on,
                                           ast::expression const& binary_predicate,
                                           join_kind join_type,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
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
  auto const nullable  = cudf::nullable(left) || cudf::nullable(right);
  auto const has_nulls = nullable && (cudf::has_nulls(left) || cudf::has_nulls(right));

  auto const parser =
    ast::detail::expression_parser{binary_predicate, left, right, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.");

  // TODO: The non-conditional join impls start with a dictionary matching,
  // figure out what that is and what it's needed for (and if conditional joins
  // need to do the same).
  auto swap_tables = (join_type == join_kind::INNER_JOIN) && (right_num_rows > left_num_rows);
  auto probe       = swap_tables ? right.select(right_on) : left.select(left_on);
  auto build       = swap_tables ? left.select(left_on) : right.select(right_on);
  auto probe_view  = table_device_view::create(probe, stream);
  auto build_view  = table_device_view::create(build, stream);
  // TODO: Introducing this as an output is a fundamental change because
  // calling code that wants to pass the size calculation to multiple join
  // calculations as an optimization must now also pass this vector. Since this
  // vector's nature changes depending on whether or not swapping occurs, we're
  // effectively exposing an implementation detail. I don't know if there's any
  // way to avoid this.
  auto matches_per_row = std::make_unique<rmm::device_uvector<size_type>>(
    swap_tables ? right.num_rows() : left.num_rows(), stream, mr);

  // Don't use multimap_type because we want a CG size of 1.
  mixed_multimap_type hash_table{compute_hash_table_size(build.num_rows()),
                                 std::numeric_limits<hash_value_type>::max(),
                                 cudf::detail::JoinNoneValue,
                                 stream.value()};

  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  // TODO: Decide how to handle null equality when mixing AST operators with hash joins.
  if ((build.num_columns() != 0) && (build.num_rows() != 0)) {
    build_join_hash_table(build, hash_table, null_equality::EQUAL, stream);
  }
  // row_equality equality{cudf::nullate::YES{}, *probe_view, *build_view, true};
  // row_equality equality{*probe_view, *build_view, compare_nulls == null_equality::EQUAL};
  auto hash_table_view = hash_table.get_device_view();

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  // For inner joins we support optimizing the join by launching one thread for
  // whichever table is larger rather than always using the left table.
  detail::grid_1d config(swap_tables ? right_num_rows : left_num_rows, DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  assert(join_type != join_kind::FULL_JOIN);

  // Allocate storage for the counter used to get the size of the join output
  rmm::device_scalar<std::size_t> size(0, stream, mr);
  CHECK_CUDA(stream.value());

  // Determine number of output rows without actually building the output to simply
  // find what the size of the output will be.
  if (has_nulls) {
    compute_mixed_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        *probe_view,
        *build_view,
        join_type,
        hash_table_view,
        parser.device_expression_data,
        swap_tables,
        size.data(),
        matches_per_row->data());
  } else {
    compute_mixed_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        *probe_view,
        *build_view,
        join_type,
        hash_table_view,
        parser.device_expression_data,
        swap_tables,
        size.data(),
        matches_per_row->data());
  }
  CHECK_CUDA(stream.value());

  return size.value(stream);
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_inner_join(table_view const& left,
                 table_view const& right,
                 std::vector<cudf::size_type> const& left_on,
                 std::vector<cudf::size_type> const& right_on,
                 ast::expression const& binary_predicate,
                 std::optional<std::size_t> output_size,
                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join(left,
                            right,
                            left_on,
                            right_on,
                            binary_predicate,
                            detail::join_kind::INNER_JOIN,
                            output_size,
                            rmm::cuda_stream_default,
                            mr);
}

std::size_t mixed_inner_join_size(table_view const& left,
                                  table_view const& right,
                                  std::vector<cudf::size_type> const& left_on,
                                  std::vector<cudf::size_type> const& right_on,
                                  ast::expression const& binary_predicate,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_mixed_join_output_size(left,
                                                right,
                                                left_on,
                                                right_on,
                                                binary_predicate,
                                                detail::join_kind::INNER_JOIN,
                                                rmm::cuda_stream_default,
                                                mr);
}

}  // namespace cudf
