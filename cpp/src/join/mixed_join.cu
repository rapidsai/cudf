/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "join_common_utils.cuh"
#include "join_common_utils.hpp"
#include "mixed_join_kernels.cuh"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/scan.h>

#include <optional>
#include <utility>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  join_kind join_type,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> const& output_size_data,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  CUDF_EXPECTS((join_type != join_kind::LEFT_SEMI_JOIN) && (join_type != join_kind::LEFT_ANTI_JOIN),
               "Left semi and anti joins should use mixed_join_semi.");

  auto const right_num_rows{right_conditional.num_rows()};
  auto const left_num_rows{left_conditional.num_rows()};
  auto const swap_tables = (join_type == join_kind::INNER_JOIN) && (right_num_rows > left_num_rows);

  // The "outer" table is the larger of the two tables. The kernels are
  // launched with one thread per row of the outer table, which also means that
  // it is the probe table for the hash
  auto const outer_num_rows{swap_tables ? right_num_rows : left_num_rows};

  // We can immediately filter out cases where the right table is empty. In
  // some cases, we return all the rows of the left table with a corresponding
  // null index for the right table; in others, we return an empty output.
  if (right_num_rows == 0) {
    switch (join_type) {
      // Left and full joins all return all the row indices from
      // left with a corresponding NULL from the right.
      case join_kind::LEFT_JOIN:
      case join_kind::FULL_JOIN: return get_trivial_left_join_indices(left_conditional, stream);
      // Inner joins return empty output because no matches can exist.
      case join_kind::INNER_JOIN:
        return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                         std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left_num_rows == 0) {
    switch (join_type) {
      // Left and inner joins all return empty sets.
      case join_kind::LEFT_JOIN:
      case join_kind::INNER_JOIN:
        return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                         std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
      // Full joins need to return the trivial complement.
      case join_kind::FULL_JOIN: {
        auto ret_flipped = get_trivial_left_join_indices(right_conditional, stream);
        return std::pair(std::move(ret_flipped.second), std::move(ret_flipped.first));
      }
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  }

  // If evaluating the expression may produce null outputs we create a nullable
  // output column and follow the null-supporting expression evaluation code
  // path.
  auto const has_nulls =
    cudf::has_nulls(left_equality) || cudf::has_nulls(right_equality) ||
    binary_predicate.may_evaluate_null(left_conditional, right_conditional, stream);

  auto const parser = ast::detail::expression_parser{
    binary_predicate, left_conditional, right_conditional, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.");

  // TODO: The non-conditional join impls start with a dictionary matching,
  // figure out what that is and what it's needed for (and if conditional joins
  // need to do the same).
  auto& probe     = swap_tables ? right_equality : left_equality;
  auto& build     = swap_tables ? left_equality : right_equality;
  auto probe_view = table_device_view::create(probe, stream);
  auto build_view = table_device_view::create(build, stream);
  row_equality equality_probe{
    cudf::nullate::DYNAMIC{has_nulls}, *probe_view, *build_view, compare_nulls};

  // Don't use multimap_type because we want a CG size of 1.
  mixed_multimap_type hash_table{
    compute_hash_table_size(build.num_rows()),
    cuco::sentinel::empty_key{std::numeric_limits<hash_value_type>::max()},
    cuco::sentinel::empty_value{cudf::detail::JoinNoneValue},
    stream.value(),
    detail::hash_table_allocator_type{default_allocator<char>{}, stream}};

  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  auto const row_bitmask = cudf::detail::bitmask_and(build, stream).first;
  build_join_hash_table(
    build, hash_table, compare_nulls, static_cast<bitmask_type const*>(row_bitmask.data()), stream);
  auto hash_table_view = hash_table.get_device_view();

  auto left_conditional_view  = table_device_view::create(left_conditional, stream);
  auto right_conditional_view = table_device_view::create(right_conditional, stream);

  // For inner joins we support optimizing the join by launching one thread for
  // whichever table is larger rather than always using the left table.
  detail::grid_1d const config(outer_num_rows, DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;
  join_kind const kernel_join_type =
    join_type == join_kind::FULL_JOIN ? join_kind::LEFT_JOIN : join_type;

  // If the join size data was not provided as an input, compute it here.
  std::size_t join_size;
  // Using an optional because we only need to allocate a new vector if one was
  // not passed as input, and rmm::device_uvector is not default constructible
  std::optional<rmm::device_uvector<size_type>> matches_per_row{};
  device_span<size_type const> matches_per_row_span{};

  if (output_size_data.has_value()) {
    join_size            = output_size_data->first;
    matches_per_row_span = output_size_data->second;
  } else {
    // Allocate storage for the counter used to get the size of the join output
    rmm::device_scalar<std::size_t> size(0, stream, mr);

    matches_per_row =
      rmm::device_uvector<size_type>{static_cast<std::size_t>(outer_num_rows), stream, mr};
    // Note that the view goes out of scope after this else statement, but the
    // data owned by matches_per_row stays alive so the data pointer is valid.
    auto mutable_matches_per_row_span = cudf::device_span<size_type>{
      matches_per_row->begin(), static_cast<std::size_t>(outer_num_rows)};
    matches_per_row_span = cudf::device_span<size_type const>{
      matches_per_row->begin(), static_cast<std::size_t>(outer_num_rows)};
    if (has_nulls) {
      compute_mixed_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, true>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_conditional_view,
          *right_conditional_view,
          *probe_view,
          *build_view,
          equality_probe,
          kernel_join_type,
          hash_table_view,
          parser.device_expression_data,
          swap_tables,
          size.data(),
          mutable_matches_per_row_span);
    } else {
      compute_mixed_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, false>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_conditional_view,
          *right_conditional_view,
          *probe_view,
          *build_view,
          equality_probe,
          kernel_join_type,
          hash_table_view,
          parser.device_expression_data,
          swap_tables,
          size.data(),
          mutable_matches_per_row_span);
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

  // Given the number of matches per row, we need to compute the offsets for insertion.
  auto join_result_offsets =
    rmm::device_uvector<size_type>{static_cast<std::size_t>(outer_num_rows), stream, mr};
  thrust::exclusive_scan(rmm::exec_policy{stream},
                         matches_per_row_span.begin(),
                         matches_per_row_span.end(),
                         join_result_offsets.begin());

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const& join_output_l = left_indices->data();
  auto const& join_output_r = right_indices->data();

  if (has_nulls) {
    mixed_join<DEFAULT_JOIN_BLOCK_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_conditional_view,
        *right_conditional_view,
        *probe_view,
        *build_view,
        equality_probe,
        kernel_join_type,
        hash_table_view,
        join_output_l,
        join_output_r,
        parser.device_expression_data,
        join_result_offsets.data(),
        swap_tables);
  } else {
    mixed_join<DEFAULT_JOIN_BLOCK_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_conditional_view,
        *right_conditional_view,
        *probe_view,
        *build_view,
        equality_probe,
        kernel_join_type,
        hash_table_view,
        join_output_l,
        join_output_r,
        parser.device_expression_data,
        join_result_offsets.data(),
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

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>>
compute_mixed_join_output_size(table_view const& left_equality,
                               table_view const& right_equality,
                               table_view const& left_conditional,
                               table_view const& right_conditional,
                               ast::expression const& binary_predicate,
                               null_equality compare_nulls,
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

  CUDF_EXPECTS(
    (join_type != join_kind::LEFT_SEMI_JOIN) && (join_type != join_kind::LEFT_ANTI_JOIN),
    "Left semi and anti join size estimation should use compute_mixed_join_output_size_semi.");

  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  auto const right_num_rows{right_conditional.num_rows()};
  auto const left_num_rows{left_conditional.num_rows()};
  auto const swap_tables = (join_type == join_kind::INNER_JOIN) && (right_num_rows > left_num_rows);

  // The "outer" table is the larger of the two tables. The kernels are
  // launched with one thread per row of the outer table, which also means that
  // it is the probe table for the hash
  auto const outer_num_rows{swap_tables ? right_num_rows : left_num_rows};

  auto matches_per_row = std::make_unique<rmm::device_uvector<size_type>>(
    static_cast<std::size_t>(outer_num_rows), stream, mr);
  auto matches_per_row_span = cudf::device_span<size_type>{
    matches_per_row->begin(), static_cast<std::size_t>(outer_num_rows)};

  // We can immediately filter out cases where one table is empty. In
  // some cases, we return all the rows of the other table with a corresponding
  // null index for the empty table; in others, we return an empty output.
  if (right_num_rows == 0) {
    switch (join_type) {
      // Left, left anti, and full all return all the row indices from left
      // with a corresponding NULL from the right.
      case join_kind::LEFT_JOIN:
      case join_kind::FULL_JOIN: {
        thrust::fill(matches_per_row->begin(), matches_per_row->end(), 1);
        return {left_num_rows, std::move(matches_per_row)};
      }
      // Inner and left semi joins return empty output because no matches can exist.
      case join_kind::INNER_JOIN: {
        thrust::fill(matches_per_row->begin(), matches_per_row->end(), 0);
        return {0, std::move(matches_per_row)};
      }
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left_num_rows == 0) {
    switch (join_type) {
      // Left, left anti, left semi, and inner joins all return empty sets.
      case join_kind::LEFT_JOIN:
      case join_kind::INNER_JOIN: {
        thrust::fill(matches_per_row->begin(), matches_per_row->end(), 0);
        return {0, std::move(matches_per_row)};
      }
      // Full joins need to return the trivial complement.
      case join_kind::FULL_JOIN: {
        thrust::fill(matches_per_row->begin(), matches_per_row->end(), 1);
        return {right_num_rows, std::move(matches_per_row)};
      }
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  }

  // If evaluating the expression may produce null outputs we create a nullable
  // output column and follow the null-supporting expression evaluation code
  // path.
  auto const has_nulls =
    cudf::has_nulls(left_equality) || cudf::has_nulls(right_equality) ||
    binary_predicate.may_evaluate_null(left_conditional, right_conditional, stream);

  auto const parser = ast::detail::expression_parser{
    binary_predicate, left_conditional, right_conditional, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.");

  // TODO: The non-conditional join impls start with a dictionary matching,
  // figure out what that is and what it's needed for (and if conditional joins
  // need to do the same).
  auto& probe     = swap_tables ? right_equality : left_equality;
  auto& build     = swap_tables ? left_equality : right_equality;
  auto probe_view = table_device_view::create(probe, stream);
  auto build_view = table_device_view::create(build, stream);
  row_equality equality_probe{
    cudf::nullate::DYNAMIC{has_nulls}, *probe_view, *build_view, compare_nulls};

  // Don't use multimap_type because we want a CG size of 1.
  mixed_multimap_type hash_table{
    compute_hash_table_size(build.num_rows()),
    cuco::sentinel::empty_key{std::numeric_limits<hash_value_type>::max()},
    cuco::sentinel::empty_value{cudf::detail::JoinNoneValue},
    stream.value(),
    detail::hash_table_allocator_type{default_allocator<char>{}, stream}};

  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  auto const row_bitmask = cudf::detail::bitmask_and(build, stream).first;
  build_join_hash_table(
    build, hash_table, compare_nulls, static_cast<bitmask_type const*>(row_bitmask.data()), stream);
  auto hash_table_view = hash_table.get_device_view();

  auto left_conditional_view  = table_device_view::create(left_conditional, stream);
  auto right_conditional_view = table_device_view::create(right_conditional, stream);

  // For inner joins we support optimizing the join by launching one thread for
  // whichever table is larger rather than always using the left table.
  detail::grid_1d const config(outer_num_rows, DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  // Allocate storage for the counter used to get the size of the join output
  rmm::device_scalar<std::size_t> size(0, stream, mr);

  // Determine number of output rows without actually building the output to simply
  // find what the size of the output will be.
  if (has_nulls) {
    compute_mixed_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_conditional_view,
        *right_conditional_view,
        *probe_view,
        *build_view,
        equality_probe,
        join_type,
        hash_table_view,
        parser.device_expression_data,
        swap_tables,
        size.data(),
        matches_per_row_span);
  } else {
    compute_mixed_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_conditional_view,
        *right_conditional_view,
        *probe_view,
        *build_view,
        equality_probe,
        join_type,
        hash_table_view,
        parser.device_expression_data,
        swap_tables,
        size.data(),
        matches_per_row_span);
  }

  return {size.value(stream), std::move(matches_per_row)};
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_inner_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> const output_size_data,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join(left_equality,
                            right_equality,
                            left_conditional,
                            right_conditional,
                            binary_predicate,
                            compare_nulls,
                            detail::join_kind::INNER_JOIN,
                            output_size_data,
                            cudf::get_default_stream(),
                            mr);
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_inner_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_mixed_join_output_size(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                detail::join_kind::INNER_JOIN,
                                                cudf::get_default_stream(),
                                                mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_left_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> const output_size_data,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join(left_equality,
                            right_equality,
                            left_conditional,
                            right_conditional,
                            binary_predicate,
                            compare_nulls,
                            detail::join_kind::LEFT_JOIN,
                            output_size_data,
                            cudf::get_default_stream(),
                            mr);
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_left_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_mixed_join_output_size(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                detail::join_kind::LEFT_JOIN,
                                                cudf::get_default_stream(),
                                                mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_full_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> const output_size_data,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join(left_equality,
                            right_equality,
                            left_conditional,
                            right_conditional,
                            binary_predicate,
                            compare_nulls,
                            detail::join_kind::FULL_JOIN,
                            output_size_data,
                            cudf::get_default_stream(),
                            mr);
}

}  // namespace cudf
