/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include "mixed_join_kernels_semi.cuh"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <optional>
#include <utility>

namespace cudf {
namespace detail {

namespace {
/**
 * @brief Device functor to create a pair of hash value and index for a given row.
 */
struct make_pair_function_semi {
  __device__ __forceinline__ cudf::detail::pair_type operator()(size_type i) const noexcept
  {
    // The value is irrelevant since we only ever use the hash map to check for
    // membership of a particular row index.
    return cuco::make_pair(static_cast<hash_value_type>(i), 0);
  }
};

/**
 * @brief Equality comparator that composes two row_equality comparators.
 */
class double_row_equality {
 public:
  double_row_equality(row_equality equality_comparator, row_equality conditional_comparator)
    : _equality_comparator{equality_comparator}, _conditional_comparator{conditional_comparator}
  {
  }

  __device__ bool operator()(size_type lhs_row_index, size_type rhs_row_index) const noexcept
  {
    return _equality_comparator(lhs_row_index, rhs_row_index) &&
           _conditional_comparator(lhs_row_index, rhs_row_index);
  }

 private:
  row_equality _equality_comparator;
  row_equality _conditional_comparator;
};

}  // namespace

std::unique_ptr<rmm::device_uvector<size_type>> mixed_join_semi(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  join_kind join_type,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> output_size_data,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS((join_type != join_kind::INNER_JOIN) && (join_type != join_kind::LEFT_JOIN) &&
                 (join_type != join_kind::FULL_JOIN),
               "Inner, left, and full joins should use mixed_join.");

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

  // We can immediately filter out cases where the right table is empty. In
  // some cases, we return all the rows of the left table with a corresponding
  // null index for the right table; in others, we return an empty output.
  if (right_num_rows == 0) {
    switch (join_type) {
      // Anti and semi return all the row indices from left
      // with a corresponding NULL from the right.
      case join_kind::LEFT_ANTI_JOIN:
        return get_trivial_left_join_indices(
                 left_conditional, stream, rmm::mr::get_current_device_resource())
          .first;
      // Inner and left semi joins return empty output because no matches can exist.
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left_num_rows == 0) {
    switch (join_type) {
      // Anti and semi joins both return empty sets.
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
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
  auto& probe                  = swap_tables ? right_equality : left_equality;
  auto& build                  = swap_tables ? left_equality : right_equality;
  auto probe_view              = table_device_view::create(probe, stream);
  auto build_view              = table_device_view::create(build, stream);
  auto left_conditional_view   = table_device_view::create(left_conditional, stream);
  auto right_conditional_view  = table_device_view::create(right_conditional, stream);
  auto& build_conditional_view = swap_tables ? left_conditional_view : right_conditional_view;
  row_equality equality_probe{
    cudf::nullate::DYNAMIC{has_nulls}, *probe_view, *build_view, compare_nulls};

  semi_map_type hash_table{compute_hash_table_size(build.num_rows()),
                           cuco::empty_key{std::numeric_limits<hash_value_type>::max()},
                           cuco::empty_value{cudf::detail::JoinNoneValue},
                           detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                           stream.value()};

  // Create hash table containing all keys found in right table
  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  auto const build_nulls = cudf::nullate::DYNAMIC{cudf::has_nulls(build)};
  row_hash const hash_build{build_nulls, *build_view};
  // Since we may see multiple rows that are identical in the equality tables
  // but differ in the conditional tables, the equality comparator used for
  // insertion must account for both sets of tables. An alternative solution
  // would be to use a multimap, but that solution would store duplicates where
  // equality and conditional rows are equal, so this approach is preferable.
  // One way to make this solution even more efficient would be to only include
  // the columns of the conditional table that are used by the expression, but
  // that requires additional plumbing through the AST machinery and is out of
  // scope for now.
  row_equality equality_build_equality{build_nulls, *build_view, *build_view, compare_nulls};
  row_equality equality_build_conditional{
    build_nulls, *build_conditional_view, *build_conditional_view, compare_nulls};
  double_row_equality equality_build{equality_build_equality, equality_build_conditional};
  make_pair_function_semi pair_func_build{};

  auto iter = cudf::detail::make_counting_transform_iterator(0, pair_func_build);

  // skip rows that are null here.
  if ((compare_nulls == null_equality::EQUAL) or (not nullable(build))) {
    hash_table.insert(iter, iter + right_num_rows, hash_build, equality_build, stream.value());
  } else {
    thrust::counting_iterator<cudf::size_type> stencil(0);
    auto const [row_bitmask, _] =
      cudf::detail::bitmask_and(build, stream, rmm::mr::get_current_device_resource());
    row_is_valid pred{static_cast<bitmask_type const*>(row_bitmask.data())};

    // insert valid rows
    hash_table.insert_if(
      iter, iter + right_num_rows, stencil, pred, hash_build, equality_build, stream.value());
  }

  auto hash_table_view = hash_table.get_device_view();

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
      compute_mixed_join_output_size_semi<DEFAULT_JOIN_BLOCK_SIZE, true>
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
      compute_mixed_join_output_size_semi<DEFAULT_JOIN_BLOCK_SIZE, false>
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

  if (join_size == 0) { return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr); }

  // Given the number of matches per row, we need to compute the offsets for insertion.
  auto join_result_offsets =
    rmm::device_uvector<size_type>{static_cast<std::size_t>(outer_num_rows), stream, mr};
  thrust::exclusive_scan(rmm::exec_policy{stream},
                         matches_per_row_span.begin(),
                         matches_per_row_span.end(),
                         join_result_offsets.begin());

  auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto const& join_output_l = left_indices->data();

  if (has_nulls) {
    mixed_join_semi<DEFAULT_JOIN_BLOCK_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_conditional_view,
        *right_conditional_view,
        *probe_view,
        *build_view,
        equality_probe,
        kernel_join_type,
        hash_table_view,
        join_output_l,
        parser.device_expression_data,
        join_result_offsets.data(),
        swap_tables);
  } else {
    mixed_join_semi<DEFAULT_JOIN_BLOCK_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_conditional_view,
        *right_conditional_view,
        *probe_view,
        *build_view,
        equality_probe,
        kernel_join_type,
        hash_table_view,
        join_output_l,
        parser.device_expression_data,
        join_result_offsets.data(),
        swap_tables);
  }

  return left_indices;
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>>
compute_mixed_join_output_size_semi(table_view const& left_equality,
                                    table_view const& right_equality,
                                    table_view const& left_conditional,
                                    table_view const& right_conditional,
                                    ast::expression const& binary_predicate,
                                    null_equality compare_nulls,
                                    join_kind join_type,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(
    (join_type != join_kind::INNER_JOIN) && (join_type != join_kind::LEFT_JOIN) &&
      (join_type != join_kind::FULL_JOIN),
    "Inner, left, and full join size estimation should use compute_mixed_join_output_size.");

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
      case join_kind::LEFT_ANTI_JOIN: {
        thrust::fill(matches_per_row->begin(), matches_per_row->end(), 1);
        return {left_num_rows, std::move(matches_per_row)};
      }
      // Inner and left semi joins return empty output because no matches can exist.
      case join_kind::LEFT_SEMI_JOIN: return {0, std::move(matches_per_row)};
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left_num_rows == 0) {
    switch (join_type) {
      // Left, left anti, left semi, and inner joins all return empty sets.
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::LEFT_SEMI_JOIN: {
        thrust::fill(matches_per_row->begin(), matches_per_row->end(), 0);
        return {0, std::move(matches_per_row)};
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
  auto& probe                  = swap_tables ? right_equality : left_equality;
  auto& build                  = swap_tables ? left_equality : right_equality;
  auto probe_view              = table_device_view::create(probe, stream);
  auto build_view              = table_device_view::create(build, stream);
  auto left_conditional_view   = table_device_view::create(left_conditional, stream);
  auto right_conditional_view  = table_device_view::create(right_conditional, stream);
  auto& build_conditional_view = swap_tables ? left_conditional_view : right_conditional_view;
  row_equality equality_probe{
    cudf::nullate::DYNAMIC{has_nulls}, *probe_view, *build_view, compare_nulls};

  semi_map_type hash_table{compute_hash_table_size(build.num_rows()),
                           cuco::empty_key{std::numeric_limits<hash_value_type>::max()},
                           cuco::empty_value{cudf::detail::JoinNoneValue},
                           detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                           stream.value()};

  // Create hash table containing all keys found in right table
  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  auto const build_nulls = cudf::nullate::DYNAMIC{cudf::has_nulls(build)};
  row_hash const hash_build{build_nulls, *build_view};
  // Since we may see multiple rows that are identical in the equality tables
  // but differ in the conditional tables, the equality comparator used for
  // insertion must account for both sets of tables. An alternative solution
  // would be to use a multimap, but that solution would store duplicates where
  // equality and conditional rows are equal, so this approach is preferable.
  // One way to make this solution even more efficient would be to only include
  // the columns of the conditional table that are used by the expression, but
  // that requires additional plumbing through the AST machinery and is out of
  // scope for now.
  row_equality equality_build_equality{build_nulls, *build_view, *build_view, compare_nulls};
  row_equality equality_build_conditional{
    build_nulls, *build_conditional_view, *build_conditional_view, compare_nulls};
  double_row_equality equality_build{equality_build_equality, equality_build_conditional};
  make_pair_function_semi pair_func_build{};

  auto iter = cudf::detail::make_counting_transform_iterator(0, pair_func_build);

  // skip rows that are null here.
  if ((compare_nulls == null_equality::EQUAL) or (not nullable(build))) {
    hash_table.insert(iter, iter + right_num_rows, hash_build, equality_build, stream.value());
  } else {
    thrust::counting_iterator<cudf::size_type> stencil(0);
    auto const [row_bitmask, _] =
      cudf::detail::bitmask_and(build, stream, rmm::mr::get_current_device_resource());
    row_is_valid pred{static_cast<bitmask_type const*>(row_bitmask.data())};

    // insert valid rows
    hash_table.insert_if(
      iter, iter + right_num_rows, stencil, pred, hash_build, equality_build, stream.value());
  }

  auto hash_table_view = hash_table.get_device_view();

  // For inner joins we support optimizing the join by launching one thread for
  // whichever table is larger rather than always using the left table.
  detail::grid_1d const config(outer_num_rows, DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  // Allocate storage for the counter used to get the size of the join output
  rmm::device_scalar<std::size_t> size(0, stream, mr);

  // Determine number of output rows without actually building the output to simply
  // find what the size of the output will be.
  if (has_nulls) {
    compute_mixed_join_output_size_semi<DEFAULT_JOIN_BLOCK_SIZE, true>
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
    compute_mixed_join_output_size_semi<DEFAULT_JOIN_BLOCK_SIZE, false>
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

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_left_semi_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_mixed_join_output_size_semi(left_equality,
                                                     right_equality,
                                                     left_conditional,
                                                     right_conditional,
                                                     binary_predicate,
                                                     compare_nulls,
                                                     detail::join_kind::LEFT_SEMI_JOIN,
                                                     cudf::get_default_stream(),
                                                     mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_semi_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> output_size_data,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join_semi(left_equality,
                                 right_equality,
                                 left_conditional,
                                 right_conditional,
                                 binary_predicate,
                                 compare_nulls,
                                 detail::join_kind::LEFT_SEMI_JOIN,
                                 output_size_data,
                                 cudf::get_default_stream(),
                                 mr);
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_left_anti_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_mixed_join_output_size_semi(left_equality,
                                                     right_equality,
                                                     left_conditional,
                                                     right_conditional,
                                                     binary_predicate,
                                                     compare_nulls,
                                                     detail::join_kind::LEFT_ANTI_JOIN,
                                                     cudf::get_default_stream(),
                                                     mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_anti_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> output_size_data,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join_semi(left_equality,
                                 right_equality,
                                 left_conditional,
                                 right_conditional,
                                 binary_predicate,
                                 compare_nulls,
                                 detail::join_kind::LEFT_ANTI_JOIN,
                                 output_size_data,
                                 cudf::get_default_stream(),
                                 mr);
}

}  // namespace cudf
