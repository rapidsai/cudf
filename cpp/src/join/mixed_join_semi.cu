/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include "mixed_join_common_utils.cuh"
#include "mixed_join_kernels_semi.cuh"
#include "mixed_join_primitive_utils.cuh"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/join/mixed_join.hpp>
#include <cudf/table/primitive_row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>

#include <memory>

namespace cudf {
namespace detail {

std::unique_ptr<rmm::device_uvector<size_type>> mixed_join_semi(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  join_kind join_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS((join_type != join_kind::INNER_JOIN) and (join_type != join_kind::LEFT_JOIN) and
                 (join_type != join_kind::FULL_JOIN),
               "Inner, left, and full joins should use mixed_join.");

  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  auto const left_num_rows{left_conditional.num_rows()};
  auto const right_num_rows{right_conditional.num_rows()};

  // We can immediately filter out cases where the right table is empty. In
  // some cases, we return all the rows of the left table with a corresponding
  // null index for the right table; in others, we return an empty output.
  if (right_num_rows == 0) {
    switch (join_type) {
      // Anti and semi return all the row indices from left
      // with a corresponding NULL from the right.
      case join_kind::LEFT_ANTI_JOIN:
        return get_trivial_left_join_indices(left_conditional, stream, mr).first;
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
  auto const has_nulls = cudf::nullate::DYNAMIC{
    cudf::has_nulls(left_equality) or cudf::has_nulls(right_equality) or
    binary_predicate.may_evaluate_null(left_conditional, right_conditional, stream)};

  auto const parser = ast::detail::expression_parser{
    binary_predicate, left_conditional, right_conditional, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.");

  // TODO: The non-conditional join impls start with a dictionary matching,
  // figure out what that is and what it's needed for (and if conditional joins
  // need to do the same).
  auto const& build_table = right_equality;
  auto const& probe_table = left_equality;

  // Create hash table containing all keys found in right table
  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  auto const build_nulls = cudf::nullate::DYNAMIC{cudf::has_nulls(build_table)};
  auto const preprocessed_build =
    cudf::experimental::row::equality::preprocessed_table::create(build_table, stream);

  // Check if we can use primitive row operators for better performance
  bool const use_primitive_operators = false;  // cudf::is_primitive_row_op_compatible(build_table);

  // Common setup for both primitive and non-primitive paths
  auto const preprocessed_build_conditional =
    cudf::experimental::row::equality::preprocessed_table::create(right_conditional, stream);
  auto const preprocessed_probe =
    cudf::experimental::row::equality::preprocessed_table::create(probe_table, stream);

  // Create device views
  auto const probe_view             = table_device_view::create(probe_table, stream);
  auto const build_view             = table_device_view::create(build_table, stream);
  auto const left_conditional_view  = table_device_view::create(left_conditional, stream);
  auto const right_conditional_view = table_device_view::create(right_conditional, stream);

  // Common kernel launch configuration
  auto const probe_num_rows = probe_table.num_rows();
  detail::grid_1d const config(probe_num_rows * DEFAULT_MIXED_SEMI_JOIN_CG_SIZE,
                               DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block =
    parser.shmem_per_thread *
    cuco::detail::int_div_ceil(config.num_threads_per_block, DEFAULT_MIXED_SEMI_JOIN_CG_SIZE);

  // Vector used to indicate indices from left/probe table which are present in output
  auto left_table_keep_mask = rmm::device_uvector<bool>(probe_table.num_rows(), stream);

  // Create row comparators and hash table based on operator compatibility

  // Lambda to insert rows into hash set - common for both paths
  auto insert_rows = [&](auto& row_set) {
    auto iter = thrust::make_counting_iterator(0);
    if ((compare_nulls == null_equality::EQUAL) or (not nullable(build_table))) {
      row_set.insert_async(iter, iter + right_num_rows, stream.value());
    } else {
      thrust::counting_iterator<cudf::size_type> stencil(0);
      auto const [row_bitmask, _] =
        cudf::detail::bitmask_and(build_table, stream, cudf::get_current_device_resource_ref());
      row_is_valid pred{static_cast<bitmask_type const*>(row_bitmask.data())};
      row_set.insert_if_async(iter, iter + right_num_rows, stencil, pred, stream.value());
    }
  };

  // Lambda to launch kernel - common for both paths
  auto launch_kernel = [&](auto const& equality_probe, auto const& row_set_ref) {
    launch_mixed_join_semi(has_nulls,
                           *left_conditional_view,
                           *right_conditional_view,
                           *probe_view,
                           *build_view,
                           equality_probe,
                           row_set_ref,
                           cudf::device_span<bool>(left_table_keep_mask),
                           parser.device_expression_data,
                           config,
                           shmem_size_per_block,
                           stream);
  };

  if (use_primitive_operators) {
    // Use primitive row operators for equality comparison (better performance)
    auto const equality_build_equality = cudf::row::primitive::row_equality_comparator{
      build_nulls, preprocessed_build, preprocessed_build, compare_nulls};
    auto const equality_build_conditional = cudf::row::primitive::row_equality_comparator{
      build_nulls, preprocessed_build_conditional, preprocessed_build_conditional, compare_nulls};

    // Use the primitive double row equality comparator with regular hash infrastructure
    hash_set_type<primitive_double_row_equality_comparator, primitive_row_hash> row_set{
      {compute_hash_table_size(build_table.num_rows())},
      cuco::empty_key{JoinNoneValue},
      {equality_build_equality, equality_build_conditional},
      {cudf::row::primitive::row_hasher{build_nulls, preprocessed_build}},
      {},
      {},
      cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
      {stream.value()}};

    insert_rows(row_set);

    auto const equality_probe = cudf::row::primitive::row_equality_comparator{
      has_nulls, preprocessed_probe, preprocessed_build, compare_nulls};
    auto const row_hash_probe = cudf::row::primitive::row_hasher{has_nulls, preprocessed_probe};

    hash_set_ref_type<primitive_double_row_equality_comparator, primitive_row_hash> const
      row_set_ref = row_set.ref(cuco::contains).rebind_hash_function(row_hash_probe);

    launch_kernel(equality_probe, row_set_ref);
  } else {
    // Use non-primitive row operators (original implementation)
    auto const row_hash_build = cudf::experimental::row::hash::row_hasher{preprocessed_build};

    // Since we may see multiple rows that are identical in the equality tables
    // but differ in the conditional tables, the equality comparator used for
    // insertion must account for both sets of tables. An alternative solution
    // would be to use a multimap, but that solution would store duplicates where
    // equality and conditional rows are equal, so this approach is preferable.
    // One way to make this solution even more efficient would be to only include
    // the columns of the conditional table that are used by the expression, but
    // that requires additional plumbing through the AST machinery and is out of
    // scope for now.
    auto const row_comparator_build = cudf::experimental::row::equality::two_table_comparator{
      preprocessed_build, preprocessed_build};
    auto const equality_build_equality =
      row_comparator_build.equal_to<false>(build_nulls, compare_nulls);
    auto const row_comparator_conditional_build =
      cudf::experimental::row::equality::two_table_comparator{preprocessed_build_conditional,
                                                              preprocessed_build_conditional};
    auto const equality_build_conditional =
      row_comparator_conditional_build.equal_to<false>(build_nulls, compare_nulls);

    hash_set_type row_set{
      {compute_hash_table_size(build_table.num_rows())},
      cuco::empty_key{JoinNoneValue},
      {equality_build_equality, equality_build_conditional},
      {row_hash_build.device_hasher(build_nulls)},
      {},
      {},
      cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
      {stream.value()}};

    insert_rows(row_set);

    auto const row_comparator = cudf::experimental::row::equality::two_table_comparator{
      preprocessed_build, preprocessed_probe};
    auto const equality_probe = row_comparator.equal_to<false>(has_nulls, compare_nulls);
    auto const row_hash_probe = cudf::experimental::row::hash::row_hasher{preprocessed_probe};
    auto const hash_probe     = row_hash_probe.device_hasher(has_nulls);

    hash_set_ref_type<double_row_equality_comparator, row_hash> const row_set_ref =
      row_set.ref(cuco::contains).rebind_hash_function(hash_probe);

    launch_kernel(equality_probe, row_set_ref);
  }

  // Common post-processing for both paths
  auto gather_map =
    std::make_unique<rmm::device_uvector<size_type>>(probe_table.num_rows(), stream, mr);

  // gather_map_end will be the end of valid data in gather_map
  auto gather_map_end =
    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::counting_iterator<size_type>(0),
                    thrust::counting_iterator<size_type>(probe_table.num_rows()),
                    left_table_keep_mask.begin(),
                    gather_map->begin(),
                    [join_type] __device__(bool keep_row) {
                      return keep_row == (join_type == detail::join_kind::LEFT_SEMI_JOIN);
                    });

  gather_map->resize(cuda::std::distance(gather_map->begin(), gather_map_end), stream);
  return gather_map;
}

}  // namespace detail

std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_semi_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join_semi(left_equality,
                                 right_equality,
                                 left_conditional,
                                 right_conditional,
                                 binary_predicate,
                                 compare_nulls,
                                 detail::join_kind::LEFT_SEMI_JOIN,
                                 stream,
                                 mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_anti_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join_semi(left_equality,
                                 right_equality,
                                 left_conditional,
                                 right_conditional,
                                 binary_predicate,
                                 compare_nulls,
                                 detail::join_kind::LEFT_ANTI_JOIN,
                                 stream,
                                 mr);
}

}  // namespace cudf
