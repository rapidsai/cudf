/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/join.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <optional>
#include <set>
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
    using experimental::row::lhs_index_type;
    using experimental::row::rhs_index_type;

    return _equality_comparator(lhs_index_type{lhs_row_index}, rhs_index_type{rhs_row_index}) &&
           _conditional_comparator(lhs_index_type{lhs_row_index}, rhs_index_type{rhs_row_index});
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
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
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
  auto const outer_num_rows{left_num_rows};

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
  auto const has_nulls = cudf::nullate::DYNAMIC{
    cudf::has_nulls(left_equality) || cudf::has_nulls(right_equality) ||
    binary_predicate.may_evaluate_null(left_conditional, right_conditional, stream)};

  auto const parser = ast::detail::expression_parser{
    binary_predicate, left_conditional, right_conditional, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.");

  // TODO: The non-conditional join impls start with a dictionary matching,
  // figure out what that is and what it's needed for (and if conditional joins
  // need to do the same).
  auto& probe                 = left_equality;
  auto& build                 = right_equality;
  auto probe_view             = table_device_view::create(probe, stream);
  auto build_view             = table_device_view::create(build, stream);
  auto left_conditional_view  = table_device_view::create(left_conditional, stream);
  auto right_conditional_view = table_device_view::create(right_conditional, stream);

  auto const preprocessed_build =
    experimental::row::equality::preprocessed_table::create(build, stream);
  auto const preprocessed_probe =
    experimental::row::equality::preprocessed_table::create(probe, stream);
  auto const row_comparator =
    cudf::experimental::row::equality::two_table_comparator{preprocessed_probe, preprocessed_build};
  auto const equality_probe = row_comparator.equal_to<false>(has_nulls, compare_nulls);

  semi_map_type hash_table{compute_hash_table_size(build.num_rows()),
                           cuco::empty_key{std::numeric_limits<hash_value_type>::max()},
                           cuco::empty_value{cudf::detail::JoinNoneValue},
                           cudf::detail::cuco_allocator{stream},
                           stream.value()};

  // Create hash table containing all keys found in right table
  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  auto const build_nulls = cudf::nullate::DYNAMIC{cudf::has_nulls(build)};

  // Since we may see multiple rows that are identical in the equality tables
  // but differ in the conditional tables, the equality comparator used for
  // insertion must account for both sets of tables. An alternative solution
  // would be to use a multimap, but that solution would store duplicates where
  // equality and conditional rows are equal, so this approach is preferable.
  // One way to make this solution even more efficient would be to only include
  // the columns of the conditional table that are used by the expression, but
  // that requires additional plumbing through the AST machinery and is out of
  // scope for now.
  auto const row_comparator_build =
    cudf::experimental::row::equality::two_table_comparator{preprocessed_build, preprocessed_build};
  auto const equality_build_equality =
    row_comparator_build.equal_to<false>(build_nulls, compare_nulls);
  auto const preprocessed_build_condtional =
    experimental::row::equality::preprocessed_table::create(right_conditional, stream);
  auto const row_comparator_conditional_build =
    cudf::experimental::row::equality::two_table_comparator{preprocessed_build_condtional,
                                                            preprocessed_build_condtional};
  auto const equality_build_conditional =
    row_comparator_conditional_build.equal_to<false>(build_nulls, compare_nulls);
  double_row_equality equality_build{equality_build_equality, equality_build_conditional};
  make_pair_function_semi pair_func_build{};

  auto iter     = cudf::detail::make_counting_transform_iterator(0, pair_func_build);
  auto find_any = [](std::initializer_list<cudf::type_id> ids,
                     std::set<cudf::type_id> const& search_set) {
    for (auto id : ids) {
      if (search_set.find(id) != search_set.end()) return true;
    }
    return false;
  };

  std::set<cudf::type_id> build_column_types, probe_column_types;
  for (auto col : build) {
    build_column_types.insert(col.type().id());
  }
  for (auto col : probe) {
    probe_column_types.insert(col.type().id());
  }

  // skip rows that are null here.
  if ((compare_nulls == null_equality::EQUAL) or (not nullable(build))) {
    if (find_any({type_id::STRUCT, type_id::LIST}, build_column_types)) {
      auto const row_hash_build = cudf::experimental::row::hash::row_hasher{preprocessed_build};
      auto const hash_build     = row_hash_build.device_hasher(build_nulls);
      hash_table.insert(iter, iter + right_num_rows, hash_build, equality_build, stream.value());
    } else if (find_any({type_id::DECIMAL32,
                         type_id::DECIMAL64,
                         type_id::DECIMAL128,
                         type_id::STRING,
                         type_id::DICTIONARY32},
                        build_column_types)) {
      auto const row_hash_build = cudf::experimental::row::hash::row_hasher{preprocessed_build};
      auto const hash_build     = row_hash_build.device_hasher<
        cudf::experimental::dispatch_void_conditional_generator<id_to_type<type_id::STRUCT>,
                                                                id_to_type<type_id::LIST>>::type>(
        build_nulls);
      hash_table.insert(iter, iter + right_num_rows, hash_build, equality_build, stream.value());
    } else {
      auto const row_hash_build = cudf::experimental::row::hash::row_hasher{preprocessed_build};
      auto const hash_build =
        row_hash_build.device_hasher<cudf::experimental::dispatch_void_conditional_generator<
          id_to_type<type_id::STRUCT>,
          id_to_type<type_id::LIST>,
          id_to_type<type_id::DECIMAL128>,
          id_to_type<type_id::DECIMAL64>,
          id_to_type<type_id::DECIMAL32>,
          id_to_type<type_id::STRING>,
          id_to_type<type_id::DICTIONARY32>>::type>(build_nulls);
      hash_table.insert(iter, iter + right_num_rows, hash_build, equality_build, stream.value());
    }
  } else {
    thrust::counting_iterator<cudf::size_type> stencil(0);
    auto const [row_bitmask, _] =
      cudf::detail::bitmask_and(build, stream, rmm::mr::get_current_device_resource());
    row_is_valid pred{static_cast<bitmask_type const*>(row_bitmask.data())};

    if (find_any({type_id::STRUCT, type_id::LIST}, build_column_types)) {
      auto const row_hash_build = cudf::experimental::row::hash::row_hasher{preprocessed_build};
      auto const hash_build     = row_hash_build.device_hasher(build_nulls);
      // insert valid rows
      hash_table.insert_if(
        iter, iter + right_num_rows, stencil, pred, hash_build, equality_build, stream.value());
    } else if (find_any({type_id::DECIMAL32,
                         type_id::DECIMAL64,
                         type_id::DECIMAL128,
                         type_id::STRING,
                         type_id::DICTIONARY32},
                        build_column_types)) {
      auto const row_hash_build = cudf::experimental::row::hash::row_hasher{preprocessed_build};
      auto const hash_build     = row_hash_build.device_hasher<
        cudf::experimental::dispatch_void_conditional_generator<id_to_type<type_id::STRUCT>,
                                                                id_to_type<type_id::LIST>>::type>(
        build_nulls);
      // insert valid rows
      hash_table.insert_if(
        iter, iter + right_num_rows, stencil, pred, hash_build, equality_build, stream.value());
    } else {
      auto const row_hash_build = cudf::experimental::row::hash::row_hasher{preprocessed_build};
      auto const hash_build =
        row_hash_build.device_hasher<cudf::experimental::dispatch_void_conditional_generator<
          id_to_type<type_id::STRUCT>,
          id_to_type<type_id::LIST>,
          id_to_type<type_id::DECIMAL128>,
          id_to_type<type_id::DECIMAL64>,
          id_to_type<type_id::DECIMAL32>,
          id_to_type<type_id::STRING>,
          id_to_type<type_id::DICTIONARY32>>::type>(build_nulls);
      // insert valid rows
      hash_table.insert_if(
        iter, iter + right_num_rows, stencil, pred, hash_build, equality_build, stream.value());
    }
  }

  auto hash_table_view = hash_table.get_device_view();

  detail::grid_1d const config(outer_num_rows, DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  // Vector used to indicate indices from left/probe table which are present in output
  auto left_table_keep_mask = rmm::device_uvector<bool>(probe.num_rows(), stream);

  if (find_any({type_id::STRUCT, type_id::LIST}, probe_column_types)) {
    auto const row_hash   = cudf::experimental::row::hash::row_hasher{preprocessed_probe};
    auto const hash_probe = row_hash.device_hasher(has_nulls);
    if (has_nulls) {
      mixed_join_semi<DEFAULT_JOIN_BLOCK_SIZE, true>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_conditional_view,
          *right_conditional_view,
          *probe_view,
          *build_view,
          hash_probe,
          equality_probe,
          hash_table_view,
          cudf::device_span<bool>(left_table_keep_mask),
          parser.device_expression_data);
    } else {
      mixed_join_semi<DEFAULT_JOIN_BLOCK_SIZE, false>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_conditional_view,
          *right_conditional_view,
          *probe_view,
          *build_view,
          hash_probe,
          equality_probe,
          hash_table_view,
          cudf::device_span<bool>(left_table_keep_mask),
          parser.device_expression_data);
    }
  } else if (find_any({type_id::DECIMAL32,
                       type_id::DECIMAL64,
                       type_id::DECIMAL128,
                       type_id::STRING,
                       type_id::DICTIONARY32},
                      probe_column_types)) {
    auto const row_hash   = cudf::experimental::row::hash::row_hasher{preprocessed_probe};
    auto const hash_probe = row_hash.device_hasher<
      cudf::experimental::dispatch_void_conditional_generator<id_to_type<type_id::STRUCT>,
                                                              id_to_type<type_id::LIST>>::type>(
      has_nulls);
    if (has_nulls) {
      mixed_join_semi<DEFAULT_JOIN_BLOCK_SIZE, true>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_conditional_view,
          *right_conditional_view,
          *probe_view,
          *build_view,
          hash_probe,
          equality_probe,
          hash_table_view,
          cudf::device_span<bool>(left_table_keep_mask),
          parser.device_expression_data);
    } else {
      mixed_join_semi<DEFAULT_JOIN_BLOCK_SIZE, false>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_conditional_view,
          *right_conditional_view,
          *probe_view,
          *build_view,
          hash_probe,
          equality_probe,
          hash_table_view,
          cudf::device_span<bool>(left_table_keep_mask),
          parser.device_expression_data);
    }
  } else {
    auto const row_hash = cudf::experimental::row::hash::row_hasher{preprocessed_probe};
    auto const hash_probe =
      row_hash.device_hasher<cudf::experimental::dispatch_void_conditional_generator<
        id_to_type<type_id::STRUCT>,
        id_to_type<type_id::LIST>,
        id_to_type<type_id::DECIMAL128>,
        id_to_type<type_id::DECIMAL64>,
        id_to_type<type_id::DECIMAL32>,
        id_to_type<type_id::STRING>,
        id_to_type<type_id::DICTIONARY32>>::type>(has_nulls);
    if (has_nulls) {
      mixed_join_semi<DEFAULT_JOIN_BLOCK_SIZE, true>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_conditional_view,
          *right_conditional_view,
          *probe_view,
          *build_view,
          hash_probe,
          equality_probe,
          hash_table_view,
          cudf::device_span<bool>(left_table_keep_mask),
          parser.device_expression_data);
    } else {
      mixed_join_semi<DEFAULT_JOIN_BLOCK_SIZE, false>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_conditional_view,
          *right_conditional_view,
          *probe_view,
          *build_view,
          hash_probe,
          equality_probe,
          hash_table_view,
          cudf::device_span<bool>(left_table_keep_mask),
          parser.device_expression_data);
    }
  }

  auto gather_map = std::make_unique<rmm::device_uvector<size_type>>(probe.num_rows(), stream, mr);

  // gather_map_end will be the end of valid data in gather_map
  auto gather_map_end =
    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::counting_iterator<size_type>(0),
                    thrust::counting_iterator<size_type>(probe.num_rows()),
                    left_table_keep_mask.begin(),
                    gather_map->begin(),
                    [join_type] __device__(bool keep_row) {
                      return keep_row == (join_type == detail::join_kind::LEFT_SEMI_JOIN);
                    });

  gather_map->resize(thrust::distance(gather_map->begin(), gather_map_end), stream);
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
                                 cudf::get_default_stream(),
                                 mr);
}

}  // namespace cudf
