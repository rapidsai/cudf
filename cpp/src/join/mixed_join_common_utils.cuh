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
#pragma once

#include "join/join_common_utils.hpp"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuco/static_multiset.cuh>
#include <cuco/static_set.cuh>

namespace cudf {
namespace detail {

using row_hash =
  cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                   cudf::nullate::DYNAMIC>;

// // This alias is used by mixed_joins, which support only non-nested types
using row_equality = cudf::experimental::row::equality::strong_index_comparator_adapter<
  cudf::experimental::row::equality::device_row_comparator<false, cudf::nullate::DYNAMIC>>;

// Comparator that always returns false to ensure all values are inserted (like hash_join)
struct mixed_join_always_not_equal {
  __device__ constexpr bool operator()(cuco::pair<hash_value_type, size_type> const&,
                                       cuco::pair<hash_value_type, size_type> const&) const noexcept
  {
    // multiset always insert
    return false;
  }
};

// hasher1 and hasher2 used for double hashing. The first hash is used to determine the initial slot
// and the second hash is used to determine the step size.
//
// For the first hash, we use the row hash value directly so there is no need to hash it again.
//
// For the second hash, we hash the row hash value again to determine the step size.
struct mixed_join_hasher1 {
  __device__ constexpr hash_value_type operator()(
    cuco::pair<hash_value_type, size_type> const& key) const noexcept
  {
    return key.first;
  }
};

struct mixed_join_hasher2 {
  mixed_join_hasher2(hash_value_type seed) : _hash{seed} {}

  __device__ constexpr hash_value_type operator()(
    cuco::pair<hash_value_type, size_type> const& key) const noexcept
  {
    return _hash(key.first);
  }

 private:
  using hash_type = cuco::murmurhash3_32<hash_value_type>;
  hash_type _hash;
};

// Hash table type used for mixed joins
using mixed_join_hash_table_t =
  cuco::static_multiset<cuco::pair<hash_value_type, size_type>,
                        cuco::extent<std::size_t>,
                        cuda::thread_scope_device,
                        mixed_join_always_not_equal,
                        cuco::double_hashing<1, mixed_join_hasher1, mixed_join_hasher2>,
                        cudf::detail::cuco_allocator<char>,
                        cuco::storage<2>>;
template <typename Tag>
using mixed_join_hash_table_ref_t = mixed_join_hash_table_t::ref_type<Tag>;

/**
 * @brief Equality comparator for use with cuco map methods that require expression evaluation.
 *
 * This class just defines the construction of the class and the necessary
 * attributes, specifically the equality operator for the non-conditional parts
 * of the operator and the evaluator used for the conditional.
 */
template <bool has_nulls>
struct expression_equality {
  __device__ expression_equality(
    cudf::ast::detail::expression_evaluator<has_nulls> const& evaluator,
    cudf::ast::detail::IntermediateDataType<has_nulls>* thread_intermediate_storage,
    bool const swap_tables,
    row_equality const& equality_probe)
    : evaluator{evaluator},
      thread_intermediate_storage{thread_intermediate_storage},
      swap_tables{swap_tables},
      equality_probe{equality_probe}
  {
  }

  cudf::ast::detail::IntermediateDataType<has_nulls>* thread_intermediate_storage;
  cudf::ast::detail::expression_evaluator<has_nulls> const& evaluator;
  bool const swap_tables;
  row_equality const& equality_probe;
};

/**
 * @brief Equality comparator for cuco::static_map queries.
 *
 * This equality comparator is designed for use with cuco::static_map's APIs. A
 * probe hit indicates that the hashes of the keys are equal, at which point
 * this comparator checks whether the keys themselves are equal (using the
 * provided equality_probe) and then evaluates the conditional expression
 */
template <bool has_nulls>
struct single_expression_equality : expression_equality<has_nulls> {
  using expression_equality<has_nulls>::expression_equality;

  __device__ __forceinline__ bool operator()(size_type const left_index,
                                             size_type const right_index) const noexcept
  {
    using cudf::experimental::row::lhs_index_type;
    using cudf::experimental::row::rhs_index_type;

    auto output_dest = cudf::ast::detail::value_expression_result<bool, has_nulls>();
    // Two levels of checks:
    // 1. The contents of the columns involved in the equality condition are equal.
    // 2. The predicate evaluated on the relevant columns (already encoded in the evaluator)
    // evaluates to true.
    if (this->equality_probe(lhs_index_type{left_index}, rhs_index_type{right_index})) {
      // For the AST evaluator, we need to map back to left/right table semantics
      auto const left_table_idx  = this->swap_tables ? right_index : left_index;
      auto const right_table_idx = this->swap_tables ? left_index : right_index;
      this->evaluator.evaluate(output_dest,
                               static_cast<size_type>(left_table_idx),
                               static_cast<size_type>(right_table_idx),
                               0,
                               this->thread_intermediate_storage);
      return (output_dest.is_valid() && output_dest.value());
    }
    return false;
  }
};

/**
 * @brief Equality comparator for cuco::static_multimap queries.
 *
 * This equality comparator is designed for use with cuco::static_multimap's
 * pair* APIs, which will compare equality based on comparing (key, value)
 * pairs. In the context of joins, these pairs are of the form
 * (row_hash, row_id). A hash probe hit indicates that hash of a probe row's hash is
 * equal to the hash of the hash of some row in the multimap, at which point we need an
 * equality comparator that will check whether the contents of the rows are
 * identical. This comparator does so by verifying key equality (i.e. that
 * probe_row_hash == build_row_hash) and then using a row_equality_comparator
 * to compare the contents of the row indices that are stored as the payload in
 * the hash map.
 */
template <bool has_nulls>
struct pair_expression_equality : public expression_equality<has_nulls> {
  using expression_equality<has_nulls>::expression_equality;

  __device__ __forceinline__ bool operator()(pair_type const& left_row,
                                             pair_type const& right_row) const noexcept
  {
    using cudf::experimental::row::lhs_index_type;
    using cudf::experimental::row::rhs_index_type;

    auto output_dest = cudf::ast::detail::value_expression_result<bool, has_nulls>();
    // Three levels of checks:
    // 1. Row hashes of the columns involved in the equality condition are equal.
    // 2. The contents of the columns involved in the equality condition are equal.
    // 3. The predicate evaluated on the relevant columns (already encoded in the evaluator)
    // evaluates to true.
    if ((left_row.first == right_row.first) &&
        this->equality_probe(lhs_index_type{left_row.second}, rhs_index_type{right_row.second})) {
      auto const lrow_idx = this->swap_tables ? right_row.second : left_row.second;
      auto const rrow_idx = this->swap_tables ? left_row.second : right_row.second;
      this->evaluator.evaluate(
        output_dest, lrow_idx, rrow_idx, 0, this->thread_intermediate_storage);
      return (output_dest.is_valid() && output_dest.value());
    }
    return false;
  }
};

/**
 * @brief Equality comparator that composes two row_equality comparators.
 */
struct double_row_equality_comparator {
  row_equality const equality_comparator;
  row_equality const conditional_comparator;

  __device__ bool operator()(size_type lhs_row_index, size_type rhs_row_index) const noexcept
  {
    using experimental::row::lhs_index_type;
    using experimental::row::rhs_index_type;

    return equality_comparator(lhs_index_type{lhs_row_index}, rhs_index_type{rhs_row_index}) &&
           conditional_comparator(lhs_index_type{lhs_row_index}, rhs_index_type{rhs_row_index});
  }
};

// A CUDA Cooperative Group of 1 thread for the hash set for mixed semi.
auto constexpr DEFAULT_MIXED_SEMI_JOIN_CG_SIZE = 1;

// The hash set type used by mixed_semi_join with the build_table.
using hash_set_type =
  cuco::static_set<size_type,
                   cuco::extent<size_t>,
                   cuda::thread_scope_device,
                   double_row_equality_comparator,
                   cuco::linear_probing<DEFAULT_MIXED_SEMI_JOIN_CG_SIZE, row_hash>,
                   cudf::detail::cuco_allocator<char>,
                   cuco::storage<1>>;

// The hash_set_ref_type used by mixed_semi_join kerenels for probing.
using hash_set_ref_type = hash_set_type::ref_type<cuco::contains_tag>;

}  // namespace detail

}  // namespace cudf
