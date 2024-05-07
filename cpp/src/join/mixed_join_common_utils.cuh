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
#pragma once

#include "join/join_common_utils.hpp"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

namespace cudf {
namespace detail {

using row_hash =
  cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                   cudf::nullate::DYNAMIC,
                                                   cudf::experimental::type_identity>;

using row_hash_no_nested = cudf::experimental::row::hash::device_row_hasher<
  cudf::hashing::detail::default_hash,
  cudf::nullate::DYNAMIC,
  cudf::experimental::dispatch_void_conditional_generator<id_to_type<type_id::STRUCT>,
                                                          id_to_type<type_id::LIST>>::type>;

using row_hash_no_complex = cudf::experimental::row::hash::device_row_hasher<
  cudf::hashing::detail::default_hash,
  cudf::nullate::DYNAMIC,
  cudf::experimental::dispatch_void_conditional_generator<id_to_type<type_id::STRUCT>,
                                                          id_to_type<type_id::LIST>,
                                                          id_to_type<type_id::DECIMAL128>,
                                                          id_to_type<type_id::DECIMAL64>,
                                                          id_to_type<type_id::DECIMAL32>,
                                                          id_to_type<type_id::STRING>,
                                                          id_to_type<type_id::DICTIONARY32>>::type>;

// // This alias is used by mixed_joins, which support only non-nested types
using row_equality = cudf::experimental::row::equality::strong_index_comparator_adapter<
  cudf::experimental::row::equality::device_row_comparator<false, cudf::nullate::DYNAMIC>>;

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

  // The parameters are build/probe rather than left/right because the operator
  // is called by cuco's kernels with parameters in this order (note that this
  // is an implementation detail that we should eventually stop relying on by
  // defining operators with suitable heterogeneous typing). Rather than
  // converting to left/right semantics, we can operate directly on build/probe
  // until we get to the expression evaluator, which needs to convert back to
  // left/right semantics because the conditional expression need not be
  // commutative.
  // TODO: The input types should really be size_type.
  __device__ __forceinline__ bool operator()(hash_value_type const build_row_index,
                                             hash_value_type const probe_row_index) const noexcept
  {
    using cudf::experimental::row::lhs_index_type;
    using cudf::experimental::row::rhs_index_type;

    auto output_dest = cudf::ast::detail::value_expression_result<bool, has_nulls>();
    // Two levels of checks:
    // 1. The contents of the columns involved in the equality condition are equal.
    // 2. The predicate evaluated on the relevant columns (already encoded in the evaluator)
    // evaluates to true.
    if (this->equality_probe(lhs_index_type{probe_row_index}, rhs_index_type{build_row_index})) {
      auto const lrow_idx = this->swap_tables ? build_row_index : probe_row_index;
      auto const rrow_idx = this->swap_tables ? probe_row_index : build_row_index;
      this->evaluator.evaluate(output_dest,
                               static_cast<size_type>(lrow_idx),
                               static_cast<size_type>(rrow_idx),
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

  // The parameters are build/probe rather than left/right because the operator
  // is called by cuco's kernels with parameters in this order (note that this
  // is an implementation detail that we should eventually stop relying on by
  // defining operators with suitable heterogeneous typing). Rather than
  // converting to left/right semantics, we can operate directly on build/probe
  // until we get to the expression evaluator, which needs to convert back to
  // left/right semantics because the conditional expression need not be
  // commutative.
  __device__ __forceinline__ bool operator()(pair_type const& build_row,
                                             pair_type const& probe_row) const noexcept
  {
    using cudf::experimental::row::lhs_index_type;
    using cudf::experimental::row::rhs_index_type;

    auto output_dest = cudf::ast::detail::value_expression_result<bool, has_nulls>();
    // Three levels of checks:
    // 1. Row hashes of the columns involved in the equality condition are equal.
    // 2. The contents of the columns involved in the equality condition are equal.
    // 3. The predicate evaluated on the relevant columns (already encoded in the evaluator)
    // evaluates to true.
    if ((probe_row.first == build_row.first) &&
        this->equality_probe(lhs_index_type{probe_row.second}, rhs_index_type{build_row.second})) {
      auto const lrow_idx = this->swap_tables ? build_row.second : probe_row.second;
      auto const rrow_idx = this->swap_tables ? probe_row.second : build_row.second;
      this->evaluator.evaluate(
        output_dest, lrow_idx, rrow_idx, 0, this->thread_intermediate_storage);
      return (output_dest.is_valid() && output_dest.value());
    }
    return false;
  }
};

}  // namespace detail

}  // namespace cudf
