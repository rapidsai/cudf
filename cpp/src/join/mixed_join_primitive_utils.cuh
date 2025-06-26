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
#include <cudf/table/primitive_row_operators.cuh>
#include <cudf/types.hpp>

namespace cudf::detail {

/**
 * @brief Equality comparator for use with cuco map methods that require expression evaluation
 * using primitive row operators.
 */
template <bool has_nulls>
struct primitive_expression_equality {
  __device__ primitive_expression_equality(
    cudf::ast::detail::expression_evaluator<has_nulls> const& evaluator,
    cudf::ast::detail::IntermediateDataType<has_nulls>* thread_intermediate_storage,
    bool const swap_tables,
    cudf::row::primitive::row_equality_comparator const& equality_probe)
    : evaluator{evaluator},
      thread_intermediate_storage{thread_intermediate_storage},
      swap_tables{swap_tables},
      equality_probe{equality_probe}
  {
  }

  cudf::ast::detail::IntermediateDataType<has_nulls>* thread_intermediate_storage;
  cudf::ast::detail::expression_evaluator<has_nulls> const& evaluator;
  bool const swap_tables;
  cudf::row::primitive::row_equality_comparator const& equality_probe;
};

/**
 * @brief Equality comparator for cuco::static_map queries using primitive row operators.
 */
template <bool has_nulls>
struct primitive_single_expression_equality : primitive_expression_equality<has_nulls> {
  using primitive_expression_equality<has_nulls>::primitive_expression_equality;

  __device__ __forceinline__ bool operator()(hash_value_type const build_row_index,
                                             hash_value_type const probe_row_index) const noexcept
  {
    auto output_dest = cudf::ast::detail::value_expression_result<bool, has_nulls>();
    // Two levels of checks:
    // 1. The contents of the columns involved in the equality condition are equal.
    // 2. The predicate evaluated on the relevant columns (already encoded in the evaluator)
    // evaluates to true.
    if (this->equality_probe(static_cast<size_type>(probe_row_index),
                             static_cast<size_type>(build_row_index))) {
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
 * @brief Primitive equality comparator that composes two primitive row_equality comparators.
 */
struct primitive_double_row_equality_comparator {
  cudf::row::primitive::row_equality_comparator const equality_comparator;
  cudf::row::primitive::row_equality_comparator const conditional_comparator;

  __device__ bool operator()(size_type lhs_row_index, size_type rhs_row_index) const noexcept
  {
    return equality_comparator(lhs_row_index, rhs_row_index) &&
           conditional_comparator(lhs_row_index, rhs_row_index);
  }
};

}  // namespace cudf::detail
