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
#pragma once

#include <join/join_common_utils.hpp>

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/detail/utilities/cuda.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

namespace cudf {
namespace detail {

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
 * @brief Device functor to create a pair of hash value and index for a given row.
 */
struct make_pair_function_semi {
  __device__ __forceinline__ cudf::detail::pair_type operator()(size_type i) const noexcept
  {
    // The value is irrelevant since we only ever use the hash map to check for
    // membership of a particular row index.
    return cuco::make_pair<hash_value_type, size_type>(i, 0);
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

}  // namespace detail

}  // namespace cudf
