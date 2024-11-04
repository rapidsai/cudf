/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <cudf/ast/detail/operators.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/type_traits>
#include <thrust/optional.h>

#include <vector>

namespace cudf {
namespace ast {
namespace detail {
namespace {

struct arity_functor {
  template <ast_operator op>
  cudf::size_type operator()(cudf::size_type& result)
  {
    // Arity is not dependent on null handling, so just use the false implementation here.
    return operator_functor<op, false>::arity;
  }
};

/**
 * @brief Functor to determine the return type of an operator from its input types.
 */
struct return_type_functor {
  /**
   * @brief Callable for binary operators to determine return type.
   *
   * @tparam OperatorFunctor Operator functor to perform.
   * @tparam LHS Left input type.
   * @tparam RHS Right input type.
   * @param result Reference whose value is assigned to the result data type.
   */
  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            std::enable_if_t<is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDF_HOST_DEVICE inline void operator()(cudf::data_type& result)
  {
    using Out = cuda::std::invoke_result_t<OperatorFunctor, LHS, RHS>;
    result    = cudf::data_type(cudf::type_to_id<Out>());
  }

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            std::enable_if_t<!is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDF_HOST_DEVICE inline void operator()(cudf::data_type& result)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid binary operation. Return type cannot be determined.");
#else
    CUDF_UNREACHABLE("Invalid binary operation. Return type cannot be determined.");
#endif
  }

  /**
   * @brief Callable for unary operators to determine return type.
   *
   * @tparam OperatorFunctor Operator functor to perform.
   * @tparam T Input type.
   * @param result Pointer whose value is assigned to the result data type.
   */
  template <typename OperatorFunctor,
            typename T,
            std::enable_if_t<is_valid_unary_op<OperatorFunctor, T>>* = nullptr>
  CUDF_HOST_DEVICE inline void operator()(cudf::data_type& result)
  {
    using Out = cuda::std::invoke_result_t<OperatorFunctor, T>;
    result    = cudf::data_type(cudf::type_to_id<Out>());
  }

  template <typename OperatorFunctor,
            typename T,
            std::enable_if_t<!is_valid_unary_op<OperatorFunctor, T>>* = nullptr>
  CUDF_HOST_DEVICE inline void operator()(cudf::data_type& result)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid unary operation. Return type cannot be determined.");
#else
    CUDF_UNREACHABLE("Invalid unary operation. Return type cannot be determined.");
#endif
  }
};

}  // namespace

cudf::data_type ast_operator_return_type(ast_operator op,
                                         std::vector<cudf::data_type> const& operand_types)
{
  auto result = cudf::data_type(cudf::type_id::EMPTY);
  switch (operand_types.size()) {
    case 1: return unary_operator_dispatcher(op, operand_types[0], detail::return_type_functor{});
    case 2:
      return binary_operator_dispatcher(
        op, operand_types[0], operand_types[1], detail::return_type_functor{});
    default: CUDF_FAIL("Unsupported operator return type."); break;
  }
  return result;
}

cudf::size_type ast_operator_arity(ast_operator op)
{
  return ast_operator_dispatcher(op, arity_functor{});
}

}  // namespace detail

}  // namespace ast

}  // namespace cudf
