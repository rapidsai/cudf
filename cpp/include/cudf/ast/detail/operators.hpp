/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cudf/ast/expressions.hpp>
#include <cudf/types.hpp>

#include <vector>

namespace CUDF_EXPORT cudf {
namespace ast::detail {

CUDF_HOST_DEVICE constexpr bool is_complex_type(cudf::type_id type)
{
  return type == cudf::type_id::DECIMAL32 || type == cudf::type_id::DECIMAL64 ||
         type == cudf::type_id::DECIMAL128 || type == cudf::type_id::STRING;
}

/**
 * @brief Gets the return type of an AST operator.
 *
 * @param op Operator used to evaluate return type.
 * @param operand_types Vector of input types to the operator.
 * @return cudf::data_type Return type of the operator.
 */
cudf::data_type ast_operator_return_type(ast_operator op,
                                         std::vector<cudf::data_type> const& operand_types);

/**
 * @brief Gets the arity (number of operands) of an AST operator.
 *
 * @param op Operator used to determine arity.
 * @return Arity of the operator.
 */
cudf::size_type ast_operator_arity(ast_operator op);

std::string_view ast_operator_string(ast_operator op);

}  // namespace ast::detail

}  // namespace CUDF_EXPORT cudf
