/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/binaryop.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include "cudf/column/column_factories.hpp"
#include "cudf/table/table_device_view.cuh"
#include "cudf/types.hpp"

namespace cudf {

enum class ast_data_source {
  COLUMN,       // A value from a column in a table
  LITERAL,      // A constant value
  INTERMEDIATE  // An internal node (not a leaf) in the AST
};

enum class ast_binary_operator {
  ADD,      // Addition
  SUBTRACT  // Subtraction
};

enum class ast_comparator {
  LESS,    // x < y
  GREATER  // x > y
};

enum class ast_table_source {
  LEFT,  // Column index in the left table
  RIGHT  // Column index in the right table
};

struct ast_expression_source {
  ast_data_source source;  // Source of data
  cudf::size_type
    data_index;  // The column index of a table, index of a literal, or index of an intermediate
  ast_table_source table_source = ast_table_source::LEFT;  // Left or right table (if applicable)
};

template <typename Element>
struct ast_binary_expression {
  ast_binary_operator op;
  ast_expression_source lhs;
  ast_expression_source rhs;
};

template <typename Element>
struct ast_comparator_expression {
  ast_comparator op;
  ast_expression_source lhs;
  ast_expression_source rhs;
};

template <ast_binary_operator>
struct binop {
};

template <>
struct binop<ast_binary_operator::ADD> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs + rhs;
  }
};

template <>
struct binop<ast_binary_operator::SUBTRACT> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs - rhs;
  }
};

template <typename F, typename... Ts>
__device__ decltype(auto) binop_dispatcher(ast_binary_operator op, F&& f, Ts&&... args)
{
  switch (op) {
    case ast_binary_operator::ADD:
      return f.template operator()<ast_binary_operator::ADD>(std::forward<Ts>(args)...);
    case ast_binary_operator::SUBTRACT:
      return f.template operator()<ast_binary_operator::SUBTRACT>(std::forward<Ts>(args)...);
    default: return 0;  // TODO: Error handling
  }
}

template <typename T>
struct do_binop {
  template <ast_binary_operator OP>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return binop<OP>{}(lhs, rhs);
  }
};

template <ast_comparator>
struct compareop {
};

template <>
struct compareop<ast_comparator::LESS> {
  template <typename T>
  __device__ bool operator()(T const& lhs, T const& rhs)
  {
    return lhs < rhs;
  }
};

template <>
struct compareop<ast_comparator::GREATER> {
  template <typename T>
  __device__ bool operator()(T const& lhs, T const& rhs)
  {
    return lhs > rhs;
  }
};

template <typename F, typename... Ts>
__device__ decltype(auto) compareop_dispatcher(ast_comparator op, F&& f, Ts&&... args)
{
  switch (op) {
    case ast_comparator::LESS:
      return f.template operator()<ast_comparator::LESS>(std::forward<Ts>(args)...);
    case ast_comparator::GREATER:
      return f.template operator()<ast_comparator::GREATER>(std::forward<Ts>(args)...);
    default: return false;  // TODO: Error handling
  }
}

template <typename T>
struct do_compareop {
  template <ast_comparator OP>
  __device__ bool operator()(T const& lhs, T const& rhs)
  {
    return compareop<OP>{}(lhs, rhs);
  }
};

template <typename Element>
__device__ Element ast_resolve_data_source(ast_expression_source expression_source,
                                           table_device_view const& table,
                                           cudf::size_type row_index)
{
  switch (expression_source.source) {
    case ast_data_source::COLUMN: {
      auto column = table.column(expression_source.data_index);
      return column.data<Element>()[row_index];
    }
    case ast_data_source::LITERAL: {
      // TODO: Fetch and return literal.
      return static_cast<Element>(0);
    }
    case ast_data_source::INTERMEDIATE: {
      // TODO: Fetch and return intermediate.
      return static_cast<Element>(0);
    }
    default: {
      // TODO: Error
      return static_cast<Element>(0);
    }
  }
}

template <typename Element>
__device__ Element ast_resolve_data_source(ast_expression_source expression_source,
                                           table_device_view const& left_table,
                                           table_device_view const& right_table,
                                           cudf::size_type left_row_index,
                                           cudf::size_type right_row_index)
{
  switch (expression_source.source) {
    case ast_data_source::COLUMN: {
      if (expression_source.table_source == ast_table_source::LEFT) {
        auto column = left_table.column(expression_source.data_index);
        return column.data<Element>()[left_row_index];
      } else {
        auto column = right_table.column(expression_source.data_index);
        return column.data<Element>()[right_row_index];
      }
    }
    case ast_data_source::LITERAL: {
      // TODO: Fetch and return literal.
      return static_cast<Element>(0);
    }
    case ast_data_source::INTERMEDIATE: {
      // TODO: Fetch and return intermediate.
      return static_cast<Element>(0);
    }
    default: {
      // TODO: Error
      return static_cast<Element>(0);
    }
  }
}

/*
template <typename Element>
__device__ Element ast_evaluate_operator(ast_binary_operator op, Element lhs, Element rhs)
{
  switch (op) {
    case ast_binary_operator::ADD: return lhs + rhs;
    case ast_binary_operator::SUBTRACT: return lhs - rhs;
    default:
      // TODO: Error
      return 0;
  }
}

template <typename Element>
__device__ bool ast_evaluate_operator(ast_comparator op, Element lhs, Element rhs)
{
  switch (op) {
    case ast_comparator::LESS: return lhs < rhs;
    case ast_comparator::GREATER: return lhs > rhs;
    default:
      // TODO: Error
      return 0;
  }
}
*/

template <typename Element>
__device__ Element ast_evaluate_expression(ast_binary_expression<Element> binary_expression,
                                           table_device_view table,
                                           cudf::size_type row_index)
{
  const Element lhs = ast_resolve_data_source<Element>(binary_expression.lhs, table, row_index);
  const Element rhs = ast_resolve_data_source<Element>(binary_expression.rhs, table, row_index);
  return binop_dispatcher(binary_expression.op, do_binop<Element>{}, lhs, rhs);
}

template <typename Element>
__device__ Element ast_evaluate_expression(ast_binary_expression<Element> expression,
                                           table_device_view left_table,
                                           table_device_view right_table,
                                           cudf::size_type left_row_index,
                                           cudf::size_type right_row_index)
{
  const Element lhs = ast_resolve_data_source<Element>(
    expression.lhs, left_table, right_table, left_row_index, right_row_index);
  const Element rhs = ast_resolve_data_source<Element>(
    expression.rhs, left_table, right_table, left_row_index, right_row_index);
  return binop_dispatcher(expression.op, do_binop<Element>{}, lhs, rhs);
}

template <typename Element>
__device__ bool ast_evaluate_expression(ast_comparator_expression<Element> expression,
                                        table_device_view table,
                                        cudf::size_type row_index)
{
  const Element lhs = ast_resolve_data_source<Element>(expression.lhs, table, row_index);
  const Element rhs = ast_resolve_data_source<Element>(expression.rhs, table, row_index);
  return compareop_dispatcher(expression.op, do_compareop<Element>{}, lhs, rhs);
}

template <typename Element>
__device__ bool ast_evaluate_expression(ast_comparator_expression<Element> expression,
                                        table_device_view left_table,
                                        table_device_view right_table,
                                        cudf::size_type left_row_index,
                                        cudf::size_type right_row_index)
{
  const Element lhs = ast_resolve_data_source<Element>(
    expression.lhs, left_table, right_table, left_row_index, right_row_index);
  const Element rhs = ast_resolve_data_source<Element>(
    expression.rhs, left_table, right_table, left_row_index, right_row_index);
  return compareop_dispatcher(expression.op, do_compareop<Element>{}, lhs, rhs);
}

template <typename Element>
__global__ void compute_ast_column_kernel(table_device_view table,
                                          ast_binary_expression<Element> expression,
                                          mutable_column_device_view output)
{
  const cudf::size_type start_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const cudf::size_type stride    = blockDim.x * gridDim.x;
  const auto num_rows             = table.num_rows();

  for (cudf::size_type row_index = start_idx; row_index < num_rows; row_index += stride) {
    output.element<Element>(row_index) = ast_evaluate_expression(expression, table, row_index);
  }
}

template <typename Element>
__global__ void compute_ast_column_kernel(table_device_view table,
                                          ast_comparator_expression<Element> expression,
                                          mutable_column_device_view output)
{
  const cudf::size_type start_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const cudf::size_type stride    = blockDim.x * gridDim.x;
  const auto num_rows             = table.num_rows();

  for (cudf::size_type row_index = start_idx; row_index < num_rows; row_index += stride) {
    output.element<bool>(row_index) = ast_evaluate_expression(expression, table, row_index);
  }
}

template <typename Element>
std::unique_ptr<column> compute_ast_column(
  table_view const& table,
  ast_binary_expression<Element> expression,
  cudaStream_t stream                 = 0,  // TODO use detail API
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  auto table_device      = table_device_view::create(table, stream);
  auto element_data_type = cudf::data_type(cudf::type_to_id<Element>());
  auto table_num_rows    = table.num_rows();
  auto output_column =
    make_fixed_width_column(element_data_type, table_num_rows, mask_state::UNALLOCATED, stream, mr);
  auto block_size = 1024;  // TODO dynamically determine block size, use shared memory
  auto mutable_output_device =
    cudf::mutable_column_device_view::create(output_column->mutable_view(), stream);

  detail::grid_1d config(table_num_rows, block_size);
  compute_ast_column_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
    *table_device, expression, *mutable_output_device);
  return output_column;
}

template <typename Element>
std::unique_ptr<column> compute_ast_column(
  table_view const& table,
  ast_comparator_expression<Element> expression,
  cudaStream_t stream                 = 0,  // TODO use detail API
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  auto table_device      = table_device_view::create(table, stream);
  auto element_data_type = cudf::data_type(cudf::type_to_id<bool>());
  auto table_num_rows    = table.num_rows();
  auto output_column =
    make_fixed_width_column(element_data_type, table_num_rows, mask_state::UNALLOCATED, stream, mr);
  auto block_size = 1024;  // TODO dynamically determine block size, use shared memory
  auto mutable_output_device =
    cudf::mutable_column_device_view::create(output_column->mutable_view(), stream);

  detail::grid_1d config(table_num_rows, block_size);
  compute_ast_column_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
    *table_device, expression, *mutable_output_device);
  return output_column;
}

}  // namespace cudf
