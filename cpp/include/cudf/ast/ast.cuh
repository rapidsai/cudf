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

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include "cudf/column/column_factories.hpp"
#include "cudf/table/table_device_view.cuh"
#include "cudf/types.hpp"
#include "operators.cuh"

namespace cudf {

namespace ast {

enum class table_reference {
  LEFT,   // Column index in the left table
  RIGHT,  // Column index in the right table
  OUTPUT  // Column index in the output table
};

namespace detail {

enum class device_data_reference_type {
  COLUMN,       // A value in a table column
  LITERAL,      // A literal value
  INTERMEDIATE  // An internal temporary value
};

struct device_data_reference {
  device_data_reference_type reference_type;  // Source of data
  cudf::size_type
    data_index;  // The column index of a table, index of a literal, or index of an intermediate
  table_reference table_reference = table_reference::LEFT;
};

}  // namespace detail

enum class data_reference {
  COLUMN,  // A value from a column in a table
  LITERAL  // A literal value
};

struct expression_source {
  data_reference source;  // Source of data
  cudf::size_type
    data_index;  // The column index of a table, index of a literal, or index of an intermediate
  table_reference table_source = table_reference::LEFT;  // Left or right table (if applicable)
};

// Forward declaration
class abstract_visitor;
class literal;
class operator_expression;

// Visitor pattern. The visitor tracks a node index as it traverses the tree.
class expression {
 public:
  virtual cudf::size_type accept(abstract_visitor& visitor) const = 0;
};

class abstract_visitor {
 public:
  virtual cudf::size_type visit(const literal* expr)             = 0;
  virtual cudf::size_type visit(const operator_expression* expr) = 0;
};

class literal : public expression {
 public:
  literal(const cudf::scalar& value) : value(std::cref(value)) {}
  std::reference_wrapper<const cudf::scalar> get_value() const { return this->value; }
  cudf::size_type accept(abstract_visitor& visitor) const override { return visitor.visit(this); }

 private:
  std::reference_wrapper<const cudf::scalar> value;
};

class operator_expression : public expression {
 public:
  ast_operator get_operator() const { return this->op; }
  virtual std::vector<std::shared_ptr<expression>> get_operands() const = 0;

 protected:
  operator_expression(ast_operator op) : op(op) {}
  const ast_operator op;
};

class binary_expression : public operator_expression {
 public:
  binary_expression(ast_operator op,
                    std::shared_ptr<expression> left,
                    std::shared_ptr<expression> right)
    : operator_expression(op), left(left), right(right)
  {
  }
  std::shared_ptr<expression> get_left() const { return this->left; }
  std::shared_ptr<expression> get_right() const { return this->right; }
  std::vector<std::shared_ptr<expression>> get_operands() const override
  {
    return std::vector<std::shared_ptr<expression>>{this->get_left(), this->get_right()};
  }
  cudf::size_type accept(abstract_visitor& visitor) const override { return visitor.visit(this); }

 private:
  const std::shared_ptr<expression> left;
  const std::shared_ptr<expression> right;
};

class linearizer : public abstract_visitor {
 public:
  linearizer() : node_counter(0) {}

  cudf::size_type visit(const literal* expr) override
  {
    std::cout << "visiting literal value" << std::endl;
    auto data_type = expr->get_value().get().type();
    this->literals.push_back(expr->get_value());
    auto source = detail::device_data_reference{
      detail::device_data_reference_type::LITERAL,
      0  // TODO: Use correct index
    };
    this->add_source(data_type, source);
    node_counter++;
    return node_counter;
  }
  cudf::size_type visit(const operator_expression* expr) override
  {
    std::cout << "visiting operator_expression" << std::endl;
    this->visit_operands(expr->get_operands());
    node_counter++;
    return node_counter;
  }

  cudf::data_type get_root_data_type() const
  {
    if (this->node_data_types.size() == 0) {
      return cudf::data_type(cudf::type_id::EMPTY);
    } else {
      return this->node_data_types.back();
    }
  }

 private:
  void visit_operands(std::vector<std::shared_ptr<expression>> operands)
  {
    std::vector<cudf::size_type> reference_indices;
    for (auto& operand : operands) {
      if (operand != nullptr) {
        reference_index = operand->accept(*this);
        node_counter++;
      }
    }
  }
  void add_source(cudf::data_type data_type, detail::device_data_reference source)
  {
    this->node_data_types.push_back(data_type);
    this->data_references.push_back(source);
  }
  // State information about the "linearized" GPU execution plan
  cudf::size_type node_counter;
  std::vector<cudf::data_type> node_data_types;
  std::vector<detail::device_data_reference> data_references;
  std::vector<ast_operator> operators;
  std::vector<cudf::size_type> operator_source_indices;
  std::vector<std::reference_wrapper<const scalar>> literals;
};

template <typename Element>
__device__ Element resolve_data_source(detail::device_data_reference device_data_reference,
                                       table_device_view const& table,
                                       cudf::size_type row_index)
{
  switch (device_data_reference.reference_type) {
    case detail::device_data_reference_type::COLUMN: {
      auto column = table.column(device_data_reference.data_index);
      return column.data<Element>()[row_index];
    }
    case detail::device_data_reference_type::LITERAL: {
      // TODO: Fetch and return literal.
      return static_cast<Element>(0);
    }
    case detail::device_data_reference_type::INTERMEDIATE: {
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
__device__ Element resolve_data_source(detail::device_data_reference device_data_reference,
                                       table_device_view const& left_table,
                                       table_device_view const& right_table,
                                       cudf::size_type left_row_index,
                                       cudf::size_type right_row_index)
{
  switch (device_data_reference.reference_type) {
    case detail::device_data_reference_type::COLUMN: {
      if (device_data_reference.table_reference == table_reference::LEFT) {
        auto column = left_table.column(device_data_reference.data_index);
        return column.data<Element>()[left_row_index];
      } else if (device_data_reference.table_reference == table_reference::RIGHT) {
        auto column = right_table.column(device_data_reference.data_index);
        return column.data<Element>()[right_row_index];
      }
    }
    case detail::device_data_reference_type::LITERAL: {
      // TODO: Fetch and return literal.
      return static_cast<Element>(0);
    }
    case detail::device_data_reference_type::INTERMEDIATE: {
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
__device__ Element evaluate_expression(ast_expression expr,
                                       table_device_view table,
                                       cudf::size_type row_index)
{
  const Element lhs = resolve_data_source<Element>(expr.lhs, table, row_index);
  const Element rhs = resolve_data_source<Element>(expr.rhs, table, row_index);
  return binop_dispatcher(expr.op, do_binop<Element>{}, lhs, rhs);
}

template <typename Element>
__device__ Element evaluate_expression(binary_expression expr,
                                       table_device_view left_table,
                                       table_device_view right_table,
                                       cudf::size_type left_row_index,
                                       cudf::size_type right_row_index)
{
  const Element lhs = resolve_data_source<Element>(
    expr.lhs, left_table, right_table, left_row_index, right_row_index);
  const Element rhs = resolve_data_source<Element>(
    expr.rhs, left_table, right_table, left_row_index, right_row_index);
  return binop_dispatcher(expr.op, do_binop<Element>{}, lhs, rhs);
}

template <typename Element>
__device__ bool evaluate_expression(comparator_expression expr,
                                    table_device_view table,
                                    cudf::size_type row_index)
{
  const Element lhs = resolve_data_source<Element>(expr.lhs, table, row_index);
  const Element rhs = resolve_data_source<Element>(expr.rhs, table, row_index);
  return compareop_dispatcher(expr.op, do_compareop<Element>{}, lhs, rhs);
}

template <typename Element>
__device__ bool evaluate_expression(comparator_expression expr,
                                    table_device_view left_table,
                                    table_device_view right_table,
                                    cudf::size_type left_row_index,
                                    cudf::size_type right_row_index)
{
  const Element lhs = resolve_data_source<Element>(
    expr.lhs, left_table, right_table, left_row_index, right_row_index);
  const Element rhs = resolve_data_source<Element>(
    expr.rhs, left_table, right_table, left_row_index, right_row_index);
  return compareop_dispatcher(expr.op, do_compareop<Element>{}, lhs, rhs);
}

template <typename Element>
__global__ void compute_column_kernel(table_device_view table,
                                      binary_expression expr,
                                      mutable_column_device_view output)
{
  const cudf::size_type start_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const cudf::size_type stride    = blockDim.x * gridDim.x;
  const auto num_rows             = table.num_rows();

  for (cudf::size_type row_index = start_idx; row_index < num_rows; row_index += stride) {
    output.element<Element>(row_index) = evaluate_expression<Element>(expr, table, row_index);
  }
}

template <typename Element>
__global__ void compute_column_kernel(table_device_view table,
                                      comparator_expression expr,
                                      mutable_column_device_view output)
{
  const cudf::size_type start_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const cudf::size_type stride    = blockDim.x * gridDim.x;
  const auto num_rows             = table.num_rows();

  for (cudf::size_type row_index = start_idx; row_index < num_rows; row_index += stride) {
    output.element<bool>(row_index) = evaluate_expression<Element>(expr, table, row_index);
  }
}

template <typename Element>
std::unique_ptr<column> compute_column(
  table_view const& table,
  std::shared_ptr<expression> expr,
  cudaStream_t stream                 = 0,  // TODO use detail API
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  // Linearize the AST
  auto expr_linearizer = linearizer();
  expr->accept(expr_linearizer);
  auto expr_data_type = expr_linearizer.get_root_data_type();

  auto table_device = table_device_view::create(table, stream);
  // auto expr_data_type = cudf::data_type(cudf::type_to_id<Element>());
  auto table_num_rows = table.num_rows();
  auto output_column =
    make_fixed_width_column(expr_data_type, table_num_rows, mask_state::UNALLOCATED, stream, mr);
  auto block_size = 1024;  // TODO dynamically determine block size, use shared memory
  auto mutable_output_device =
    cudf::mutable_column_device_view::create(output_column->mutable_view(), stream);

  cudf::detail::grid_1d config(table_num_rows, block_size);
  compute_column_kernel<Element><<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
    *table_device, expr, *mutable_output_device);
  return output_column;
}

}  // namespace ast

}  // namespace cudf
