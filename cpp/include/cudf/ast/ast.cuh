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
class column_reference;
class operator_expression;

// Visitor pattern. The visitor tracks a node index as it traverses the tree.
class expression {
 public:
  virtual cudf::size_type accept(abstract_visitor& visitor) const = 0;
};

class abstract_visitor {
 public:
  virtual cudf::size_type visit(literal const& expr)             = 0;
  virtual cudf::size_type visit(column_reference const& expr)    = 0;
  virtual cudf::size_type visit(operator_expression const& expr) = 0;
};

class literal : public expression {
 public:
  literal(std::reference_wrapper<const cudf::scalar> value) : value(std::cref(value)) {}
  std::reference_wrapper<const cudf::scalar> get_value() const { return this->value; }
  cudf::size_type accept(abstract_visitor& visitor) const override { return visitor.visit(*this); }
  cudf::data_type get_data_type() const { return this->get_value().get().type(); }

 private:
  std::reference_wrapper<const cudf::scalar> value;
};

class column_reference : public expression {
 public:
  column_reference(cudf::size_type column_index, table_reference table_source)
    : column_index(column_index), table_source(table_source)
  {
  }
  cudf::size_type get_column_index() const { return this->column_index; }
  table_reference get_table_source() const { return this->table_source; }
  cudf::data_type get_data_type(const table_view& table) const
  {
    return table.column(this->get_column_index()).type();
  }
  cudf::data_type get_data_type(const table_view& left_table, const table_view& right_table) const
  {
    const auto table = [&] {
      if (this->get_table_source() == table_reference::LEFT) {
        return left_table;
      } else if (this->get_table_source() == table_reference::RIGHT) {
        return right_table;
      } else {
        CUDF_FAIL("Column reference data type cannot be determined from unknown table.");
      }
    }();
    return table.column(this->get_column_index()).type();
  }
  // TODO: Fetch value for a given table and row

  cudf::size_type accept(abstract_visitor& visitor) const override { return visitor.visit(*this); }

 private:
  cudf::size_type column_index;
  table_reference table_source;
};

class operator_expression : public expression {
 public:
  ast_operator get_operator() const { return this->op; }
  virtual std::vector<std::reference_wrapper<const expression>> get_operands() const = 0;

 protected:
  operator_expression(ast_operator op) : op(op) {}
  const ast_operator op;
};

class binary_expression : public operator_expression {
 public:
  binary_expression(ast_operator op,
                    std::reference_wrapper<const expression> left,
                    std::reference_wrapper<const expression> right)
    : operator_expression(op), left(left), right(right)
  {
  }
  std::reference_wrapper<const expression> get_left() const { return this->left; }
  std::reference_wrapper<const expression> get_right() const { return this->right; }
  std::vector<std::reference_wrapper<const expression>> get_operands() const override
  {
    return std::vector<std::reference_wrapper<const expression>>{this->get_left(),
                                                                 this->get_right()};
  }
  cudf::size_type accept(abstract_visitor& visitor) const override { return visitor.visit(*this); }

 private:
  std::reference_wrapper<const expression> left;
  std::reference_wrapper<const expression> right;
};

class linearizer : public abstract_visitor {
 public:
  linearizer(cudf::table_view tables) : node_counter(0) {}

  cudf::size_type visit(literal const& expr) override
  {
    std::cout << "visiting literal value" << std::endl;

    // Resolve node type
    auto data_type = expr.get_data_type();
    // Push node type
    // TODO: Use scalar device view
    this->literals.push_back(expr.get_value());
    // TODO: Push literal
    // Push data reference
    auto source = detail::device_data_reference{
      detail::device_data_reference_type::LITERAL,
      0  // TODO: Use correct index
    };
    this->add_source(data_type, source);
    auto index = this->node_counter;
    this->node_counter++;
    return index;
  }

  cudf::size_type visit(column_reference const& expr) override
  {
    std::cout << "visiting column reference" << std::endl;
    // TODO: Resolve node type
    auto data_type = cudf::data_type(cudf::type_id::EMPTY);
    // auto data_type = expr.get_data_type();
    // TODO: Push node type
    // Push data reference
    auto source = detail::device_data_reference{
      detail::device_data_reference_type::COLUMN, expr.get_column_index(), expr.get_table_source()};
    this->add_source(data_type, source);
    auto index = this->node_counter;
    this->node_counter++;
    return index;
  }

  cudf::size_type visit(operator_expression const& expr) override
  {
    std::cout << "visiting operator_expression" << std::endl;
    auto op                             = expr.get_operator();
    auto operand_data_reference_indices = this->visit_operands(expr.get_operands());
    std::cout << "visited operands" << std::endl;
    // TODO: Resolve types
    // TODO: Validate types of operand data references match
    auto input_type = cudf::data_type(cudf::type_id::EMPTY);
    // TODO: Resolve type for this operator expression
    auto data_type = cudf::data_type(cudf::type_id::EMPTY);
    // Push operator
    this->operators.push_back(op);
    // Push data reference
    auto source = detail::device_data_reference{
      detail::device_data_reference_type::INTERMEDIATE,
      0  // TODO: Use correct index -- potentially reuse indices
    };
    this->add_source(data_type, source);
    // TODO: Push operand_data_reference_indices to global reference list
    auto index = this->node_counter;
    this->node_counter++;
    // Insert source indices from all operands (sources) and operator (destination)
    this->operator_source_indices.insert(this->operator_source_indices.end(),
                                         operand_data_reference_indices.cbegin(),
                                         operand_data_reference_indices.cend());
    this->operator_source_indices.push_back(index);
    std::cout << "visited operator_expression" << std::endl;
    return index;
  }

  cudf::data_type get_root_data_type() const
  {
    if (this->node_data_types.size() == 0) {
      return cudf::data_type(cudf::type_id::EMPTY);
    } else {
      return this->node_data_types.back();
    }
  }
  cudf::size_type get_node_counter() const { return this->node_counter; }
  std::vector<cudf::data_type> get_node_data_types() const { return this->node_data_types; }
  std::vector<detail::device_data_reference> get_data_references() const
  {
    return this->data_references;
  }
  std::vector<ast_operator> get_operators() const { return this->operators; }
  std::vector<cudf::size_type> get_operator_source_indices() const
  {
    return this->operator_source_indices;
  }
  std::vector<std::reference_wrapper<const scalar>> get_literals() const { return this->literals; }

 private:
  std::vector<cudf::size_type> visit_operands(
    std::vector<std::reference_wrapper<const expression>> operands)
  {
    auto operand_data_reference_indices = std::vector<cudf::size_type>();
    for (auto& operand : operands) {
      auto operand_data_reference_index = operand.get().accept(*this);
      operand_data_reference_indices.push_back(operand_data_reference_index);
    }
    return operand_data_reference_indices;
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

/*
template <typename Element>
__device__ Element evaluate_expression(binary_expression expr,
                                       table_device_view table,
                                       cudf::size_type row_index)
{
  const Element lhs = resolve_data_source<Element>(expr.get_left(), table, row_index);
  const Element rhs = resolve_data_source<Element>(expr.get_right(), table, row_index);
  return binop_dispatcher(expr.get_operator(), do_binop<Element>{}, lhs, rhs);
}

template <typename Element>
__device__ Element evaluate_expression(binary_expression expr,
                                       table_device_view left_table,
                                       table_device_view right_table,
                                       cudf::size_type left_row_index,
                                       cudf::size_type right_row_index)
{
  const Element lhs = resolve_data_source<Element>(
    expr.get_left(), left_table, right_table, left_row_index, right_row_index);
  const Element rhs = resolve_data_source<Element>(
    expr.get_right(), left_table, right_table, left_row_index, right_row_index);
  return binop_dispatcher(expr.get_operator(), do_binop<Element>{}, lhs, rhs);
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
*/

std::unique_ptr<column> compute_column(
  table_view const& table,
  std::reference_wrapper<const expression> expr,
  cudaStream_t stream                 = 0,  // TODO use detail API
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  // Linearize the AST
  auto expr_linearizer = linearizer(table);
  expr.get().accept(expr_linearizer);
  auto expr_data_type = expr_linearizer.get_root_data_type();
  std::cout << "LINEARIZER INFO:" << std::endl;
  std::cout << "Node counter: " << expr_linearizer.get_node_counter() << std::endl;
  std::cout << "Number of node data types: " << expr_linearizer.get_node_data_types().size()
            << std::endl;
  std::cout << "Number of data references: " << expr_linearizer.get_data_references().size()
            << std::endl;
  std::cout << "Number of operators: " << expr_linearizer.get_operators().size() << std::endl;
  std::cout << "Number of operator source indices: "
            << expr_linearizer.get_operator_source_indices().size() << std::endl;
  std::cout << "Number of literals: " << expr_linearizer.get_literals().size() << std::endl;
  std::cout << "Operator source indices: ";
  auto operator_source_indices = expr_linearizer.get_operator_source_indices();
  for (auto v : operator_source_indices) { std::cout << v << ", "; }
  std::cout << std::endl;

  auto table_device = table_device_view::create(table, stream);
  // auto expr_data_type = cudf::data_type(cudf::type_to_id<Element>());
  auto table_num_rows = table.num_rows();
  // TODO: Remove this assignment below, temporary hack
  expr_data_type = cudf::data_type(cudf::type_id::INT32);
  auto output_column =
    make_fixed_width_column(expr_data_type, table_num_rows, mask_state::UNALLOCATED, stream, mr);
  std::cout << "Created output." << std::endl;
  /*
  auto block_size = 1024;  // TODO dynamically determine block size, use shared memory
  auto mutable_output_device =
    cudf::mutable_column_device_view::create(output_column->mutable_view(), stream);

  cudf::detail::grid_1d config(table_num_rows, block_size);
  compute_column_kernel<Element><<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
    *table_device, expr, *mutable_output_device);
  */
  return output_column;
}

}  // namespace ast

}  // namespace cudf
