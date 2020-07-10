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

#include <thrust/device_vector.h>
#include <algorithm>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <functional>
#include <iterator>
#include <type_traits>
#include "cudf/column/column_device_view.cuh"
#include "cudf/column/column_factories.hpp"
#include "cudf/table/table_device_view.cuh"
#include "cudf/types.hpp"
#include "cudf/utilities/traits.hpp"
#include "operators.cuh"
#include "thrust/detail/raw_pointer_cast.h"

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
  cudf::data_type data_type;                  // Type of data
  cudf::size_type
    data_index;  // The column index of a table, index of a literal, or index of an intermediate
  table_reference table_reference = table_reference::LEFT;
};

class intermediate_counter {
 public:
  intermediate_counter() : used_values(), max_used(0) {}
  cudf::size_type take()
  {
    auto first_missing =
      this->used_values.size() ? this->find_first_missing(0, this->used_values.size() - 1) : 0;
    this->used_values.insert(this->used_values.cbegin() + first_missing, first_missing);
    this->max_used = std::max(max_used, first_missing + 1);
    return first_missing;
  }
  void give(cudf::size_type value)
  {
    auto found = std::find(this->used_values.cbegin(), this->used_values.cend(), value);
    if (found != this->used_values.cend()) { this->used_values.erase(found); }
  }
  cudf::size_type get_max_used() const { return this->max_used; }

 private:
  cudf::size_type find_first_missing(cudf::size_type start, cudf::size_type end)
  {
    // Given a sorted container, find the smallest value not already in the container
    if (start > end) return end + 1;

    // If the value at an index is not equal to its index, it must be the missing value
    if (start != used_values.at(start)) return start;

    auto mid = (start + end) / 2;

    // Use binary search and check the left half or right half
    // We can assume the missing value must be in the right half
    if (used_values.at(mid) == mid) return this->find_first_missing(mid + 1, end);
    // The missing value must be in the left half
    return this->find_first_missing(start, mid);
  }

  std::vector<cudf::size_type> used_values;
  cudf::size_type max_used;
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
  column_reference(cudf::size_type column_index,
                   table_reference table_source = table_reference::LEFT)
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
  linearizer(cudf::table_view table) : table(table), node_counter(0), intermediate_counter() {}

  cudf::size_type visit(literal const& expr) override
  {
    // Resolve node type
    auto data_type = expr.get_data_type();
    // TODO: Use scalar device view (?)
    // Push literal
    auto literal_index = cudf::size_type(this->literals.size());
    this->literals.push_back(expr.get_value());
    // Push data reference
    auto source = detail::device_data_reference{
      detail::device_data_reference_type::LITERAL, data_type, literal_index};
    this->data_references.push_back(source);
    // Increment counter
    auto index = this->node_counter;
    this->node_counter++;
    return index;
  }

  cudf::size_type visit(column_reference const& expr) override
  {
    // Resolve node type
    auto data_type = expr.get_data_type(this->table);
    // Push data reference
    auto source = detail::device_data_reference{detail::device_data_reference_type::COLUMN,
                                                data_type,
                                                expr.get_column_index(),
                                                expr.get_table_source()};
    this->data_references.push_back(source);
    // Increment counter
    auto index = this->node_counter;
    this->node_counter++;
    return index;
  }

  cudf::size_type visit(operator_expression const& expr) override
  {
    const auto op                       = expr.get_operator();
    auto operand_data_reference_indices = this->visit_operands(expr.get_operands());
    // Resolve operand types
    auto operand_types = std::vector<cudf::data_type>();
    std::transform(operand_data_reference_indices.cbegin(),
                   operand_data_reference_indices.cend(),
                   std::back_inserter(operand_types),
                   [this](auto const& data_reference_index) -> cudf::data_type {
                     return this->get_data_references()[data_reference_index].data_type;
                   });
    // Validate types of operand data references match
    if (std::adjacent_find(operand_types.cbegin(), operand_types.cend(), std::not_equal_to<>()) !=
        operand_types.cend()) {
      CUDF_FAIL("An AST operator expression was provided non-matching operand types.");
    }
    // Give back intermediate storage locations that are consumed by this operation
    std::for_each(
      operand_data_reference_indices.cbegin(),
      operand_data_reference_indices.cend(),
      [this](auto const& data_reference_index) {
        auto operand_source = this->get_data_references()[data_reference_index];
        if (operand_source.reference_type == detail::device_data_reference_type::INTERMEDIATE) {
          auto intermediate_index = operand_source.data_index;
          this->intermediate_counter.give(intermediate_index);
        }
      });
    // Resolve node type
    auto data_type = operand_types.at(0);
    /* TODO: Need to fix. Can't support comparators yet.
    auto data_type = [&] {
      const ast_operator oper = op;
      if (cudf::ast::is_arithmetic_operator<oper>()) {
        return operand_types.at(0);
      } else {
        return cudf::data_type(cudf::type_id::EMPTY);
      }
    }();
    */
    // Push operator
    this->operators.push_back(op);
    // Push data reference
    auto source = detail::device_data_reference{detail::device_data_reference_type::INTERMEDIATE,
                                                data_type,
                                                this->intermediate_counter.take()};
    this->data_references.push_back(source);
    // Increment counter
    auto index = this->node_counter;
    this->node_counter++;
    // Insert source indices from all operands (sources) and operator (destination)
    this->operator_source_indices.insert(this->operator_source_indices.end(),
                                         operand_data_reference_indices.cbegin(),
                                         operand_data_reference_indices.cend());
    this->operator_source_indices.push_back(index);
    return index;
  }

  cudf::data_type get_root_data_type() const
  {
    if (this->get_data_references().size() == 0) {
      return cudf::data_type(cudf::type_id::EMPTY);
    } else {
      return this->get_data_references().back().data_type;
    }
  }
  cudf::size_type get_node_counter() const { return this->node_counter; }
  cudf::size_type get_intermediate_count() const
  {
    return this->intermediate_counter.get_max_used();
  }
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
  // State information about the "linearized" GPU execution plan
  cudf::table_view table;
  cudf::size_type node_counter;
  detail::intermediate_counter intermediate_counter;
  std::vector<detail::device_data_reference> data_references;
  std::vector<ast_operator> operators;
  std::vector<cudf::size_type> operator_source_indices;
  std::vector<std::reference_wrapper<const scalar>> literals;
};

template <typename Element>
__device__ Element resolve_input_data_reference(detail::device_data_reference device_data_reference,
                                                table_device_view const& table,
                                                std::int64_t* thread_intermediate_storage,
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
      return *reinterpret_cast<Element*>(
        &thread_intermediate_storage[device_data_reference.data_index]);
    }
    default: {
      // TODO: Error
      return static_cast<Element>(0);
    }
  }
}

template <typename Element>
__device__ Element* resolve_output_data_reference(
  detail::device_data_reference device_data_reference,
  table_device_view const& table,
  std::int64_t* thread_intermediate_storage,
  cudf::size_type row_index)
{
  switch (device_data_reference.reference_type) {
    case detail::device_data_reference_type::COLUMN: {
      // TODO: Support output columns?
      return nullptr;
    }
    case detail::device_data_reference_type::INTERMEDIATE: {
      return reinterpret_cast<Element*>(
        &thread_intermediate_storage[device_data_reference.data_index]);
    }
    default: {
      // TODO: Error
      return nullptr;
    }
  }
}

/*
// TODO: Resolve data sources for two tables
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
*/

struct typed_binop_dispatch {
  template <typename Element, std::enable_if_t<cudf::is_numeric<Element>()>* = nullptr>
  __device__ void operator()(ast_operator op,
                             table_device_view const& table,
                             std::int64_t* thread_intermediate_storage,
                             cudf::size_type row_index,
                             detail::device_data_reference lhs,
                             detail::device_data_reference rhs,
                             detail::device_data_reference output)
  {
    auto typed_lhs =
      resolve_input_data_reference<Element>(lhs, table, thread_intermediate_storage, row_index);
    auto typed_rhs =
      resolve_input_data_reference<Element>(rhs, table, thread_intermediate_storage, row_index);
    auto typed_output =
      resolve_output_data_reference<Element>(output, table, thread_intermediate_storage, row_index);
    *typed_output = ast_operator_dispatcher_numeric(op, do_binop<Element>{}, typed_lhs, typed_rhs);
    /*
    if (row_index == 0) {
      printf("lhs index %i = %f, rhs index %i = %f, output index %i = %f\n",
             lhs.data_index,
             float(typed_lhs),
             rhs.data_index,
             float(typed_rhs),
             output.data_index,
             float(*typed_output));
    }
    */
  }

  template <typename Element, std::enable_if_t<!cudf::is_numeric<Element>()>* = nullptr>
  __device__ void operator()(ast_operator op,
                             table_device_view const& table,
                             std::int64_t* thread_intermediate_storage,
                             cudf::size_type row_index,
                             detail::device_data_reference lhs,
                             detail::device_data_reference rhs,
                             detail::device_data_reference output)
  {
    // TODO: How else to make this compile? Need a template to match unsupported types, or prevent
    // the compiler from attempting to compile unsupported types here.
  }
};

__device__ void operate(ast_operator op,
                        table_device_view const& table,
                        std::int64_t* thread_intermediate_storage,
                        cudf::size_type row_index,
                        detail::device_data_reference lhs,
                        detail::device_data_reference rhs,
                        detail::device_data_reference output)
{
  type_dispatcher(lhs.data_type,
                  typed_binop_dispatch{},
                  op,
                  table,
                  thread_intermediate_storage,
                  row_index,
                  lhs,
                  rhs,
                  output);
}

struct output_copy_functor {
  template <typename Element, std::enable_if_t<cudf::is_numeric<Element>()>* = nullptr>
  __device__ void operator()(mutable_column_device_view output_column,
                             table_device_view const& table,
                             std::int64_t* thread_intermediate_storage,
                             cudf::size_type row_index,
                             detail::device_data_reference expression_output)
  {
    output_column.element<Element>(row_index) = resolve_input_data_reference<Element>(
      expression_output, table, thread_intermediate_storage, row_index);
  };
  template <typename Element, std::enable_if_t<!cudf::is_numeric<Element>()>* = nullptr>
  __device__ void operator()(mutable_column_device_view output_column,
                             table_device_view const& table,
                             std::int64_t* thread_intermediate_storage,
                             cudf::size_type row_index,
                             detail::device_data_reference expression_output){
    // TODO: How else to make this compile? Need a template to match unsupported types, or prevent
    // the compiler from attempting to compile unsupported types here.
  };
};

__device__ void evaluate_row_expression(table_device_view const& table,
                                        detail::device_data_reference* data_references,
                                        // scalar* literals,
                                        ast_operator* operators,
                                        cudf::size_type* operator_source_indices,
                                        cudf::size_type num_operators,
                                        cudf::size_type row_index,
                                        std::int64_t* thread_intermediate_storage,
                                        mutable_column_device_view output)
{
  auto operator_source_index = cudf::size_type(0);
  for (cudf::size_type operator_index(0); operator_index < num_operators; operator_index++) {
    // Execute operator
    auto const& op = operators[operator_index];
    if (is_binary_operator(op)) {
      auto lhs_data_ref    = data_references[operator_source_indices[operator_source_index]];
      auto rhs_data_ref    = data_references[operator_source_indices[operator_source_index + 1]];
      auto output_data_ref = data_references[operator_source_indices[operator_source_index + 2]];
      /*
      if (row_index == 0) {
        printf("Operator id %i is ", operator_index);
        switch (op) {
          case ast_operator::ADD: printf("ADDing "); break;
          case ast_operator::SUB: printf("SUBtracting "); break;
          case ast_operator::MUL: printf("MULtiplying "); break;
          default: break;
        }
        printf("lhs index %i and ", operator_source_indices[operator_source_index]);
        printf("rhs index %i to ", operator_source_indices[operator_source_index + 1]);
        printf("output index %i.\n", operator_source_indices[operator_source_index + 2]);
      }
      */

      operator_source_index += 3;
      operate(op,
              table,
              thread_intermediate_storage,
              row_index,
              lhs_data_ref,
              rhs_data_ref,
              output_data_ref);
    } else {
      // TODO: Support ternary operator
      // Assume operator is unary
      // auto input_data_ref  = data_references[operator_source_indices[operator_source_index]];
      // auto output_data_ref = data_references[operator_source_indices[operator_source_index + 1]];
      operator_source_index += 2;
      // TODO: Unary operations
    }
  }
  // Copy from last data reference to output column
  auto expression_output = data_references[operator_source_indices[operator_source_index - 1]];
  type_dispatcher(expression_output.data_type,
                  output_copy_functor{},
                  output,
                  table,
                  thread_intermediate_storage,
                  row_index,
                  expression_output);
}

__global__ void compute_column_kernel(table_device_view table,
                                      detail::device_data_reference* data_references,
                                      // scalar* literals,
                                      ast_operator* operators,
                                      cudf::size_type* operator_source_indices,
                                      cudf::size_type num_operators,
                                      cudf::size_type num_intermediates,
                                      mutable_column_device_view output)
{
  extern __shared__ std::int64_t intermediate_storage[];
  auto thread_intermediate_storage = &intermediate_storage[threadIdx.x * num_intermediates];
  const cudf::size_type start_idx  = threadIdx.x + blockIdx.x * blockDim.x;
  const cudf::size_type stride     = blockDim.x * gridDim.x;
  const auto num_rows              = table.num_rows();

  for (cudf::size_type row_index = start_idx; row_index < num_rows; row_index += stride) {
    evaluate_row_expression(table,
                            data_references,
                            // literals,
                            operators,
                            operator_source_indices,
                            num_operators,
                            row_index,
                            thread_intermediate_storage,
                            output);
    // output.element<bool>(row_index) = evaluate_expression<Element>(expr, table, row_index);
  }
}

std::unique_ptr<column> compute_column(
  table_view const& table,
  std::reference_wrapper<const expression> expr,
  cudaStream_t stream                 = 0,  // TODO use detail API
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  // Linearize the AST
  auto expr_linearizer = linearizer(table);
  expr.get().accept(expr_linearizer);
  auto data_references         = expr_linearizer.get_data_references();
  auto literals                = expr_linearizer.get_literals();
  auto operators               = expr_linearizer.get_operators();
  auto num_operators           = cudf::size_type(operators.size());
  auto operator_source_indices = expr_linearizer.get_operator_source_indices();
  auto expr_data_type          = expr_linearizer.get_root_data_type();

  // Create device data
  auto device_data_references =
    thrust::device_vector<detail::device_data_reference>(data_references);
  // TODO: Literals
  // auto device_literals = thrust::device_vector<const scalar>();
  auto device_operators = thrust::device_vector<cudf::ast::ast_operator>(operators);
  auto device_operator_source_indices =
    thrust::device_vector<cudf::size_type>(operator_source_indices);

  // Output linearizer info
  /*
  std::cout << "LINEARIZER INFO:" << std::endl;
  std::cout << "Node counter: " << expr_linearizer.get_node_counter() << std::endl;
  std::cout << "Number of data references: " << data_references.size() << std::endl;
  std::cout << "Data references: ";
  for (auto dr : data_references) {
    switch (dr.reference_type) {
      case detail::device_data_reference_type::COLUMN: std::cout << "C"; break;
      case detail::device_data_reference_type::LITERAL: std::cout << "L"; break;
      case detail::device_data_reference_type::INTERMEDIATE: std::cout << "I";
    }
    std::cout << dr.data_index << ", ";
  }
  std::cout << std::endl;
  std::cout << "Number of operators: " << num_operators << std::endl;
  std::cout << "Number of operator source indices: " << operator_source_indices.size() << std::endl;
  std::cout << "Number of literals: " << literals.size() << std::endl;
  std::cout << "Operator source indices: ";
  for (auto v : operator_source_indices) { std::cout << v << ", "; }
  std::cout << std::endl;
  */

  // Create table device view
  auto table_device   = table_device_view::create(table, stream);
  auto table_num_rows = table.num_rows();

  // Prepare output column
  auto output_column =
    make_fixed_width_column(expr_data_type, table_num_rows, mask_state::UNALLOCATED, stream, mr);
  auto mutable_output_device =
    cudf::mutable_column_device_view::create(output_column->mutable_view(), stream);

  // Configure kernel parameters
  auto block_size = 1024;  // TODO: Dynamically determine block size
  cudf::detail::grid_1d config(table_num_rows, block_size);
  auto num_intermediates = expr_linearizer.get_intermediate_count();
  auto shmem_size_per_block =
    sizeof(std::int64_t) * num_intermediates * config.num_threads_per_block;
  // std::cout << "Requesting " << shmem_size_per_block << " bytes of shared memory." << std::endl;

  // Execute the kernel
  compute_column_kernel<<<config.num_blocks,
                          config.num_threads_per_block,
                          shmem_size_per_block,
                          stream>>>(*table_device,
                                    thrust::raw_pointer_cast(device_data_references.data()),
                                    // device_literals,
                                    thrust::raw_pointer_cast(device_operators.data()),
                                    thrust::raw_pointer_cast(device_operator_source_indices.data()),
                                    num_operators,
                                    num_intermediates,
                                    *mutable_output_device);
  return output_column;
}

}  // namespace ast

}  // namespace cudf
