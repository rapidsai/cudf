/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cudf/jit/row_ir.hpp>

#include <algorithm>
#include <numeric>

namespace cudf {

namespace row_ir {

std::string instance_context::make_tmp_id()
{
  return std::format("{}{}", tmp_prefix_, num_tmp_vars_++);
}

get_input::get_input(uint32_t input) : id_(), input_(input), type_(type_id::EMPTY), nullable_() {}

std::string_view get_input::get_id() { return id_; }

data_type get_input::get_type() { return type_; }

bool get_input::get_nullable() { return nullable_; }

void get_input::instantiate(instance_context& ctx, instance_info const& info)
{
  id_               = ctx.make_tmp_id();
  auto const& input = info.inputs[input_];
  type_             = input.type;
  nullable_         = input.nullable;
}

std::string get_input::generate_code(instance_context& ctx,
                                     target_info const& info,
                                     instance_info const& instance)
{
  switch (info.id) {
    case target::CUDA: {
      return std::format("auto {} = {};", id_, instance.inputs[input_].id);
    }
    default: CUDF_FAIL("Unsupported target: " + std::to_string(static_cast<int>(info.id)));
  }
}

set_output::set_output(uint32_t output, std::unique_ptr<node> source)
  : id_(),
    output_(output),
    source_(std::move(source)),
    type_(type_id::EMPTY),
    nullable_(false),
    output_id_()
{
}

std::string_view set_output::get_id() { return id_; }

data_type set_output::get_type() { return type_; }

bool set_output::get_nullable() { return nullable_; }

void set_output::instantiate(instance_context& ctx, instance_info const& info)
{
  id_        = ctx.make_tmp_id();
  type_      = source_->get_type();
  nullable_  = source_->get_nullable();
  output_id_ = info.outputs[output_].id;
}

std::string set_output::generate_code(instance_context& ctx,
                                      target_info const& info,
                                      instance_info const& instance)
{
  switch (info.id) {
    case target::CUDA: {
      return std::format("{} = {};", output_id_, source_->get_id());
    }
    default: CUDF_FAIL("Unsupported target: " + std::to_string(static_cast<int>(info.id)));
  }
}

operation::operation(opcode op, std::vector<std::unique_ptr<node>> operands)
  : id_(), op_(op), operands_(std::move(operands)), type_(type_id::EMPTY), nullable_(false)
{
  CUDF_EXPECTS(operands.size() == cudf::ast::detail::ast_operator_arity(op),
               "Invalid number of arguments for operator.");
  CUDF_EXPECTS(operands.size() > 0, "Operator must have at least one operand");
}

std::string_view operation::get_id() { return id_; }

data_type operation::get_type() { return type_; }

bool operation::get_nullable() { return nullable_; }

void operation::instantiate(instance_context& ctx, instance_info const& info)
{
  for (auto& arg : operands_) {
    arg->instantiate(ctx, info);
  }

  id_ = ctx.make_tmp_id();
  std::vector<cudf::data_type> operand_types;
  auto nullable = std::any_of(
    operands_.begin(), operands_.end(), [](auto const& arg) { return arg->get_nullable(); });

  for (auto& arg : operands_) {
    operand_types.emplace_back(arg->get_type());
  }

  type_     = cudf::ast::detail::ast_operator_return_type(op_, operand_types);
  nullable_ = nullable;
}

std::string operation::generate_code(instance_context& ctx,
                                     target_info const& info,
                                     instance_info const& instance)
{
  std::string operands_code;

  for (auto& arg : operands_) {
    operands_code += arg->generate_code(ctx, info, instance) + "\n";
  }

  auto operation_code = [&]() {
    switch (info.id) {
      case target::CUDA: {
        auto first_operand = operands_[0]->get_id();
        auto operands_str =
          (operands_.size() == 1)
            ? std::format("{}", first_operand)
            : std::format("{}, {}",
                          first_operand,
                          std::accumulate(operands_.begin() + 1,
                                          operands_.end(),
                                          std::string{},
                                          [](std::string_view a, auto& node) {
                                            return std::format("{}, {}", a, node->get_id());
                                          }));
        auto cuda =
          std::format("auto {} = cudf::ast::operator_functor<cudf::ast::ast_operator::{}, {}>({});",
                      id_,
                      ast::detail::ast_operator_string(op_),
                      nullable_,
                      operands_str);
        return cuda;
      }
      default: CUDF_FAIL("Unsupported target: " + std::to_string(static_cast<int>(info.id)));
    }
  }();

  return operands_code + operation_code;
}

std::span<ast_input_ref const> ast_converter::get_input_refs() const { return input_refs_; }

uint32_t ast_converter::add_ast_input_ref(ast_input_ref in)
{
  auto id = static_cast<uint32_t>(input_refs_.size());
  input_refs_.push_back(std::move(in));
  return id;
}

std::unique_ptr<row_ir::node> ast_converter::visit(cudf::ast::literal const& expr)
{
  // fix scalar views
  //   add_input(scalar_input{expr.get_value(), expr.get_value()});
  //   return std::make_unique<row_ir::get_input>(expr.get_value(), expr.is_nullable());
}

std::unique_ptr<row_ir::node> ast_converter::visit(cudf::ast::column_reference const& expr)
{
  auto index =
    add_ast_input_ref(ast_column_input_ref{static_cast<uint32_t>(expr.get_table_source()),
                                           static_cast<uint32_t>(expr.get_column_index())});
  return std::make_unique<row_ir::get_input>(index);
}

std::unique_ptr<row_ir::node> ast_converter::visit(cudf::ast::operation const& expr)
{
  std::vector<std::unique_ptr<row_ir::node>> operands;
  for (auto const& operand : expr.get_operands()) {
    operands.push_back(operand.get().accept(*this));
  }
  return std::make_unique<row_ir::operation>(expr.get_operator(), std::move(operands));
}

std::unique_ptr<row_ir::node> ast_converter::visit(cudf::ast::column_name_reference const& expr)
{
  auto index = add_ast_input_ref(ast_named_column_input_ref{expr.get_column_name()});
  return std::make_unique<row_ir::get_input>(index);
}

void ast_converter::add_input_var(ast_column_input_ref const& in,
                                  bool nullable,
                                  named_table_view const& left_table,
                                  named_table_view const& right_table)
{
  // TODO: mangle column name to make debugging easier
  auto id   = std::format("in_{}", input_vars_.size());
  auto type = left_table.table.column(in.column).type();
  input_vars_.emplace_back(std::move(id), type, nullable);
}

void ast_converter::add_input_var(ast_named_column_input_ref const& in,
                                  bool nullable,
                                  named_table_view const& left_table,
                                  named_table_view const& right_table)
{
  auto id   = std::format("in_{}", input_vars_.size());
  auto type = left_table.table.column(left_table.column_names.at(in.name)).type();
  input_vars_.emplace_back(std::move(id), type, nullable);
}

void ast_converter::add_input_var(ast_scalar_input_ref const& in,
                                  bool nullable,
                                  named_table_view const& left_table,
                                  named_table_view const& right_table)
{
  auto id   = std::format("in_{}", input_vars_.size());
  auto type = in.scalar.get().type();
  input_vars_.emplace_back(std::move(id), type, nullable);
}

void ast_converter::add_output_var()
{
  auto id = std::format("out_{}", output_vars_.size());
  output_vars_.emplace_back(std::move(id));
}

void ast_converter::exec(target target_id,
                         cudf::ast::expression const& expr,
                         bool null_aware,
                         named_table_view const& left_table,
                         named_table_view const& right_table)
{
  input_refs_.clear();
  input_vars_.clear();
  output_vars_.clear();
  output_ir_.reset();

  // [ ] make sure only called once
  auto output_expr_ir = expr.accept(*this);
  output_ir_          = std::make_unique<row_ir::set_output>(0, std::move(output_expr_ir));

  // [ ] fix null-aware handling

  // resolve the flattened input references into IR input variables
  for (auto const& input : input_refs_) {
    dispatch_input_ref(
      input, [this](auto&... arg) { add_input_var(arg...); }, null_aware, left_table, right_table);
  }

  // add 1 deduced output variable
  add_output_var();

  instance_context instance_ctx;
  instance_info instance{input_vars_, output_vars_};

  // instantiate the IR nodes
  output_ir_->instantiate(instance_ctx, instance);

  target_info target{target_id};

  auto cuda = output_ir_->generate_code(instance_ctx, target, instance);

  // [ ] collect the inputs as an array of column_views

  // [ ] wrap the generated code in a CUDA UDF function

  switch (target.id) {
    case target::CUDA: {
      // [ ] generate the CUDA UDF function
      // [ ] return the generated code
      break;
    }
    default: CUDF_FAIL("Unsupported target: " + std::to_string(static_cast<int>(target.id)));
  }
}

}  // namespace row_ir
}  // namespace cudf
