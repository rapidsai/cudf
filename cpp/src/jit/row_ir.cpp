/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "jit/row_ir.hpp"

#include <cudf/column/column_factories.hpp>

#include <algorithm>
#include <numeric>

namespace cudf {

namespace row_ir {

std::string cuda_type(type_info type)
{
  auto non_null_type_str = type_to_name(type.type);
  auto type_str =
    type.nullable ? std::format("cuda::std::optional<{}>", non_null_type_str) : non_null_type_str;
  return type_str;
}

std::string instance_context::make_tmp_id()
{
  return std::format("{}{}", tmp_prefix_, num_tmp_vars_++);
}

void instance_context::reset(){
  num_tmp_vars_ = 0;
}

get_input::get_input(int32_t input) : id_(), input_(input), type_() {}

std::string_view get_input::get_id() { return id_; }

type_info get_input::get_type() { return type_; }

void get_input::instantiate(instance_context& ctx, instance_info const& info)
{
  id_               = ctx.make_tmp_id();
  auto const& input = info.inputs[input_];
  type_             = input.type;
}

std::string get_input::generate_code(instance_context& ctx,
                                     target_info const& info,
                                     instance_info const& instance)
{
  switch (info.id) {
    case target::CUDA: {
      return std::format("{} {} = {};", cuda_type(type_), id_, instance.inputs[input_].id);
    }
    default: CUDF_FAIL("Unsupported target: " + std::to_string(static_cast<int>(info.id)));
  }
}

set_output::set_output(int32_t output, std::unique_ptr<node> source)
  : id_(), output_(output), source_(std::move(source)), type_(), output_id_()
{
}

std::string_view set_output::get_id() { return id_; }

type_info set_output::get_type() { return type_; }

node& set_output::get_source() { return *source_; }

void set_output::instantiate(instance_context& ctx, instance_info const& info)
{
  source_->instantiate(ctx, info);
  id_        = ctx.make_tmp_id();
  type_      = source_->get_type();
  output_id_ = info.outputs[output_].id;
}

std::string set_output::generate_code(instance_context& ctx,
                                      target_info const& info,
                                      instance_info const& instance)
{
  switch (info.id) {
    case target::CUDA: {
      auto source_code = source_->generate_code(ctx, info, instance);
      return std::format(
        "{}\n"
        "{} {} = {};\n"
        "*{} = {};",
        source_code,
        cuda_type(this->get_type()),
        id_,
        source_->get_id(),
        output_id_,
        id_);
    }
    default: CUDF_FAIL("Unsupported target: " + std::to_string(static_cast<int>(info.id)));
  }
}

operation::operation(opcode op, std::vector<std::unique_ptr<node>> operands)
  : id_(), op_(op), operands_(std::move(operands)), type_()
{
  CUDF_EXPECTS(static_cast<size_type>(operands_.size()) == ast::detail::ast_operator_arity(op),
               "Invalid number of arguments for operator.");
  CUDF_EXPECTS(operands_.size() > 0, "Operator must have at least one operand");
}

std::string_view operation::get_id() { return id_; }

type_info operation::get_type() { return type_; }

opcode operation::get_opcode() const { return op_; }

std::span<std::unique_ptr<node> const> operation::get_operands() const { return operands_; }

void operation::instantiate(instance_context& ctx, instance_info const& info)
{
  for (auto& arg : operands_) {
    arg->instantiate(ctx, info);
  }

  id_ = ctx.make_tmp_id();
  std::vector<data_type> operand_types;
  auto nullable = std::any_of(
    operands_.begin(), operands_.end(), [](auto const& arg) { return arg->get_type().nullable; });

  for (auto& arg : operands_) {
    operand_types.emplace_back(arg->get_type().type);
  }

  type_ = type_info{ast::detail::ast_operator_return_type(op_, operand_types), nullable};
}

std::string operation::generate_code(instance_context& ctx,
                                     target_info const& info,
                                     instance_info const& instance)
{
  std::string operands_code;

  for (auto& arg : operands_) {
    operands_code =
      std::format("{}{}{}", operands_code, arg->generate_code(ctx, info, instance), "\n");
  }

  auto operation_code = [&]() {
    switch (info.id) {
      case target::CUDA: {
        auto first_operand = operands_[0]->get_id();
        auto operands_str  = (operands_.size() == 1)
                               ? std::string{first_operand}
                               : std::accumulate(operands_.begin() + 1,
                                                operands_.end(),
                                                std::string{first_operand},
                                                [](auto const& a, auto& node) {
                                                  return std::format("{}, {}", a, node->get_id());
                                                });

        auto cuda =
          std::format("{} {} = cudf::ast::operator_functor<cudf::ast::ast_operator::{}, {}>({});",
                      cuda_type(type_),
                      id_,
                      ast::detail::ast_operator_string(op_),
                      type_.nullable,
                      operands_str);
        return cuda;
      }
      default: CUDF_FAIL("Unsupported target: " + std::to_string(static_cast<int>(info.id)));
    }
  }();

  return operands_code + operation_code;
}

std::span<ast_input_spec const> ast_converter::get_input_specs() const { return input_specs_; }

int32_t ast_converter::add_ast_input(ast_input_spec in)
{
  auto id = static_cast<int32_t>(input_specs_.size());
  input_specs_.push_back(std::move(in));
  return id;
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::literal const& expr)
{
  auto index = add_ast_input(ast_scalar_input_spec{
    expr.get_scalar(), expr.get_value(), make_column_from_scalar(expr.get_scalar(), 1)});
  return std::make_unique<row_ir::get_input>(index);
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::column_reference const& expr)
{
  CUDF_EXPECTS(expr.get_table_source() == ast::table_reference::LEFT,
               "IR input column reference must be LEFT.");
  auto index =
    add_ast_input(ast_column_input_spec{expr.get_table_source(), expr.get_column_index()});
  return std::make_unique<row_ir::get_input>(index);
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::operation const& expr)
{
  std::vector<std::unique_ptr<row_ir::node>> operands;
  for (auto const& operand : expr.get_operands()) {
    operands.push_back(operand.get().accept(*this));
  }
  return std::make_unique<row_ir::operation>(expr.get_operator(), std::move(operands));
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::column_name_reference const& expr)
{
  auto index = add_ast_input(ast_named_column_input_spec{expr.get_column_name()});
  return std::make_unique<row_ir::get_input>(index);
}

void ast_converter::add_input_var(ast_column_input_spec const& in,
                                  bool nullable,
                                  ast_args const& args)
{
  // TODO: mangle column name to make debugging easier
  auto id   = std::format("in_{}", input_vars_.size());
  auto type = args.table.column(in.column).type();
  input_vars_.emplace_back(std::move(id), type_info{type, nullable});
}

void ast_converter::add_input_var(ast_named_column_input_spec const& in,
                                  bool nullable,
                                  ast_args const& args)
{
  auto column_index_iter = args.table_column_names.find(in.name);
  CUDF_EXPECTS(column_index_iter != args.table_column_names.end(),
               "Column name not found in table.");
  return add_input_var(
    ast_column_input_spec{ast::table_reference::LEFT, column_index_iter->second}, nullable, args);
}

void ast_converter::add_input_var(ast_scalar_input_spec const& in,
                                  bool nullable,
                                  [[maybe_unused]] ast_args const& args)
{
  auto id   = std::format("in_{}", input_vars_.size());
  auto type = in.ref.get().type();
  input_vars_.emplace_back(std::move(id), type_info{type, nullable});
}

void ast_converter::add_output_var()
{
  auto id = std::format("out_{}", output_vars_.size());
  output_vars_.emplace_back(std::move(id));
}

void ast_converter::clear()
{
  input_specs_.clear();
  input_vars_.clear();
  output_vars_.clear();
  output_irs_.clear();
  code_.clear();
}

template <typename Fn, typename... Args>
decltype(auto) dispatch_input_spec(ast_input_spec const& in, Fn&& fn, Args&&... args)
{
  if (std::holds_alternative<ast_column_input_spec>(in)) {
    return fn(std::get<ast_column_input_spec>(in), std::forward<Args>(args)...);
  } else if (std::holds_alternative<ast_named_column_input_spec>(in)) {
    return fn(std::get<ast_named_column_input_spec>(in), std::forward<Args>(args)...);
  } else if (std::holds_alternative<ast_scalar_input_spec>(in)) {
    return fn(std::get<ast_scalar_input_spec>(in), std::forward<Args>(args)...);
  } else {
    CUDF_FAIL("Unsupported input type");
  }
}

column_view get_column_view(ast_column_input_spec const& spec, ast_args const& args)
{
  CUDF_EXPECTS(spec.table == ast::table_reference::LEFT, "Table reference must be LEFT");
  return args.table.column(spec.column);
}

column_view get_column_view(ast_named_column_input_spec const& spec, ast_args const& args)
{
  auto column_index_iter = args.table_column_names.find(spec.name);
  CUDF_EXPECTS(column_index_iter != args.table_column_names.end(),
               "Column name not found in table.");
  return args.table.column(column_index_iter->second);
}

column_view get_column_view(ast_scalar_input_spec const& spec, ast_args const& args)
{
  return spec.broadcast_column->view();
}

bool map_copy_mask(ast_column_input_spec const& spec,
                   ast_args const&,
                   std::vector<bool> const& copy_mask)
{
  CUDF_EXPECTS(spec.table == ast::table_reference::LEFT, "Table reference must be LEFT");
  return copy_mask[spec.column];
}

bool map_copy_mask(ast_named_column_input_spec const& spec,
                   ast_args const& args,
                   std::vector<bool> const& copy_mask)
{
  auto column_index_iter = args.table_column_names.find(spec.name);
  CUDF_EXPECTS(column_index_iter != args.table_column_names.end(),
               "Column name not found in table.");
  return copy_mask[column_index_iter->second];
}

bool map_copy_mask(ast_scalar_input_spec const&, ast_args const&, std::vector<bool> const&)
{
  return false;  // AST Scalars do not have a copy mask
}

void ast_converter::generate_code(target target_id,
                                  ast::expression const& expr,
                                  bool null_aware,
                                  ast_args const& args,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref const& resource_ref)
{
  auto output_expr_ir = expr.accept(*this);
  output_irs_.emplace_back(std::make_unique<row_ir::set_output>(0, std::move(output_expr_ir)));

  // resolve the flattened input references into IR input variables
  for (auto const& input : input_specs_) {
    dispatch_input_spec(input, [this](auto&... args) { add_input_var(args...); }, null_aware, args);
  }

  // add 1 auto-deduced output variable
  add_output_var();

  instance_context instance_ctx;
  instance_info instance{input_vars_, output_vars_};

  // instantiate the IR nodes
  for (auto& ir : output_irs_) {
    ir->instantiate(instance_ctx, instance);
  }

  target_info target{target_id};

  std::string body;

  for (auto& ir : output_irs_) {
    body = std::format("{}{}{}", body, ir->generate_code(instance_ctx, target, instance), "\n");
  }

  switch (target.id) {
    case target::CUDA: {
      {
        auto output_decl = [&](size_t i) {
          auto const& var  = output_vars_[i];
          auto const& ir   = output_irs_[i];
          auto output_type = ir->get_type();
          return std::format("{} * {}", cuda_type(output_type), var.id);
        };

        auto input_decl = [&](size_t i) {
          auto const& var = input_vars_[i];
          return std::format("{} {}", cuda_type(var.type), var.id);
        };

        std::vector<std::string> params_decls;

        for (size_t i = 0; i < input_vars_.size(); ++i) {
          params_decls.push_back(input_decl(i));
        }

        for (size_t i = 0; i < output_vars_.size(); ++i) {
          params_decls.push_back(output_decl(i));
        }

        std::string params_decl;

        if (params_decls.empty()) {
        } else if (params_decls.size() == 1) {
          params_decl = params_decls[0];
        } else {
          params_decl = std::accumulate(
            params_decls.begin() + 1,
            params_decls.end(),
            params_decls[0],
            [](auto const& a, auto const& b) { return std::format("{}, {}", a, b); });
        }

        code_ = std::format(
          R"***(
__device__ void transform({})
{{
{}
return;
}}
)***",
          params_decl,
          body);
      }
      break;
    }
    default: CUDF_FAIL("Unsupported target: " + std::to_string(static_cast<int>(target.id)));
  }
}

// Due to the AST expression tree structure, we can't generate the IR without the target
// tables

transform_result ast_converter::compute_column(target target_id,
                                               ast::expression const& expr,
                                               bool null_aware,
                                               ast_args const& args,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref const& resource_ref)
{
  clear();
  generate_code(target_id, expr, null_aware, args, stream, resource_ref);

  std::vector<column_view> columns;
  std::vector<std::unique_ptr<column>> scalar_columns;

  for (auto& input : input_specs_) {
    auto column_view =
      dispatch_input_spec(input, [](auto&... args) { return get_column_view(args...); }, args);
    columns.push_back(column_view);

    if (std::holds_alternative<ast_scalar_input_spec>(input)) {
      auto& scalar_input = std::get<ast_scalar_input_spec>(input);
      scalar_columns.push_back(std::move(scalar_input.broadcast_column));
    }
  }

  auto output_column_type = output_irs_[0]->get_type();

  transform_result transform{
    transform_args{std::move(columns), std::move(code_), output_column_type.type, false},
    std::move(scalar_columns)};

  clear();

  return transform;
}

filter_result ast_converter::filter(target target_id,
                                    ast::expression const& expr,
                                    bool null_aware,
                                    ast_args const& args,
                                    std::optional<std::vector<bool>> table_copy_mask,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref const& resource_ref)
{
  clear();
  generate_code(target_id, expr, null_aware, args, stream, resource_ref);

  CUDF_EXPECTS(output_irs_[0]->get_type().type == data_type{type_id::BOOL8},
               "Filter expression must return a boolean type.");
  CUDF_EXPECTS(!output_irs_[0]->get_type().nullable,
               "Filter expression must return a non-nullable boolean type.");

  std::vector<column_view> columns;
  std::vector<std::unique_ptr<column>> scalar_columns;
  std::vector<bool> copy_mask;

  for (auto& input : input_specs_) {
    auto column_view =
      dispatch_input_spec(input, [](auto&... args) { return get_column_view(args...); }, args);
    columns.push_back(column_view);

    // move the scalar broadcast column so the user can make it live long enough
    // to be used in the filter result.
    if (std::holds_alternative<ast_scalar_input_spec>(input)) {
      auto& scalar_input = std::get<ast_scalar_input_spec>(input);
      scalar_columns.push_back(std::move(scalar_input.broadcast_column));
    }

    // if the table_copy_mask is provided, we need to extract the copy mask for each input
    if (table_copy_mask.has_value()) {
      copy_mask.push_back(dispatch_input_spec(
        input, [](auto&... args) { return map_copy_mask(args...); }, args, *table_copy_mask));
    } else {
      copy_mask.push_back(true);
    }
  }

  filter_result filter{
    filter_args{std::move(columns), std::move(code_), false, std::nullopt, copy_mask},
    std::move(scalar_columns)};

  clear();

  return filter;
}

}  // namespace row_ir
}  // namespace cudf
