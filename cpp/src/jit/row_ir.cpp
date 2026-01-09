/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir.hpp"

#include "runtime/context.hpp"

#include <cudf/column/column_factories.hpp>

#include <algorithm>
#include <format>
#include <iostream>
#include <iterator>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>

namespace cudf {

namespace detail {

namespace row_ir {

std::string cuda_type(cudf::data_type type, bool nullable)
{
  auto name = type_to_name(type);
  return nullable ? std::format("cuda::std::optional<{}>", name) : name;
}

std::string instance_context::make_tmp_id()
{
  return std::format("{}{}", tmp_prefix_, num_tmp_vars_++);
}

bool instance_context::has_nulls() const { return has_nulls_; }

void instance_context::set_has_nulls(bool has_nulls) { has_nulls_ = has_nulls; }

get_input::get_input(int32_t input) : id_(), input_(input), type_() {}

std::string_view get_input::get_id() { return id_; }

data_type get_input::get_type() { return type_; }

bool get_input::is_null_aware() { return false; }

bool get_input::is_always_valid() { return false; }

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
      return std::format(
        "{} {} = {};", cuda_type(type_, ctx.has_nulls()), id_, instance.inputs[input_].id);
    }
    default:
      CUDF_FAIL("Unsupported target: " + std::to_string(static_cast<int>(info.id)),
                std::invalid_argument);
  }
}

set_output::set_output(int32_t output, std::unique_ptr<node> source)
  : id_(), output_(output), source_(std::move(source)), type_(), output_id_()
{
}

std::string_view set_output::get_id() { return id_; }

data_type set_output::get_type() { return type_; }

bool set_output::is_null_aware() { return source_->is_null_aware(); }

bool set_output::is_always_valid() { return source_->is_always_valid(); }

node& set_output::get_source() { return *source_; }

void set_output::instantiate(instance_context& ctx, instance_info const& info)
{
  source_->instantiate(ctx, info);
  id_              = ctx.make_tmp_id();
  auto source_type = source_->get_type();
  type_            = source_type;
  output_id_       = info.outputs[output_].id;
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
        cuda_type(type_, ctx.has_nulls()),
        id_,
        source_->get_id(),
        output_id_,
        id_);
    }
    default:
      CUDF_FAIL("Unsupported target: " + std::to_string(static_cast<int>(info.id)),
                std::invalid_argument);
  }
}

operation::operation(opcode op, std::unique_ptr<node>* move_begin, std::unique_ptr<node>* move_end)
  : id_(), op_(op), operands_(), type_()
{
  operands_.insert(
    operands_.begin(), std::make_move_iterator(move_begin), std::make_move_iterator(move_end));
  CUDF_EXPECTS(static_cast<size_type>(operands_.size()) == ast::detail::ast_operator_arity(op),
               "Invalid number of arguments for operator.",
               std::invalid_argument);
  CUDF_EXPECTS(
    operands_.size() > 0, "Operator must have at least one operand", std::invalid_argument);
}

operation::operation(opcode op, std::vector<std::unique_ptr<node>> operands)
  : operation(op, operands.data(), operands.data() + operands.size())
{
}

std::string_view operation::get_id() { return id_; }

data_type operation::get_type() { return type_; }

inline bool is_operator_null_aware(opcode op)
{
  switch (op) {
    case ast::ast_operator::IS_NULL:
    case ast::ast_operator::NULL_EQUAL:
    case ast::ast_operator::NULL_LOGICAL_AND:
    case ast::ast_operator::NULL_LOGICAL_OR: return true;

    case ast::ast_operator::ADD:
    case ast::ast_operator::SUB:
    case ast::ast_operator::MUL:
    case ast::ast_operator::DIV:
    case ast::ast_operator::TRUE_DIV:
    case ast::ast_operator::FLOOR_DIV:
    case ast::ast_operator::MOD:
    case ast::ast_operator::PYMOD:
    case ast::ast_operator::POW:
    case ast::ast_operator::NOT_EQUAL:
    case ast::ast_operator::EQUAL:
    case ast::ast_operator::LESS:
    case ast::ast_operator::GREATER:
    case ast::ast_operator::LESS_EQUAL:
    case ast::ast_operator::GREATER_EQUAL:
    case ast::ast_operator::BITWISE_AND:
    case ast::ast_operator::BITWISE_OR:
    case ast::ast_operator::BITWISE_XOR:
    case ast::ast_operator::LOGICAL_AND:
    case ast::ast_operator::LOGICAL_OR:
    case ast::ast_operator::IDENTITY:
    case ast::ast_operator::SIN:
    case ast::ast_operator::COS:
    case ast::ast_operator::TAN:
    case ast::ast_operator::ARCSIN:
    case ast::ast_operator::ARCCOS:
    case ast::ast_operator::ARCTAN:
    case ast::ast_operator::SINH:
    case ast::ast_operator::COSH:
    case ast::ast_operator::TANH:
    case ast::ast_operator::ARCSINH:
    case ast::ast_operator::ARCCOSH:
    case ast::ast_operator::ARCTANH:
    case ast::ast_operator::EXP:
    case ast::ast_operator::LOG:
    case ast::ast_operator::SQRT:
    case ast::ast_operator::CBRT:
    case ast::ast_operator::CEIL:
    case ast::ast_operator::FLOOR:
    case ast::ast_operator::ABS:
    case ast::ast_operator::RINT:
    case ast::ast_operator::BIT_INVERT:
    case ast::ast_operator::NOT:
    case ast::ast_operator::CAST_TO_INT64:
    case ast::ast_operator::CAST_TO_UINT64:
    case ast::ast_operator::CAST_TO_FLOAT64: return false;

    default: CUDF_UNREACHABLE("Unrecognized operator type.");
  }
}

bool operation::is_null_aware()
{
  return is_operator_null_aware(op_) ||
         std::any_of(
           operands_.begin(), operands_.end(), [](auto& op) { return op->is_null_aware(); });
}

inline bool is_operator_always_valid(opcode op)
{
  switch (op) {
    case ast::ast_operator::IS_NULL:
    case ast::ast_operator::NULL_EQUAL: return true;

    case ast::ast_operator::NULL_LOGICAL_AND:
    case ast::ast_operator::NULL_LOGICAL_OR:
    case ast::ast_operator::ADD:
    case ast::ast_operator::SUB:
    case ast::ast_operator::MUL:
    case ast::ast_operator::DIV:
    case ast::ast_operator::TRUE_DIV:
    case ast::ast_operator::FLOOR_DIV:
    case ast::ast_operator::MOD:
    case ast::ast_operator::PYMOD:
    case ast::ast_operator::POW:
    case ast::ast_operator::NOT_EQUAL:
    case ast::ast_operator::EQUAL:
    case ast::ast_operator::LESS:
    case ast::ast_operator::GREATER:
    case ast::ast_operator::LESS_EQUAL:
    case ast::ast_operator::GREATER_EQUAL:
    case ast::ast_operator::BITWISE_AND:
    case ast::ast_operator::BITWISE_OR:
    case ast::ast_operator::BITWISE_XOR:
    case ast::ast_operator::LOGICAL_AND:
    case ast::ast_operator::LOGICAL_OR:
    case ast::ast_operator::IDENTITY:
    case ast::ast_operator::SIN:
    case ast::ast_operator::COS:
    case ast::ast_operator::TAN:
    case ast::ast_operator::ARCSIN:
    case ast::ast_operator::ARCCOS:
    case ast::ast_operator::ARCTAN:
    case ast::ast_operator::SINH:
    case ast::ast_operator::COSH:
    case ast::ast_operator::TANH:
    case ast::ast_operator::ARCSINH:
    case ast::ast_operator::ARCCOSH:
    case ast::ast_operator::ARCTANH:
    case ast::ast_operator::EXP:
    case ast::ast_operator::LOG:
    case ast::ast_operator::SQRT:
    case ast::ast_operator::CBRT:
    case ast::ast_operator::CEIL:
    case ast::ast_operator::FLOOR:
    case ast::ast_operator::ABS:
    case ast::ast_operator::RINT:
    case ast::ast_operator::BIT_INVERT:
    case ast::ast_operator::NOT:
    case ast::ast_operator::CAST_TO_INT64:
    case ast::ast_operator::CAST_TO_UINT64:
    case ast::ast_operator::CAST_TO_FLOAT64: return false;

    default: CUDF_UNREACHABLE("Unrecognized operator type.");
  }
}

bool operation::is_always_valid()
{
  return is_operator_always_valid(op_) ||
         std::all_of(
           operands_.begin(), operands_.end(), [](auto& op) { return op->is_always_valid(); });
}

opcode operation::get_opcode() const { return op_; }

std::span<std::unique_ptr<node> const> operation::get_operands() const { return operands_; }

void operation::instantiate(instance_context& ctx, instance_info const& info)
{
  for (auto& arg : operands_) {
    arg->instantiate(ctx, info);
  }

  id_ = ctx.make_tmp_id();
  std::vector<data_type> operand_types;

  for (auto& arg : operands_) {
    operand_types.emplace_back(arg->get_type());
  }

  type_ = ast::detail::ast_operator_return_type(op_, operand_types);
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

        auto cuda = std::format(
          "{} {} = cudf::ast::detail::operator_functor<cudf::ast::ast_operator::{}, "
          "{}>{{}}({});",
          cuda_type(type_, ctx.has_nulls()),
          id_,
          ast::detail::ast_operator_string(op_),
          ctx.has_nulls(),
          operands_str);
        return cuda;
      }
      default:
        CUDF_FAIL("Unsupported target: " + std::to_string(static_cast<int>(info.id)),
                  std::invalid_argument);
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
  auto index = add_ast_input(
    ast_scalar_input_spec{expr.get_scalar(),
                          expr.get_value(),
                          make_column_from_scalar(expr.get_scalar(), 1, stream_, mr_)});
  return std::make_unique<row_ir::get_input>(index);
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::column_reference const& expr)
{
  CUDF_EXPECTS(expr.get_table_source() == ast::table_reference::LEFT,
               "IR input column reference must be LEFT.",
               std::invalid_argument);
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

void ast_converter::add_input_var(ast_column_input_spec const& in, ast_args const& args)
{
  // TODO(lamarrr): consider mangling column name to make debugging easier
  auto id   = std::format("in_{}", input_vars_.size());
  auto type = args.table.column(in.column).type();
  input_vars_.emplace_back(std::move(id), type);
}

void ast_converter::add_input_var(ast_scalar_input_spec const& in,
                                  [[maybe_unused]] ast_args const& args)
{
  auto id   = std::format("in_{}", input_vars_.size());
  auto type = in.ref.get().type();
  input_vars_.emplace_back(std::move(id), type);
}

void ast_converter::add_output_var()
{
  auto id = std::format("out_{}", output_vars_.size());
  output_vars_.emplace_back(std::move(id));
}

template <typename Fn, typename... Args>
decltype(auto) dispatch_input_spec(ast_input_spec const& in, Fn&& fn, Args&&... args)
{
  if (std::holds_alternative<ast_column_input_spec>(in)) {
    return fn(std::get<ast_column_input_spec>(in), std::forward<Args>(args)...);
  } else if (std::holds_alternative<ast_scalar_input_spec>(in)) {
    return fn(std::get<ast_scalar_input_spec>(in), std::forward<Args>(args)...);
  } else {
    CUDF_FAIL("Unsupported input type");
  }
}

column_view get_column_view(ast_column_input_spec const& spec, ast_args const& args)
{
  CUDF_EXPECTS(spec.table == ast::table_reference::LEFT,
               "Table reference must be LEFT",
               std::invalid_argument);
  return args.table.column(spec.column);
}

column_view get_column_view(ast_scalar_input_spec const& spec, ast_args const& args)
{
  return spec.broadcast_column->view();
}

std::tuple<null_aware, output_nullability> ast_converter::generate_code(target target_id,
                                                                        ast::expression const& expr,
                                                                        ast_args const& args)
{
  auto output_expr_ir = expr.accept(*this);
  output_irs_.emplace_back(std::make_unique<row_ir::set_output>(0, std::move(output_expr_ir)));

  bool uses_input_table =
    std::any_of(input_specs_.begin(), input_specs_.end(), [](auto const& spec) {
      return std::holds_alternative<ast_column_input_spec>(spec);
    });

  if (!uses_input_table && args.table.num_columns() > 0) {
    // this means none of the inputs tables to the IR are actually used in the expression. In
    // order to still run the transform-equivalent operation of AST, we need to add one of the
    // table's columns as an unused input. This is done because the output size of a transform is
    // determined by the largest input column.
    input_specs_.emplace_back(ast_column_input_spec{ast::table_reference::LEFT, 0});
  }

  // resolve the flattened input references into IR input variables
  for (auto const& input : input_specs_) {
    dispatch_input_spec(input, [this](auto&... args) { add_input_var(args...); }, args);
  }

  bool has_nullable_inputs =
    std::any_of(input_specs_.begin(), input_specs_.end(), [&](auto const& input) {
      return dispatch_input_spec(
               input, [](auto&... args) { return get_column_view(args...); }, args)
        .nullable();
    });

  // add 1 auto-deduced output variable
  add_output_var();

  instance_context instance_ctx;
  instance_info instance{input_vars_, output_vars_};

  auto is_null_aware =
    std::any_of(
      output_irs_.cbegin(), output_irs_.cend(), [](auto& ir) { return ir->is_null_aware(); })
      ? null_aware::YES
      : null_aware::NO;

  bool output_is_always_valid = std::all_of(
    output_irs_.cbegin(), output_irs_.cend(), [](auto& ir) { return ir->is_always_valid(); });

  bool may_evaluate_null = !output_is_always_valid && has_nullable_inputs;
  auto null_policy =
    may_evaluate_null ? output_nullability::PRESERVE : output_nullability::ALL_VALID;

  instance_ctx.set_has_nulls(is_null_aware == null_aware::YES);

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
          return std::format("{}* {}", cuda_type(output_type, instance_ctx.has_nulls()), var.id);
        };

        auto input_decl = [&](size_t i) {
          auto const& var = input_vars_[i];
          return std::format("{} {}", cuda_type(var.type, instance_ctx.has_nulls()), var.id);
        };

        std::vector<std::string> params_decls;

        for (size_t i = 0; i < output_vars_.size(); ++i) {
          params_decls.push_back(output_decl(i));
        }

        for (size_t i = 0; i < input_vars_.size(); ++i) {
          params_decls.push_back(input_decl(i));
        }

        auto params_decl = [&] {
          if (params_decls.empty()) {
            return std::string{};
          } else if (params_decls.size() == 1) {
            return params_decls[0];
          } else {
            return std::accumulate(
              params_decls.begin() + 1,
              params_decls.end(),
              params_decls[0],
              [](auto const& a, auto const& b) { return std::format("{}, {}", a, b); });
          }
        }();

        code_ = std::format(
          R"***(
__device__ void expression({})
{{
{}
return;
}}
)***",
          params_decl,
          body);

        return {is_null_aware, null_policy};
      }
      break;
    }
    default:
      CUDF_FAIL("Unsupported target: " + std::to_string(static_cast<int>(target.id)),
                std::invalid_argument);
  }
}

// Due to the AST expression tree structure, we can't generate the IR without the target
// tables
transform_args ast_converter::compute_column(target target_id,
                                             ast::expression const& expr,
                                             ast_args const& args,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  ast_converter converter{stream, mr};

  // TODO(lamarrr): consider deduplicating ast expression's input column references. See
  // TransformTest/1.DeeplyNestedArithmeticLogicalExpression for reference

  auto [is_null_aware, output_nullability] = converter.generate_code(target_id, expr, args);

  std::vector<column_view> columns;
  std::vector<std::unique_ptr<column>> scalar_columns;

  for (auto& input : converter.input_specs_) {
    auto column_view =
      dispatch_input_spec(input, [](auto&... args) { return get_column_view(args...); }, args);
    columns.push_back(column_view);

    if (std::holds_alternative<ast_scalar_input_spec>(input)) {
      auto& scalar_input = std::get<ast_scalar_input_spec>(input);
      scalar_columns.push_back(std::move(scalar_input.broadcast_column));
    }
  }

  auto& out               = converter.output_irs_[0];
  auto output_column_type = out->get_type();

  transform_args transform{std::move(scalar_columns),
                           std::move(columns),
                           std::move(converter.code_),
                           output_column_type,
                           false,
                           std::nullopt,
                           is_null_aware,
                           output_nullability};

  if (get_context().dump_codegen()) {
    std::cout << "Generated code for transform: " << transform.udf << std::endl;
  }

  return transform;
}

filter_args ast_converter::filter(target target_id,
                                  ast::expression const& expr,
                                  ast_args const& args,
                                  table_view const& filter_table,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  ast_converter converter{stream, mr};
  auto [is_null_aware, _null_output] = converter.generate_code(target_id, expr, args);

  CUDF_EXPECTS(converter.output_irs_.size() == 1,
               "Filter expression must return a single output.",
               std::invalid_argument);

  auto& out_ir = converter.output_irs_[0];

  CUDF_EXPECTS(out_ir->get_type() == data_type{type_id::BOOL8},
               "Filter expression must return a boolean type.",
               std::invalid_argument);

  std::vector<column_view> predicate_columns;
  std::vector<std::unique_ptr<column>> scalar_columns;

  for (auto& input : converter.input_specs_) {
    auto column_view =
      dispatch_input_spec(input, [](auto&... args) { return get_column_view(args...); }, args);
    predicate_columns.push_back(column_view);

    // move the scalar broadcast column so the user can make it live long enough
    // to be used in the filter result.
    if (std::holds_alternative<ast_scalar_input_spec>(input)) {
      auto& scalar_input = std::get<ast_scalar_input_spec>(input);
      scalar_columns.push_back(std::move(scalar_input.broadcast_column));
    }
  }

  std::vector<column_view> filter_columns;
  std::transform(filter_table.begin(),
                 filter_table.end(),
                 std::back_inserter(filter_columns),
                 [](auto const& col) { return col; });

  filter_args filter{std::move(scalar_columns),
                     std::move(predicate_columns),
                     std::move(converter.code_),
                     std::move(filter_columns),
                     false,
                     std::nullopt,
                     is_null_aware};

  if (get_context().dump_codegen()) {
    std::cout << "Generated code for filter: " << filter.predicate_udf << std::endl;
  }

  return filter;
}

}  // namespace row_ir
}  // namespace detail
}  // namespace cudf
