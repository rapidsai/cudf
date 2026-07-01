/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir.hpp"

#include "runtime/context.hpp"

#include <cudf/column/column_factories.hpp>

#include <algorithm>
#include <format>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>

namespace cudf::detail::row_ir {

inline ast::ast_operator as_ast_op(opcode op)
{
  switch (op) {
    case opcode::ADD: return ast::ast_operator::ADD;
    case opcode::SUB: return ast::ast_operator::SUB;
    case opcode::MUL: return ast::ast_operator::MUL;
    case opcode::DIV: return ast::ast_operator::DIV;
    case opcode::TRUE_DIV: return ast::ast_operator::TRUE_DIV;
    case opcode::FLOOR_DIV: return ast::ast_operator::FLOOR_DIV;
    case opcode::MOD: return ast::ast_operator::MOD;
    case opcode::PYMOD: return ast::ast_operator::PYMOD;
    case opcode::POW: return ast::ast_operator::POW;
    case opcode::EQUAL: return ast::ast_operator::EQUAL;
    case opcode::NULL_EQUAL: return ast::ast_operator::NULL_EQUAL;
    case opcode::NOT_EQUAL: return ast::ast_operator::NOT_EQUAL;
    case opcode::LESS: return ast::ast_operator::LESS;
    case opcode::GREATER: return ast::ast_operator::GREATER;
    case opcode::LESS_EQUAL: return ast::ast_operator::LESS_EQUAL;
    case opcode::GREATER_EQUAL: return ast::ast_operator::GREATER_EQUAL;
    case opcode::BITWISE_AND: return ast::ast_operator::BITWISE_AND;
    case opcode::BITWISE_OR: return ast::ast_operator::BITWISE_OR;
    case opcode::BITWISE_XOR: return ast::ast_operator::BITWISE_XOR;
    case opcode::LOGICAL_AND: return ast::ast_operator::LOGICAL_AND;
    case opcode::NULL_LOGICAL_AND: return ast::ast_operator::NULL_LOGICAL_AND;
    case opcode::LOGICAL_OR: return ast::ast_operator::LOGICAL_OR;
    case opcode::NULL_LOGICAL_OR: return ast::ast_operator::NULL_LOGICAL_OR;
    case opcode::IDENTITY: return ast::ast_operator::IDENTITY;
    case opcode::IS_NULL: return ast::ast_operator::IS_NULL;
    case opcode::SIN: return ast::ast_operator::SIN;
    case opcode::COS: return ast::ast_operator::COS;
    case opcode::TAN: return ast::ast_operator::TAN;
    case opcode::ARCSIN: return ast::ast_operator::ARCSIN;
    case opcode::ARCCOS: return ast::ast_operator::ARCCOS;
    case opcode::ARCTAN: return ast::ast_operator::ARCTAN;
    case opcode::SINH: return ast::ast_operator::SINH;
    case opcode::COSH: return ast::ast_operator::COSH;
    case opcode::TANH: return ast::ast_operator::TANH;
    case opcode::ARCSINH: return ast::ast_operator::ARCSINH;
    case opcode::ARCCOSH: return ast::ast_operator::ARCCOSH;
    case opcode::ARCTANH: return ast::ast_operator::ARCTANH;
    case opcode::EXP: return ast::ast_operator::EXP;
    case opcode::LOG: return ast::ast_operator::LOG;
    case opcode::SQRT: return ast::ast_operator::SQRT;
    case opcode::CBRT: return ast::ast_operator::CBRT;
    case opcode::CEIL: return ast::ast_operator::CEIL;
    case opcode::FLOOR: return ast::ast_operator::FLOOR;
    case opcode::ABS: return ast::ast_operator::ABS;
    case opcode::RINT: return ast::ast_operator::RINT;
    case opcode::BIT_INVERT: return ast::ast_operator::BIT_INVERT;
    case opcode::NOT: return ast::ast_operator::NOT;
    case opcode::CAST_TO_INT64: return ast::ast_operator::CAST_TO_INT64;
    case opcode::CAST_TO_UINT64: return ast::ast_operator::CAST_TO_UINT64;
    case opcode::CAST_TO_FLOAT64: return ast::ast_operator::CAST_TO_FLOAT64;
    case opcode::CAST_TO_DECIMAL32: return ast::ast_operator::CAST_TO_DECIMAL32;
    case opcode::CAST_TO_DECIMAL64: return ast::ast_operator::CAST_TO_DECIMAL64;
    case opcode::CAST_TO_DECIMAL128: return ast::ast_operator::CAST_TO_DECIMAL128;
    default: CUDF_FAIL("Invalid operator type.");
  }
}

inline opcode as_opcode(ast::ast_operator op)
{
  switch (op) {
    case ast::ast_operator::ADD: return opcode::ADD;
    case ast::ast_operator::SUB: return opcode::SUB;
    case ast::ast_operator::MUL: return opcode::MUL;
    case ast::ast_operator::DIV: return opcode::DIV;
    case ast::ast_operator::TRUE_DIV: return opcode::TRUE_DIV;
    case ast::ast_operator::FLOOR_DIV: return opcode::FLOOR_DIV;
    case ast::ast_operator::MOD: return opcode::MOD;
    case ast::ast_operator::PYMOD: return opcode::PYMOD;
    case ast::ast_operator::POW: return opcode::POW;
    case ast::ast_operator::EQUAL: return opcode::EQUAL;
    case ast::ast_operator::NULL_EQUAL: return opcode::NULL_EQUAL;
    case ast::ast_operator::NOT_EQUAL: return opcode::NOT_EQUAL;
    case ast::ast_operator::LESS: return opcode::LESS;
    case ast::ast_operator::GREATER: return opcode::GREATER;
    case ast::ast_operator::LESS_EQUAL: return opcode::LESS_EQUAL;
    case ast::ast_operator::GREATER_EQUAL: return opcode::GREATER_EQUAL;
    case ast::ast_operator::BITWISE_AND: return opcode::BITWISE_AND;
    case ast::ast_operator::BITWISE_OR: return opcode::BITWISE_OR;
    case ast::ast_operator::BITWISE_XOR: return opcode::BITWISE_XOR;
    case ast::ast_operator::LOGICAL_AND: return opcode::LOGICAL_AND;
    case ast::ast_operator::NULL_LOGICAL_AND: return opcode::NULL_LOGICAL_AND;
    case ast::ast_operator::LOGICAL_OR: return opcode::LOGICAL_OR;
    case ast::ast_operator::NULL_LOGICAL_OR: return opcode::NULL_LOGICAL_OR;
    case ast::ast_operator::IDENTITY: return opcode::IDENTITY;
    case ast::ast_operator::IS_NULL: return opcode::IS_NULL;
    case ast::ast_operator::SIN: return opcode::SIN;
    case ast::ast_operator::COS: return opcode::COS;
    case ast::ast_operator::TAN: return opcode::TAN;
    case ast::ast_operator::ARCSIN: return opcode::ARCSIN;
    case ast::ast_operator::ARCCOS: return opcode::ARCCOS;
    case ast::ast_operator::ARCTAN: return opcode::ARCTAN;
    case ast::ast_operator::SINH: return opcode::SINH;
    case ast::ast_operator::COSH: return opcode::COSH;
    case ast::ast_operator::TANH: return opcode::TANH;
    case ast::ast_operator::ARCSINH: return opcode::ARCSINH;
    case ast::ast_operator::ARCCOSH: return opcode::ARCCOSH;
    case ast::ast_operator::ARCTANH: return opcode::ARCTANH;
    case ast::ast_operator::EXP: return opcode::EXP;
    case ast::ast_operator::LOG: return opcode::LOG;
    case ast::ast_operator::SQRT: return opcode::SQRT;
    case ast::ast_operator::CBRT: return opcode::CBRT;
    case ast::ast_operator::CEIL: return opcode::CEIL;
    case ast::ast_operator::FLOOR: return opcode::FLOOR;
    case ast::ast_operator::ABS: return opcode::ABS;
    case ast::ast_operator::RINT: return opcode::RINT;
    case ast::ast_operator::BIT_INVERT: return opcode::BIT_INVERT;
    case ast::ast_operator::NOT: return opcode::NOT;
    case ast::ast_operator::CAST_TO_INT64: return opcode::CAST_TO_INT64;
    case ast::ast_operator::CAST_TO_UINT64: return opcode::CAST_TO_UINT64;
    case ast::ast_operator::CAST_TO_FLOAT64: return opcode::CAST_TO_FLOAT64;
    case ast::ast_operator::CAST_TO_DECIMAL32: return opcode::CAST_TO_DECIMAL32;
    case ast::ast_operator::CAST_TO_DECIMAL64: return opcode::CAST_TO_DECIMAL64;
    case ast::ast_operator::CAST_TO_DECIMAL128: return opcode::CAST_TO_DECIMAL128;
    default: CUDF_FAIL("Invalid operator type.");
  }
}

int32_t instance_context::add_output()
{
  auto id     = static_cast<int32_t>(output_vars_.size());
  auto id_str = std::format("out_{}", id);
  output_vars_.emplace_back(std::move(id_str));
  return id;
}

int32_t instance_context::add_input(input in)
{
  auto id     = static_cast<int32_t>(inputs_.size());
  auto id_str = std::format("in_{}", id);

  data_type const type = [&in] {
    if (auto* col = std::get_if<column_input>(&in)) {
      return col->column.type();
    } else {
      auto& scalar = std::get<scalar_input>(in);
      return scalar.scalar_column->type();
    }
  }();
  inputs_.emplace_back(std::move(in));
  input_vars_.emplace_back(std::move(id_str), type);
  return id;
}

std::string instance_context::make_tmp_id()
{
  return std::format("{}{}", tmp_prefix_, num_tmp_vars_++);
}

bool instance_context::has_nulls() const { return has_nulls_; }

void instance_context::set_has_nulls(bool has_nulls) { has_nulls_ = has_nulls; }

std::span<input const> instance_context::get_inputs() const { return inputs_; }

std::span<var_info const> instance_context::get_input_vars() const { return input_vars_; }

std::span<untyped_var_info const> instance_context::get_output_vars() const { return output_vars_; }

node::node(opcode op, std::optional<int32_t> target_scale, std::vector<std::unique_ptr<node>> args)
  : op_{op}, target_scale_{target_scale}, args_{std::move(args)}
{
  CUDF_EXPECTS(op_ != opcode::GET_INPUT && op_ != opcode::SET_OUTPUT,
               std::format("Invalid opcode `{}` for operation node.", static_cast<int>(op_)),
               std::runtime_error);
  CUDF_EXPECTS(
    op_ != opcode::RESCALE, "Opcode `RESCALE` is not implemented yet", std::runtime_error);

  auto expected_arity = op == opcode::PREDICATE
                          ? 1
                          : static_cast<size_t>(ast::detail::ast_operator_arity(as_ast_op(op_)));
  auto actual_arity   = args_.size();
  CUDF_EXPECTS(actual_arity == expected_arity,
               std::format("Invalid number of arguments for operator `{}`. Expected {}, Got {}.",
                           static_cast<int>(op_),
                           expected_arity,
                           actual_arity),
               std::runtime_error);
}

node::node(input_reference input)
  : reference_{input}, op_{opcode::GET_INPUT}  // NOLINT(modernize-use-default-member-init)
{
}

node::node(output_reference reference, std::unique_ptr<node> arg)
  : reference_{reference}, op_{opcode::SET_OUTPUT}
{
  args_.emplace_back(std::move(arg));
}

node::node(output_reference reference, node arg)
  : node{reference, std::make_unique<node>(std::move(arg))}
{
}

std::string_view node::get_id() const { return id_; }

data_type node::get_type() const { return type_; }

std::optional<int32_t> node::get_target_scale() const { return target_scale_; }

opcode node::get_opcode() const { return op_; }

std::span<std::unique_ptr<node> const> node::get_args() const { return args_; }

inline bool get_op_requires_nulls(opcode op)
{
  switch (op) {
    case opcode::IS_NULL:
    case opcode::NULL_EQUAL:
    case opcode::NULL_LOGICAL_AND:
    case opcode::NULL_LOGICAL_OR:
    case opcode::PREDICATE: return true;

    default: return false;
  }
}

enum class [[nodiscard]] null_output : uint8_t {
  PROPAGATE       = 0,
  ALWAYS_VALID    = 1,
  ALWAYS_NULLABLE = 2,
};

[[nodiscard]] inline null_output get_op_null_output(opcode op)
{
  switch (op) {
    case opcode::IS_NULL:
    case opcode::NULL_EQUAL:
    case opcode::PREDICATE: return null_output::ALWAYS_VALID;

    case opcode::NULL_LOGICAL_AND:
    case opcode::NULL_LOGICAL_OR: return null_output::ALWAYS_NULLABLE;

    default: return null_output::PROPAGATE;
  }
}

bool node::is_null_aware() const
{
  if (op_ == opcode::GET_INPUT) { return false; }

  // to emit nulls for always-nullable operators, we  need to mark them as null-aware
  if (get_op_null_output(op_) == null_output::ALWAYS_NULLABLE) { return true; }

  if (get_op_requires_nulls(op_)) { return true; }

  CUDF_EXPECTS(!args_.empty(),
               "Unexpectedly found an operator node with no arguments. All operator nodes should "
               "have at least one argument.",
               std::runtime_error);

  return std::any_of(args_.begin(), args_.end(), [](auto& a) { return a->is_null_aware(); });
}

bool node::is_always_valid() const
{
  if (op_ == opcode::GET_INPUT) { return false; }

  if (get_op_null_output(op_) == null_output::ALWAYS_VALID) { return true; }

  CUDF_EXPECTS(!args_.empty(),
               "Unexpectedly found an operator node with no arguments. All operator nodes should "
               "have at least one argument.",
               std::runtime_error);

  return std::all_of(args_.begin(), args_.end(), [](auto& a) { return a->is_always_valid(); });
}

std::string to_cuda_type(cudf::data_type type, bool nullable)
{
  auto name = type_to_name(type);
  return nullable ? std::format("cuda::std::optional<{}>", name) : name;
}

void node::instantiate(instance_context& ctx)
{
  for (auto& arg : args_) {
    arg->instantiate(ctx);
  }

  id_ = ctx.make_tmp_id();

  switch (op_) {
    case opcode::GET_INPUT: {
      type_ = ctx.get_input_vars()[std::get<input_reference>(reference_).index].type;
    } break;
    case opcode::SET_OUTPUT: {
      type_ = args_[0]->get_type();
    } break;
    case opcode::PREDICATE: {
      CUDF_EXPECTS(args_[0]->get_type().id() == type_id::BOOL8,
                   "Predicate operator requires a boolean argument.",
                   std::runtime_error);
      type_ = data_type{type_id::BOOL8};
    } break;
    case opcode::CAST_TO_DECIMAL32: {
      type_ = data_type{type_id::DECIMAL32, target_scale_.value_or(0)};
    } break;
    case opcode::CAST_TO_DECIMAL64: {
      type_ = data_type{type_id::DECIMAL64, target_scale_.value_or(0)};
    } break;
    case opcode::CAST_TO_DECIMAL128: {
      type_ = data_type{type_id::DECIMAL128, target_scale_.value_or(0)};
    } break;
    default: {
      std::vector<data_type> arg_types;
      for (auto& arg : args_) {
        arg_types.emplace_back(arg->get_type());
      }

      type_ = ast::detail::ast_operator_return_type(as_ast_op(op_), arg_types);
    } break;
  }
}

void node::emit_code(instance_context& instance, target_info const& info, code_sink& sink) const
{
  for (auto& arg : args_) {
    arg->emit_code(instance, info, sink);
  }

  switch (info.id) {
    case target::CUDA: {
      auto type = to_cuda_type(type_, instance.has_nulls());

      switch (op_) {
        case opcode::GET_INPUT: {
          sink.emit(
            std::format(R"***({} {} = {};
)***",
                        type,
                        id_,
                        instance.get_input_vars()[std::get<input_reference>(reference_).index].id));
        } break;

        case opcode::SET_OUTPUT: {
          sink.emit(std::format(
            R"***({} {} = {};
*{} = {};
)***",
            type,
            id_,
            args_[0]->get_id(),
            instance.get_output_vars()[std::get<output_reference>(reference_).index].id,
            id_));
        } break;

        case opcode::CAST_TO_DECIMAL32:
        case opcode::CAST_TO_DECIMAL64:
        case opcode::CAST_TO_DECIMAL128: {
          CUDF_EXPECTS(
            target_scale_.has_value(), "Decimal cast requires target scale", std::runtime_error);
          auto const scale       = *target_scale_;
          auto const neg_scale   = static_cast<int32_t>(-scale);
          auto const& operand_id = args_[0]->get_id();

          // Determine the rep type name and target decimal type name
          std::string_view rep_type_name;
          std::string_view decimal_type_name;
          switch (op_) {
            case opcode::CAST_TO_DECIMAL32:
              rep_type_name     = "int32_t";
              decimal_type_name = "numeric::decimal32";
              break;
            case opcode::CAST_TO_DECIMAL64:
              rep_type_name     = "int64_t";
              decimal_type_name = "numeric::decimal64";
              break;
            case opcode::CAST_TO_DECIMAL128:
              rep_type_name     = "__int128_t";
              decimal_type_name = "numeric::decimal128";
              break;
            default: CUDF_UNREACHABLE("Invalid decimal cast opcode");
          }

          // Determine the conversion code based on source operand type
          // We check at codegen time to avoid if-constexpr issues in NVRTC
          auto const src_type_id = args_[0]->get_type().id();
          std::string conversion_code;
          if (src_type_id == type_id::DECIMAL32 || src_type_id == type_id::DECIMAL64 ||
              src_type_id == type_id::DECIMAL128) {
            // Decimal-to-decimal: rescale using raw rep and scale difference
            conversion_code = std::format(
              "  auto _src_neg_ = static_cast<int32_t>(-_raw_.scale());\n"
              "  auto _combined_ = {} - _src_neg_;\n"
              "  if (_combined_ >= 0) {{\n"
              "    _rep_ = static_cast<_RepT_>(_raw_.value()) * "
              "numeric::detail::exp10<_RepT_>(_combined_);\n"
              "  }} else {{\n"
              "    _rep_ = static_cast<_RepT_>(_raw_.value()) / "
              "numeric::detail::exp10<_RepT_>(-_combined_);\n"
              "  }}\n",
              neg_scale);
          } else if (src_type_id == type_id::FLOAT32 || src_type_id == type_id::FLOAT64) {
            // Float-to-decimal: multiply by 10^neg_scale
            conversion_code = std::format(
              "  auto _mult_ = numeric::detail::exp10<double>({});\n"
              "  _rep_ = static_cast<_RepT_>(_raw_ * _mult_);\n",
              neg_scale);
          } else {
            // Integer-to-decimal: multiply or divide depending on scale sign
            if (neg_scale >= 0) {
              conversion_code = std::format(
                "  auto _mult_ = numeric::detail::exp10<_RepT_>({});\n"
                "  _rep_ = static_cast<_RepT_>(_raw_) * _mult_;\n",
                neg_scale);
            } else {
              conversion_code = std::format(
                "  auto _div_ = numeric::detail::exp10<_RepT_>({});\n"
                "  _rep_ = static_cast<_RepT_>(_raw_) / _div_;\n",
                -neg_scale);
            }
          }

          if (instance.has_nulls()) {
            sink.emit(
              std::format("{} {} = [&]() {{\n"
                          "  auto _val_ = {};\n"
                          "  if (!_val_.has_value()) return {{}};\n"
                          "  auto _raw_ = *_val_;\n"
                          "  using _RepT_ = {};\n"
                          "  _RepT_ _rep_;\n"
                          "{}"
                          "  return {}{{{}{{numeric::scaled_integer<_RepT_>{{_rep_, "
                          "numeric::scale_type{{{}}}}}}}}};\n"
                          "}}();\n",
                          type,
                          id_,
                          operand_id,
                          rep_type_name,
                          conversion_code,
                          type,
                          decimal_type_name,
                          scale));
          } else {
            sink.emit(
              std::format("{} {} = [&]() {{\n"
                          "  auto _raw_ = {};\n"
                          "  using _RepT_ = {};\n"
                          "  _RepT_ _rep_;\n"
                          "{}"
                          "  return {}{{numeric::scaled_integer<_RepT_>{{_rep_, "
                          "numeric::scale_type{{{}}}}}}};\n"
                          "}}();\n",
                          type,
                          id_,
                          operand_id,
                          rep_type_name,
                          conversion_code,
                          decimal_type_name,
                          scale));
          }
        } break;

        default: {
          CUDF_EXPECTS(op_ != opcode::RESCALE, "Rescale is not implemented", std::runtime_error);

          auto first_arg = std::format("{}", args_[0]->get_id());
          auto args_str  = (args_.size() == 1)
                             ? std::string{first_arg}
                             : std::accumulate(args_.begin() + 1,
                                              args_.end(),
                                              std::string{first_arg},
                                              [](auto const& a, auto& node) {
                                                return std::format("{}, {}", a, node->get_id());
                                              });

          if (op_ == opcode::PREDICATE) {
            sink.emit(std::format(
              R"***(bool {} = cudf::detail::ops::predicate({});
)***",
              id_,
              args_str));
          } else {
            sink.emit(std::format(
              R"***({} {} = cudf::ast::detail::operator_functor<cudf::ast::ast_operator::{}>{{}}({});
)***",
              type,
              id_,
              ast::detail::ast_operator_string(as_ast_op(op_)),
              args_str));
          }
        } break;
      }
    } break;

    default:
      CUDF_FAIL(std::format("Unsupported target: {}", static_cast<int>(info.id)),
                std::invalid_argument);
  }
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::literal const& expr)
{
  auto id = instance_.add_input(expr.get_scalar());
  return std::make_unique<row_ir::node>(input_reference{id});
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::column_reference const& expr)
{
  // resolve the table for a column input spec, preferring left_table/right_table for join cases,
  // falling back to args.table for the single-table case.
  auto resolve = [&](ast::table_reference ref) {
    CUDF_EXPECTS(ref == ast::table_reference::LEFT || ref == ast::table_reference::RIGHT,
                 "Invalid table reference in column expression",
                 std::invalid_argument);
    return ref == ast::table_reference::LEFT ? left_table_ : right_table_;
  };

  auto table = resolve(expr.get_table_source());
  auto id    = instance_.add_input(
    column_input{.column       = table.column(expr.get_column_index()),
                    .table_source = (expr.get_table_source() == ast::table_reference::LEFT ? 0 : 1),
                    .column_index = static_cast<int32_t>(expr.get_column_index())});
  return std::make_unique<row_ir::node>(input_reference{id});
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::operation const& expr)
{
  std::vector<std::unique_ptr<row_ir::node>> args;
  for (auto& operand : expr.get_operands()) {
    args.emplace_back(operand.get().accept(*this));
  }
  return std::make_unique<row_ir::node>(
    as_opcode(expr.get_operator()), std::nullopt, std::move(args));
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::cast const& expr)
{
  auto operand     = expr.get_operand().accept(*this);
  auto target_type = expr.get_target_type();

  // Map the cast target type to the appropriate opcode
  opcode cast_op;
  switch (target_type.id()) {
    case type_id::DECIMAL32: cast_op = opcode::CAST_TO_DECIMAL32; break;
    case type_id::DECIMAL64: cast_op = opcode::CAST_TO_DECIMAL64; break;
    case type_id::DECIMAL128: cast_op = opcode::CAST_TO_DECIMAL128; break;
    case type_id::INT64: cast_op = opcode::CAST_TO_INT64; break;
    case type_id::UINT64: cast_op = opcode::CAST_TO_UINT64; break;
    case type_id::FLOAT64: cast_op = opcode::CAST_TO_FLOAT64; break;
    default:
      CUDF_FAIL("Unsupported cast target type for JIT: " +
                  std::to_string(static_cast<int>(target_type.id())),
                std::invalid_argument);
  }

  std::optional<int32_t> scale = std::nullopt;
  if (target_type.id() == type_id::DECIMAL32 || target_type.id() == type_id::DECIMAL64 ||
      target_type.id() == type_id::DECIMAL128) {
    scale = target_type.scale();
  }

  std::vector<std::unique_ptr<row_ir::node>> args;
  args.emplace_back(std::move(operand));
  return std::make_unique<row_ir::node>(cast_op, scale, std::move(args));
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::detail::predicate const& expr)
{
  return std::make_unique<row_ir::node>(
    row_ir::opcode::PREDICATE, std::nullopt, expr.get_operand().accept(*this));
}

bool is_nullable(scalar_input const& in) { return in.scalar_column->view().nullable(); }

bool is_nullable(column_input const& in) { return in.column.nullable(); }

std::tuple<std::string, null_aware, output_nullability> ast_converter::generate_code(
  target target_id, ast::expression const& expr, std::string_view function_name)
{
  // add 1 auto-deduced output variable
  [[maybe_unused]] auto output_id = instance_.add_output();

  output_irs_.emplace_back(std::make_unique<row_ir::node>(output_reference{0}, expr.accept(*this)));

  bool has_nullable_inputs =
    std::any_of(instance_.inputs_.begin(), instance_.inputs_.end(), [&](auto& in) {
      return std::visit([](auto& c) { return is_nullable(c); }, in);
    });

  bool is_null_aware = std::any_of(
    output_irs_.cbegin(), output_irs_.cend(), [](auto& ir) { return ir->is_null_aware(); });

  bool output_is_always_valid = std::all_of(
    output_irs_.cbegin(), output_irs_.cend(), [](auto& ir) { return ir->is_always_valid(); });

  bool may_evaluate_null = output_is_always_valid ? false : (has_nullable_inputs || is_null_aware);

  auto null_policy =
    may_evaluate_null ? output_nullability::PRESERVE : output_nullability::ALL_VALID;

  instance_.set_has_nulls(is_null_aware);

  // instantiate the IR nodes
  for (auto& ir : output_irs_) {
    ir->instantiate(instance_);
  }

  target_info target{target_id};

  CUDF_EXPECTS(
    target.id == target::CUDA, "Unsupported target for code generation", std::invalid_argument);

  auto output_decl = [&](auto i) {
    auto& var = instance_.output_vars_[i];
    auto& ir  = output_irs_[i];
    return std::format("{}* {}", to_cuda_type(ir->get_type(), instance_.has_nulls()), var.id);
  };

  auto input_decl = [&](auto i) {
    auto& var = instance_.input_vars_[i];
    return std::format("{} {}", to_cuda_type(var.type, instance_.has_nulls()), var.id);
  };

  std::vector<std::string> arg_decls;

  for (size_t i = 0; i < instance_.output_vars_.size(); ++i) {
    arg_decls.emplace_back(output_decl(i));
  }

  for (size_t i = 0; i < instance_.input_vars_.size(); ++i) {
    arg_decls.emplace_back(input_decl(i));
  }

  auto args_decl = [&] {
    if (arg_decls.empty()) {
      return std::string{};
    } else if (arg_decls.size() == 1) {
      return arg_decls[0];
    } else {
      return std::accumulate(
        arg_decls.begin() + 1, arg_decls.end(), arg_decls[0], [](auto const& a, auto const& b) {
          return std::format("{}, {}", a, b);
        });
    }
  }();

  code_sink sink;
  sink.emit(std::format("__device__ void {}(", function_name));
  sink.emit(args_decl);
  sink.emit(")\n{\n");
  for (auto& ir : output_irs_) {
    ir->emit_code(instance_, target, sink);
  }
  sink.emit("return;\n}");
  return {sink.get_code(), is_null_aware ? null_aware::YES : null_aware::NO, null_policy};
}

std::variant<column_view, scalar_column_view> get_column_view(scalar_input const& in)
{
  return scalar_column_view{in.scalar_column->view()};
}

std::variant<column_view, scalar_column_view> get_column_view(column_input const& in)
{
  return column_view{in.column};
}

// Due to the AST expression tree structure, we can't generate the IR without the target
// tables
transform_args ast_converter::compute_column(target target_id,
                                             ast::expression const& expr,
                                             table_view const& left_table,
                                             table_view const& right_table,
                                             std::string_view function_name,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  ast_converter converter{stream, mr, left_table, right_table};

  // TODO(lamarrr): consider deduplicating ast expression's input column references. See
  // TransformTest/1.DeeplyNestedArithmeticLogicalExpression for reference

  auto [code, is_null_aware, output_nullability] =
    converter.generate_code(target_id, expr, function_name);
  std::vector<std::variant<column_view, scalar_column_view>> inputs;
  std::vector<std::unique_ptr<column>> scalar_columns;
  std::vector<std::optional<int32_t>> table_sources;
  std::vector<std::optional<int32_t>> column_indices;

  for (auto& input : converter.instance_.inputs_) {
    if (std::holds_alternative<column_input>(input)) {
      auto& col = std::get<column_input>(input);
      table_sources.emplace_back(col.table_source);
      column_indices.emplace_back(col.column_index);
    } else {
      table_sources.emplace_back(std::nullopt);
      column_indices.emplace_back(std::nullopt);
    }

    auto view = std::visit([](auto& in) { return get_column_view(in); }, input);
    inputs.emplace_back(view);

    if (std::holds_alternative<scalar_input>(input)) {
      auto& scalar = std::get<scalar_input>(input);
      scalar_columns.emplace_back(std::move(scalar.scalar_column));
    }
  }

  auto& out               = converter.output_irs_[0];
  auto output_column_type = out->get_type();
  auto output   = transform_output{.type = output_column_type, .nullability = output_nullability};
  auto row_size = std::max({left_table.num_rows(), right_table.num_rows()});
  auto result   = transform_args{.scalar_columns       = std::move(scalar_columns),
                                 .input_table_sources  = std::move(table_sources),
                                 .input_column_indices = std::move(column_indices),
                                 .udf                  = std::move(code),
                                 .source_type          = cudf::udf_source_type::CUDA,
                                 .is_null_aware        = is_null_aware,
                                 .user_data            = std::nullopt,
                                 .inputs               = inputs,
                                 .outputs{output},
                                 .string_offsets{},
                                 .row_size = row_size};
  if (get_context().dump_codegen()) {
    std::cout << "Generated code for transform: \n" << result.udf << std::endl;
  }

  return result;
}

transform_args ast_converter::filter(target target_id,
                                     ast::expression const& expr,
                                     table_view const& left_table,
                                     table_view const& right_table,
                                     std::string_view function_name,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  auto filter = ast::detail::predicate{expr};
  return compute_column(target_id, filter, left_table, right_table, function_name, stream, mr);
}

}  // namespace cudf::detail::row_ir
