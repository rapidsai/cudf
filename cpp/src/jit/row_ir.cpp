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
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>

namespace cudf::detail::row_ir {

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

  data_type type{type_id::EMPTY, 0};
  if (auto* col = std::get_if<column_input>(&in)) {
    type = col->column.type();
  } else {
    auto& scalar = std::get<scalar_input>(in);
    type         = scalar.scalar_column->type();
  }
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
  CUDF_EXPECTS(op != opcode::GET_INPUT && op != opcode::SET_OUTPUT,
               std::format("Invalid opcode `{}` for operation node.", get_op_name(op)));
  CUDF_EXPECTS(args_.size() == static_cast<size_t>(get_op_arity(op)),
               std::format("Invalid number of arguments for operator `{}`. Expected {}, Got {}.",
                           get_op_name(op),
                           get_op_arity(op),
                           args_.size()));
  CUDF_EXPECTS(target_scale_.has_value() == (op == opcode::RESCALE),
               std::format("Target scale must be provided for RESCALE operator and must be nullopt "
                           "for other operators."));
}

node::node(input_reference input) : reference_{input}, op_{opcode::GET_INPUT} {}

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

bool node::is_null_aware() const
{
  return get_op_null_output(op_) ==
           null_output::ALWAYS_NULLABLE ||  // to emit nulls for always-nullable operators, we need
                                            // to mark them as null-aware
         get_op_requires_nulls(op_) ||
         std::any_of(args_.begin(), args_.end(), [](auto& a) { return a->is_null_aware(); });
}

bool node::is_always_valid() const
{
  return get_op_null_output(op_) == null_output::ALWAYS_VALID ||
         std::all_of(args_.begin(), args_.end(), [](auto& a) { return a->is_always_valid(); });
}

bool node::is_fallible() const
{
  return get_op_is_fallible(op_) ||
         std::any_of(args_.begin(), args_.end(), [](auto& a) { return a->is_fallible(); });
}

row_ir::type as_typing(data_type type)
{
  switch (type.id()) {
    case type_id::BOOL8: return type::BOOL8;
    case type_id::INT8: return type::INT8;
    case type_id::INT16: return type::INT16;
    case type_id::INT32: return type::INT32;
    case type_id::INT64: return type::INT64;
    case type_id::UINT8: return type::UINT8;
    case type_id::UINT16: return type::UINT16;
    case type_id::UINT32: return type::UINT32;
    case type_id::UINT64: return type::UINT64;
    case type_id::FLOAT32: return type::FLOAT32;
    case type_id::FLOAT64: return type::FLOAT64;
    case type_id::DECIMAL32: return type::DECIMAL32;
    case type_id::DECIMAL64: return type::DECIMAL64;
    case type_id::DECIMAL128: return type::DECIMAL128;
    case type_id::TIMESTAMP_DAYS: return type::TIMESTAMP_DAYS;
    case type_id::TIMESTAMP_SECONDS: return type::TIMESTAMP_SECONDS;
    case type_id::TIMESTAMP_MILLISECONDS: return type::TIMESTAMP_MILLISECONDS;
    case type_id::TIMESTAMP_MICROSECONDS: return type::TIMESTAMP_MICROSECONDS;
    case type_id::TIMESTAMP_NANOSECONDS: return type::TIMESTAMP_NANOSECONDS;
    case type_id::DURATION_DAYS: return type::DURATION_DAYS;
    case type_id::DURATION_SECONDS: return type::DURATION_SECONDS;
    case type_id::DURATION_MILLISECONDS: return type::DURATION_MILLISECONDS;
    case type_id::DURATION_MICROSECONDS: return type::DURATION_MICROSECONDS;
    case type_id::DURATION_NANOSECONDS: return type::DURATION_NANOSECONDS;
    case type_id::STRING: return type::STRING;
    default:
      CUDF_FAIL(std::format("Unsupported data type for Row IR: {}", type_to_name(type)),
                std::invalid_argument);
  }
}

type_id as_type_id(type type)
{
  switch (type) {
    case type::BOOL8: return type_id::BOOL8;
    case type::INT8: return type_id::INT8;
    case type::INT16: return type_id::INT16;
    case type::INT32: return type_id::INT32;
    case type::INT64: return type_id::INT64;
    case type::UINT8: return type_id::UINT8;
    case type::UINT16: return type_id::UINT16;
    case type::UINT32: return type_id::UINT32;
    case type::UINT64: return type_id::UINT64;
    case type::FLOAT32: return type_id::FLOAT32;
    case type::FLOAT64: return type_id::FLOAT64;
    case type::DECIMAL32: return type_id::DECIMAL32;
    case type::DECIMAL64: return type_id::DECIMAL64;
    case type::DECIMAL128: return type_id::DECIMAL128;
    case type::TIMESTAMP_DAYS: return type_id::TIMESTAMP_DAYS;
    case type::TIMESTAMP_SECONDS: return type_id::TIMESTAMP_SECONDS;
    case type::TIMESTAMP_MILLISECONDS: return type_id::TIMESTAMP_MILLISECONDS;
    case type::TIMESTAMP_MICROSECONDS: return type_id::TIMESTAMP_MICROSECONDS;
    case type::TIMESTAMP_NANOSECONDS: return type_id::TIMESTAMP_NANOSECONDS;
    case type::DURATION_DAYS: return type_id::DURATION_DAYS;
    case type::DURATION_SECONDS: return type_id::DURATION_SECONDS;
    case type::DURATION_MILLISECONDS: return type_id::DURATION_MILLISECONDS;
    case type::DURATION_MICROSECONDS: return type_id::DURATION_MICROSECONDS;
    case type::DURATION_NANOSECONDS: return type_id::DURATION_NANOSECONDS;
    case type::STRING: return type_id::STRING;
    default:
      CUDF_FAIL(std::format("Invalid typing for {}: {}", __FUNCTION__, static_cast<int>(type)),
                std::invalid_argument);
  }
}

opcode as_opcode(ast::ast_operator op)
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
    case ast::ast_operator::BITWISE_AND: return opcode::BIT_AND;
    case ast::ast_operator::BITWISE_OR: return opcode::BIT_OR;
    case ast::ast_operator::BITWISE_XOR: return opcode::BIT_XOR;
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
    case ast::ast_operator::NOT: return opcode::LOGICAL_NOT;
    case ast::ast_operator::CAST_TO_INT64: return opcode::CAST_TO_I64;
    case ast::ast_operator::CAST_TO_UINT64: return opcode::CAST_TO_U64;
    case ast::ast_operator::CAST_TO_FLOAT64: return opcode::CAST_TO_F64;
    default: CUDF_UNREACHABLE("Invalid opcode");
  }
}

std::string to_cuda_type(cudf::data_type type, bool nullable)
{
  auto name = type_to_name(type);
  return nullable ? std::format("cuda::std::optional<{}>", name) : name;
}

data_type get_return_type(opcode op,
                          std::span<data_type const> args,
                          std::optional<int32_t> target_scale)
{
  std::vector<row_ir::type> arg_types;
  std::vector<int32_t> arg_scales;

  for (auto& type : args) {
    arg_types.emplace_back(as_typing(type));
    arg_scales.emplace_back(type.scale());
  }

  auto op_type_match = get_op_typing(op);
  auto rescaled      = op_rescale(op, arg_scales, target_scale);

  for (size_t i = 0; i < args.size(); ++i) {
    auto required_type = op_type_match.args[i];
    auto arg_type      = arg_types[i];

    if ((required_type & type::ARG_MASK) != type::NONE) {
      auto src_index = static_cast<size_t>(required_type & ~type::ARG_MASK);
      CUDF_EXPECTS(
        src_index < i,
        std::format(
          "Invalid type match rule for operator `{}` at argument #{}", get_op_name(op), i),
        std::runtime_error);
      CUDF_EXPECTS(args[i].id() == args[src_index].id(),
                   std::format("Argument #{} of operator `{}` does not match type of argument "
                               "#{}. Got `{}`, expected `{}`",
                               i,
                               get_op_name(op),
                               src_index,
                               type_to_name(args[i]),
                               type_to_name(args[src_index])));
    } else {
      CUDF_EXPECTS(
        (arg_type & required_type) != 0,
        std::format("Argument #{} of operator `{}` does not match expected types. Got {}",
                    i,
                    get_op_name(op),
                    type_to_name(args[i])));
    }
  }

  if ((op_type_match.output & type::ARG_MASK) != type::NONE) {
    auto arg_index = static_cast<size_t>(op_type_match.output & ~type::ARG_MASK);
    auto type      = args[arg_index].id();
    auto scale     = numeric::scale_type{is_fixed_point(data_type{type}) ? rescaled : 0};
    return data_type{type, scale};
  } else {
    CUDF_EXPECTS(
      op_type_match.output != type::NONE && (op_type_match.output & type::DECIMALS) == type::NONE,
      std::format("Invalid type match rule for operator `{}` return type", get_op_name(op)),
      std::runtime_error);
    auto type  = as_type_id(op_type_match.output);
    auto scale = numeric::scale_type{is_fixed_point(data_type{type}) ? rescaled : 0};
    return data_type{type, scale};
  }
}

void node::instantiate(instance_context& ctx)
{
  id_ = ctx.make_tmp_id();

  for (auto& arg : args_) {
    arg->instantiate(ctx);
  }

  switch (op_) {
    case opcode::GET_INPUT: {
      type_ = ctx.get_input_vars()[std::get<input_reference>(reference_).index].type;
    } break;
    case opcode::SET_OUTPUT: {
      type_ = args_[0]->get_type();
    } break;
    default: {
      std::vector<data_type> arg_types;
      for (auto& arg : args_) {
        arg_types.emplace_back(arg->get_type());
      }

      if (op_ == opcode::RESCALE) {
        scale_reference_ =
          input_reference{ctx.add_input(cudf::numeric_scalar<int32_t>{target_scale_.value_or(0)})};
      }

      type_ = get_return_type(op_, arg_types, target_scale_);
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

        default: {
          auto first_arg = std::format("&{}", args_[0]->get_id());
          auto args_str  = (args_.size() == 1)
                             ? std::string{first_arg}
                             : std::accumulate(args_.begin() + 1,
                                              args_.end(),
                                              std::string{first_arg},
                                              [](auto const& a, auto& node) {
                                                return std::format("{}, &{}", a, node->get_id());
                                              });

          if (op_ == opcode::RESCALE) {
            args_str = std::format(
              "{}, &{}", args_str, instance.get_input_vars()[scale_reference_.index].id);
          }

          bool fallible = get_op_is_fallible(op_);
          auto op_name  = get_op_name(op_);

          if (!fallible) {
            sink.emit(std::format(
              R"***({} {};
cudf::ops::{}(&{}, {});
)***",
              type,
              id_,
              op_name,
              id_,
              args_str));
          } else {
            sink.emit(std::format(
              R"***({} {};
if(cudf::ops::errc e = cudf::ops::{}(&{}, {}); e != cudf::ops::errc::OK) {{
return e;
}}
)***",
              type,
              id_,
              op_name,
              id_,
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
                 "Invalid table reference in column expression");
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

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::detail::predicate const& expr)
{
  return std::make_unique<row_ir::node>(
    row_ir::opcode::PREDICATE, std::nullopt, expr.get_operand().accept(*this));
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::jit::detail::operation const& expr)
{
  std::vector<std::unique_ptr<row_ir::node>> args;
  for (auto &arg : expr.get_arguments()) {
    args.emplace_back(arg.get().accept(*this));
  }
  return std::make_unique<row_ir::node>(
    expr.get_opcode(), expr.get_target_scale(), std::move(args));
}

bool is_nullable(scalar_input const& in) { return in.scalar_column->view().nullable(); }

bool is_nullable(column_input const& in) { return in.column.nullable(); }

std::tuple<std::string, null_aware, output_nullability, bool> ast_converter::generate_code(
  target target_id, ast::expression const& expr, std::string_view function_name)
{
  // add 1 auto-deduced output variable
  [[maybe_unused]] auto output_id = instance_.add_output();

  output_irs_.emplace_back(std::make_unique<row_ir::node>(output_reference{0}, expr.accept(*this)));

  bool has_nullable_inputs =
    std::any_of(instance_.inputs_.begin(), instance_.inputs_.end(), [&](auto& in) {
      return std::visit([](auto& c) { return is_nullable(c); }, in);
    });

  auto is_null_aware =
    std::any_of(
      output_irs_.cbegin(), output_irs_.cend(), [](auto& ir) { return ir->is_null_aware(); })
      ? null_aware::YES
      : null_aware::NO;

  bool output_is_always_valid = std::all_of(
    output_irs_.cbegin(), output_irs_.cend(), [](auto& ir) { return ir->is_always_valid(); });

  bool may_evaluate_null = !output_is_always_valid || has_nullable_inputs;
  auto null_policy =
    may_evaluate_null ? output_nullability::PRESERVE : output_nullability::ALL_VALID;

  auto is_fallible = std::any_of(
    output_irs_.cbegin(), output_irs_.cend(), [](auto& ir) { return ir->is_fallible(); });

  instance_.set_has_nulls(is_null_aware == null_aware::YES);

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
  sink.emit(std::format("__device__ cudf::ops::errc {}(", function_name));
  sink.emit(args_decl);
  sink.emit(")\n{\n");
  for (auto& ir : output_irs_) {
    ir->emit_code(instance_, target, sink);
  }
  sink.emit("return cudf::ops::errc::OK;\n}");
  return {sink.get_code(), is_null_aware, null_policy, is_fallible};
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

  auto [code, is_null_aware, output_nullability, is_fallible] =
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
  auto result =
    transform_args{.scalar_columns       = std::move(scalar_columns),
                   .input_table_sources  = std::move(table_sources),
                   .input_column_indices = std::move(column_indices),
                   .udf                  = std::move(code),
                   .source_type          = cudf::udf_source_type::CUDA,
                   .is_null_aware        = is_null_aware,
                   .user_data            = std::nullopt,
                   .inputs               = inputs,
                   .outputs{output},
                   .string_offsets{},
                   .row_size   = row_size,
                   .error_mode = is_fallible ? ops::error_mode::ANY_ROW : ops::error_mode::IGNORE};
  if (get_context().dump_codegen()) {
    std::cout << "Generated code for transform: " << result.udf << std::endl;
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
  auto transform =
    compute_column(target_id, filter, left_table, right_table, function_name, stream, mr);

  CUDF_EXPECTS(transform.outputs.size() == 1,
               "Filter expression must have exactly one output column.");
  CUDF_EXPECTS(transform.outputs[0].type.id() == type_id::BOOL8,
               "Filter expression must return a boolean type.",
               std::invalid_argument);

  return transform;
}

}  // namespace cudf::detail::row_ir
