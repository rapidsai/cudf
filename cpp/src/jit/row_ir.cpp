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

namespace cudf::detail::row_ir {

std::string instance_context::make_tmp_id()
{
  return std::format("{}{}", tmp_prefix_, num_tmp_vars_++);
}

bool instance_context::has_nulls() const { return has_nulls_; }

void instance_context::set_has_nulls(bool has_nulls) { has_nulls_ = has_nulls; }

node::node(opcode op, std::vector<node> args) : op_{op}, args_{std::move(args)}
{
  CUDF_EXPECTS(op != opcode::GET_INPUT && op != opcode::SET_OUTPUT,
               std::format("Invalid opcode `{}` for operation node.", get_op_name(op)));
  CUDF_EXPECTS(args_.size() == get_op_arity(op),
               std::format("Invalid number of arguments for operator `{}`. Expected {}, Got {}.",
                           get_op_name(op),
                           get_op_arity(op),
                           args_.size()));
  // TODO: check argument types, this will be after resolving types
}

node::node(input_reference input) : reference_{input}, op_{opcode::SET_OUTPUT} {}

node::node(output_reference reference, node arg)
  : reference_{reference}, op_{opcode::SET_OUTPUT}, args_{std::move(arg)}
{
}

std::string_view node::get_id() const { return id_; }

data_type node::get_type() const { return type_; }

opcode node::get_opcode() const { return op_; }

std::span<node const> node::get_args() const { return args_; }

bool node::is_null_aware() const
{
  return get_op_null_output(op_) ==
           null_output::ALWAYS_NULLABLE ||  // to emit nulls for always-nullable operators, we need
                                            // to mark them as null-aware
         get_op_requires_nulls(op_) ||
         std::any_of(args_.begin(), args_.end(), [](auto& a) { return a.is_null_aware(); });
}

bool node::is_always_valid() const
{
  return get_op_null_output(op_) == null_output::ALWAYS_VALID ||
         std::all_of(args_.begin(), args_.end(), [](auto& a) { return a.is_always_valid(); });
}

bool node::is_fallible() const
{
  return get_op_is_fallible(op_) ||
         std::any_of(args_.begin(), args_.end(), [](auto& a) { return a.is_fallible(); });
}

row_ir::typing as_typing(data_type type)
{
  switch (type.id()) {
    case type_id::BOOL8: return typing::BOOL8;
    case type_id::INT8: return typing::INT8;
    case type_id::INT16: return typing::INT16;
    case type_id::INT32: return typing::INT32;
    case type_id::INT64: return typing::INT64;
    case type_id::UINT8: return typing::UINT8;
    case type_id::UINT16: return typing::UINT16;
    case type_id::UINT32: return typing::UINT32;
    case type_id::UINT64: return typing::UINT64;
    case type_id::FLOAT32: return typing::FLOAT32;
    case type_id::FLOAT64: return typing::FLOAT64;
    case type_id::DECIMAL32: return typing::DECIMAL32;
    case type_id::DECIMAL64: return typing::DECIMAL64;
    case type_id::DECIMAL128: return typing::DECIMAL128;
    case type_id::TIMESTAMP_DAYS: return typing::TIMESTAMP_DAYS;
    case type_id::TIMESTAMP_SECONDS: return typing::TIMESTAMP_SECONDS;
    case type_id::TIMESTAMP_MILLISECONDS: return typing::TIMESTAMP_MILLISECONDS;
    case type_id::TIMESTAMP_MICROSECONDS: return typing::TIMESTAMP_MICROSECONDS;
    case type_id::TIMESTAMP_NANOSECONDS: return typing::TIMESTAMP_NANOSECONDS;
    case type_id::DURATION_DAYS: return typing::DURATION_DAYS;
    case type_id::DURATION_SECONDS: return typing::DURATION_SECONDS;
    case type_id::DURATION_MILLISECONDS: return typing::DURATION_MILLISECONDS;
    case type_id::DURATION_MICROSECONDS: return typing::DURATION_MICROSECONDS;
    case type_id::DURATION_NANOSECONDS: return typing::DURATION_NANOSECONDS;
    case type_id::STRING: return typing::STRING;
    default:
      CUDF_FAIL(std::format("Unsupported data type for Row IR: {}", type_to_name(type)),
                std::invalid_argument);
  }
}

data_type get_return_type(opcode op, std::span<data_type const> args)
{
  std::vector<row_ir::typing> typings;

  for (auto& type : args) {
    typings.push_back(as_typing(type));
  }

  auto op_type_match = get_op_typing(op);
  // TODO: match typing and get return type
  //
  //
  // TODO(lamarrr): figure out scale propagation rules and creation/assignment rules
  //
  // TODO(lamarrr): implement filter_predicate to return false on nulls
  //
  // TODO: scale-propagation rules
  // TODO: decimal ansi operators(precision and scale-oriented non-templated arguments)
  // TODO: cast operators to match AST
  // TODO: decimal cast operators to match AST
  // TODO: datetime cast operators & arithmetic
  // TODO: decimal ansi cast
  // TODO: ansi_mod, div operations for fixed-point and duration types

  data_type return_type;
}

void node::instantiate(instance_context& ctx, instance_info const& info)
{
  id_ = ctx.make_tmp_id();

  for (auto& arg : args_) {
    arg.instantiate(ctx, info);
  }

  switch (op_) {
    case opcode::GET_INPUT: {
      type_ = info.inputs[std::get<input_reference>(reference_).index].type;
    } break;
    case opcode::SET_OUTPUT: {
      type_ = args_[0].get_type();
    } break;
    default: {
      std::vector<data_type> arg_types;
      for (auto& arg : args_) {
        arg_types.emplace_back(arg.get_type());
      }
      type_ = get_return_type(op_, arg_types);
    } break;
  }
}

void node::emit_code(instance_context& ctx,
                     target_info const& info,
                     instance_info const& instance,
                     code_sink& sink) const
{
  auto to_cuda_type = [](cudf::data_type type, bool nullable) {
    auto name = type_to_name(type);
    return nullable ? std::format("cuda::std::optional<{}>", name) : name;
  };

  for (auto& arg : args_) {
    arg.emit_code(ctx, info, instance, sink);
  }

  switch (info.id) {
    case target::CUDA: {
      auto type = to_cuda_type(type_, ctx.has_nulls());

      switch (op_) {
        case opcode::GET_INPUT: {
          sink.emit(std::format("{} {} = {};",
                                type,
                                id_,
                                instance.inputs[std::get<input_reference>(reference_).index].id));
        } break;

        case opcode::SET_OUTPUT: {
          sink.emit(std::format(
            R"**({} {} = {};
*{} = {};
)**",
            type,
            id_,
            args_[0].get_id(),
            instance.outputs[std::get<output_reference>(reference_).index].id,
            id_));
        } break;

        default: {
          auto first_arg = args_[0].get_id();
          auto args_str  = (args_.size() == 1)
                             ? std::string{first_arg}
                             : std::accumulate(args_.begin() + 1,
                                              args_.end(),
                                              std::string{first_arg},
                                              [](auto const& a, auto& node) {
                                                return std::format("{}, &{}", a, node.get_id());
                                              });

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
if(cudf::ops::errc e = cudf::ops::{}(&{}, {}); e != cudf::ops::errc::SUCCESS) {{
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

/*
// TODO: transitive null-ness

filter_predicate::filter_predicate(std::unique_ptr<node> source) : id_(), source_(std::move(source))
{
}

[[nodiscard]] std::string filter_predicate::generate_code(instance_context& ctx,
                                                          target_info const& info,
                                                          instance_info const& instance)
{
  switch (info.id) {
    case target::CUDA: {
      auto source_code = source_->generate_code(ctx, info, instance);
      return std::format(
        "{}\n"
        "bool {} = cudf::ast::detail::flatten_predicate({});\n",
        source_code,
        id_,
        source_->get_id());
    }
    default:
      CUDF_FAIL("Unsupported target: " + std::to_string(static_cast<int>(info.id)),
                std::invalid_argument);
  }
}
  */

std::span<ast_input_spec const> ast_converter::get_input_specs() const { return input_specs_; }

int32_t ast_converter::add_ast_input(ast_input_spec in)
{
  auto id = static_cast<int32_t>(input_specs_.size());
  input_specs_.push_back(std::move(in));
  return id;
}

row_ir::node ast_converter::add_ir_node(ast::literal const& expr)
{
  auto index = add_ast_input(
    ast_scalar_input_spec{expr.get_scalar(),
                          expr.get_value(),
                          make_column_from_scalar(expr.get_scalar(), 1, stream_, mr_)});
  return row_ir::node(input_reference{index});
}

row_ir::node ast_converter::add_ir_node(ast::column_reference const& expr)
{
  auto index =
    add_ast_input(ast_column_input_spec{expr.get_table_source(), expr.get_column_index()});
  return row_ir::node(input_reference{index});
}

row_ir::node ast_converter::add_ir_node(ast::operation const& expr)
{
  std::vector<row_ir::node> operands;
  for (auto const& operand : expr.get_operands()) {
    operands.push_back(operand.get().accept(*this));
  }
  return row_ir::node(row_ir::operation{expr.get_operator(), std::move(operands)});
}

row_ir::node ast_converter::add_ir_node(ast::detail::filter_predicate const& expr)
{
  auto operand = expr.get_operand().accept(*this);
  return row_ir::node(row_ir::filter_predicate{std::move(operand)});
}

// Resolve the table for a column input spec, preferring left_table/right_table for join cases,
// falling back to args.table for the single-table case.
table_view const& resolve_table(ast_column_input_spec const& in, ast_args const& args)
{
  if (in.table == ast::table_reference::LEFT) {
    return args.left_table.num_columns() > 0 ? args.left_table : args.table;
  }
  return args.right_table;
}

void ast_converter::add_input_var(ast_column_input_spec const& in, ast_args const& args)
{
  auto id   = std::format("in_{}", input_vars_.size());
  auto type = resolve_table(in, args).column(in.column).type();
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

std::variant<column_view, scalar_column_view> get_column_view(ast_column_input_spec const& spec,
                                                              ast_args const& args)
{
  return resolve_table(spec, args).column(spec.column);
}

std::variant<column_view, scalar_column_view> get_column_view(ast_scalar_input_spec const& spec,
                                                              ast_args const& args)
{
  return scalar_column_view{spec.broadcast_column->view()};
}

std::tuple<null_aware, output_nullability> ast_converter::generate_code(target target_id,
                                                                        ast::expression const& expr,
                                                                        ast_args const& args)
{
  output_irs_.emplace_back(std::make_unique<row_ir::set_output>(0, expr.accept(*this)));

  // resolve the flattened input references into IR input variables
  for (auto const& input : input_specs_) {
    dispatch_input_spec(input, [this](auto&... args) { add_input_var(args...); }, args);
  }

  bool has_nullable_inputs =
    std::any_of(input_specs_.begin(), input_specs_.end(), [&](auto const& input) {
      return dispatch_input_spec(
        input,
        [](auto&... args) {
          auto col = get_column_view(args...);
          return std::visit([](auto& view) { return view.nullable(); }, col);
        },
        args);
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
__device__ cudf::ops::errc expression({})
{{
{}
return cudf::ops::errc::SUCCESS;
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

  std::vector<std::variant<column_view, scalar_column_view>> inputs;
  std::vector<std::unique_ptr<column>> scalar_columns;

  for (auto& input : converter.input_specs_) {
    auto column_view =
      dispatch_input_spec(input, [](auto&... args) { return get_column_view(args...); }, args);
    inputs.emplace_back(column_view);

    if (std::holds_alternative<ast_scalar_input_spec>(input)) {
      auto& scalar_input = std::get<ast_scalar_input_spec>(input);
      scalar_columns.push_back(std::move(scalar_input.broadcast_column));
    }
  }

  auto& out               = converter.output_irs_[0];
  auto output_column_type = out->get_type();

  auto result = transform_args{.scalar_columns = std::move(scalar_columns),
                               .inputs         = inputs,
                               .udf            = std::move(converter.code_),
                               .output_type    = output_column_type,
                               .source_type    = cudf::udf_source_type::CUDA,
                               .user_data      = std::nullopt,
                               .is_null_aware  = is_null_aware,
                               .null_policy    = output_nullability,
                               .row_size       = args.table.num_rows(),
                               .input_specs    = std::move(converter.input_specs_)};

  if (get_context().dump_codegen()) {
    std::cout << "Generated code for transform: " << result.udf << std::endl;
  }

  return result;
}

filter_args ast_converter::filter(target target_id,
                                  ast::expression const& expr,
                                  ast_args const& args,
                                  table_view const& filter_table,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  auto filter    = ast::detail::filter_predicate{expr};
  auto transform = compute_column(target_id, filter, args, stream, mr);

  CUDF_EXPECTS(transform.output_type.id() == type_id::BOOL8,
               "Filter expression must return a boolean type.",
               std::invalid_argument);

  std::vector<column_view> filter_columns;
  std::transform(filter_table.begin(),
                 filter_table.end(),
                 std::back_inserter(filter_columns),
                 [](auto const& col) { return col; });

  auto result = filter_args{.scalar_columns        = std::move(transform.scalar_columns),
                            .inputs                = std::move(transform.inputs),
                            .filter_columns        = std::move(filter_columns),
                            .udf                   = std::move(transform.udf),
                            .source_type           = transform.source_type,
                            .user_data             = transform.user_data,
                            .is_null_aware         = transform.is_null_aware,
                            .predicate_nullability = transform.null_policy,
                            .input_specs           = std::move(transform.input_specs)};

  return result;
}

}  // namespace cudf::detail::row_ir
