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

get_input::get_input(uint32_t input, bool null_aware)
  : id_(), input_(input), type_(type_id::EMPTY), null_aware_(null_aware)
{
}

std::string_view get_input::get_id() { return id_; }

type_id get_input::get_type() { return type_; }

bool get_input::get_null_aware() { return null_aware_; }

void get_input::instantiate(context& ctx, instance_info const& info)
{
  id_               = ctx.make_tmp_id();
  auto const& input = info.inputs[input_];
  type_             = input.type;
  null_aware_       = input.nullable;
}

std::string get_input::generate_code(context& ctx, target_info const& info)
{
  switch (info.target) {
    case target::CUDA: {
      return std::format("auto {} = {};", id_, info.instance.inputs[input_].id);
    }
    default: CUDF_FAIL("Unsupported target");
  }
}

set_output::set_output(uint32_t output, std::unique_ptr<node> source)
  : id_(),
    output_(output),
    source_(std::move(source)),
    type_(type_id::EMPTY),
    null_aware_(false),
    output_id_()
{
}

std::string_view set_output::get_id() { return id_; }

type_id set_output::get_type() { return type_; }

bool set_output::get_null_aware() { return null_aware_; }

void set_output::instantiate(context& ctx, instance_info const& info)
{
  id_         = ctx.make_tmp_id();
  type_       = source_->get_type();
  null_aware_ = source_->get_null_aware();
  output_id_  = info.outputs[output_].id;
  CUDF_EXPECTS(source_->get_type() == info.outputs[output_].type,
               "Output type does not match source type.");
}

std::string set_output::generate_code(context& ctx, target_info const& info)
{
  switch (info.target) {
    case target::CUDA: {
      return std::format("{} = {};", output_id_, source_->get_id());
    }
    default: CUDF_FAIL("Unsupported target");
  }
}

operation::operation(opcode op, std::vector<std::unique_ptr<node>> operands)
  : id_(), op_(op), operands_(std::move(operands)), type_(type_id::EMPTY), null_aware_(false)
{
  CUDF_EXPECTS(operands.size() == cudf::ast::detail::ast_operator_arity(op),
               "Invalid number of arguments for operator.");
  CUDF_EXPECTS(operands.size() > 0, "Operator must have at least one operand");
}

std::string_view operation::get_id() { return id_; }

type_id operation::get_type() { return type_; }

bool operation::get_null_aware() { return null_aware_; }

void operation::instantiate(context& ctx, instance_info const& info)
{
  for (auto& arg : operands_) {
    arg->instantiate(ctx, info);
  }

  id_ = ctx.make_tmp_id();
  std::vector<cudf::data_type> operand_types;
  auto null_aware = operands_[0]->get_null_aware();
  for (auto& arg : operands_) {
    operand_types.emplace_back(arg->get_type());
    CUDF_EXPECTS(arg->get_null_aware() == null_aware,
                 "All operands of an operator must have the same nullability.");
  }

  type_       = cudf::ast::detail::ast_operator_return_type(op_, operand_types).id();
  null_aware_ = null_aware;
}

std::string operation::generate_code(context& ctx, target_info const& info)
{
  std::string operands_code;

  for (auto& arg : operands_) {
    operands_code += arg->generate_code(ctx, info) + "\n";
  }

  auto operation_code = [&]() {
    switch (info.target) {
      case target::CUDA: {
        auto operands_str =
          std::format("{}, {}",
                      operands_[0]->get_id(),
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
                      null_aware_,
                      operands_str);
        return cuda;
      }
      default: CUDF_FAIL("Unsupported target");
    }
  }();

  return operands_code + operation_code;
}

}  // namespace row_ir
}  // namespace cudf