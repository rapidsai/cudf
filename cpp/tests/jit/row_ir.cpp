/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir.hpp"

#include "cudf_test/column_wrapper.hpp"

#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/transform.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <array>
#include <cctype>
#include <functional>

namespace row_ir = cudf::detail::row_ir;

struct RowIRCodeGenTest : public ::testing::Test {
  std::unique_ptr<cudf::column> f32 =
    cudf::test::fixed_width_column_wrapper<float>({1.0f, 2.0f, 3.0f}).release();
  std::unique_ptr<cudf::column> f64 =
    cudf::test::fixed_width_column_wrapper<double>({1.0, 2.0, 3.0}).release();
  std::unique_ptr<cudf::column> d32 =
    cudf::test::fixed_point_column_wrapper<int32_t>({1, 2, 3}, numeric::scale_type{2}).release();
  std::unique_ptr<cudf::column> i32 =
    cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}).release();
  std::unique_ptr<cudf::column> b8 =
    cudf::test::fixed_width_column_wrapper<bool>({true, false, true}).release();
  cudf::table_view table = cudf::table_view({*f32, *f64, *d32, *i32});
};

TEST_F(RowIRCodeGenTest, GetInput)
{
  row_ir::target_info target_info{row_ir::target::TRANSFORM};

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    row_ir::code_sink sink;
    row_ir::node get_input_0{row_ir::input_reference{0}};
    get_input_0.instantiate(ctx);
    get_input_0.emit_code(ctx, target_info, sink);

    auto expected_code = R"ABI(optional<i32> tmp_0 = in_0;
)ABI";

    EXPECT_EQ(sink.get_code(), expected_code);
  }

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    row_ir::code_sink sink;
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*f32);
    row_ir::node get_input_1{row_ir::input_reference{1}};
    get_input_1.instantiate(ctx);
    get_input_1.emit_code(ctx, target_info, sink);

    auto expected_null_code = R"ABI(optional<f32> tmp_0 = in_1;
)ABI";

    EXPECT_EQ(sink.get_code(), expected_null_code);
  }
}

TEST_F(RowIRCodeGenTest, SetOutput)
{
  row_ir::target_info target_info{row_ir::target::TRANSFORM};

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};

    [[maybe_unused]] auto in0  = ctx.add_input(*i32);
    [[maybe_unused]] auto in1  = ctx.add_input(*f32);
    [[maybe_unused]] auto out0 = ctx.add_output();
    [[maybe_unused]] auto out1 = ctx.add_output();
    row_ir::code_sink sink;
    row_ir::node set_output_0{row_ir::output_reference{0},
                              row_ir::node{row_ir::input_reference{0}}};
    set_output_0.instantiate(ctx);
    set_output_0.emit_code(ctx, target_info, sink);

    auto expected_code = R"ABI(optional<i32> tmp_0 = in_0;
set_validity(outputs, 0, row, tmp_0.validity);
if (tmp_0.validity) {
auto stored_tmp_1 = set_i32(outputs, 0, row, tmp_0.value);
if (stored_tmp_1 != success) { return stored_tmp_1; }
}
)ABI";

    EXPECT_EQ(sink.get_code(), expected_code);
  }

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    row_ir::code_sink sink;
    [[maybe_unused]] auto in0  = ctx.add_input(*i32);
    [[maybe_unused]] auto in1  = ctx.add_input(*f32);
    [[maybe_unused]] auto out0 = ctx.add_output();
    [[maybe_unused]] auto out1 = ctx.add_output();
    row_ir::node set_output_1{row_ir::output_reference{1},
                              row_ir::node{row_ir::input_reference{1}}};
    set_output_1.instantiate(ctx);
    set_output_1.emit_code(ctx, target_info, sink);

    auto expected_code = R"ABI(optional<f32> tmp_0 = in_1;
set_validity(outputs, 1, row, tmp_0.validity);
if (tmp_0.validity) {
auto stored_tmp_1 = set_f32(outputs, 1, row, tmp_0.value);
if (stored_tmp_1 != success) { return stored_tmp_1; }
}
)ABI";

    EXPECT_EQ(sink.get_code(), expected_code);
  }
}

TEST_F(RowIRCodeGenTest, UnaryOperation)
{
  row_ir::target_info target_info{row_ir::target::TRANSFORM};

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*f32);

    row_ir::code_sink sink;
    row_ir::node op{row_ir::opcode::IDENTITY,
                    std::nullopt,
                    cudf::error_policy::PROPAGATE,
                    row_ir::node{row_ir::input_reference{0}}};
    op.instantiate(ctx);
    op.emit_code(ctx, target_info, sink);

    auto expected_code = R"ABI(optional<i32> tmp_0 = in_0;
optional<i32> tmp_1{};
if (tmp_0.validity) {
auto evaluated_tmp_1 = cudf_row_op_IDENTITY_i32(tmp_0.value);
tmp_1 = {evaluated_tmp_1, 1};
}
)ABI";

    EXPECT_EQ(sink.get_code(), expected_code);
  }

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*d32);
    row_ir::code_sink sink;
    row_ir::node op{row_ir::opcode::IDENTITY,
                    std::nullopt,
                    cudf::error_policy::PROPAGATE,
                    row_ir::node{row_ir::input_reference{1}}};
    op.instantiate(ctx);
    op.emit_code(ctx, target_info, sink);

    auto expected_null_code = R"ABI(optional<decimal32> tmp_0 = in_1;
optional<decimal32> tmp_1{};
if (tmp_0.validity) {
auto evaluated_tmp_1 = cudf_row_op_IDENTITY_decimal32(tmp_0.value);
tmp_1 = {evaluated_tmp_1, 1};
}
)ABI";

    EXPECT_EQ(sink.get_code(), expected_null_code);
  }
}

TEST_F(RowIRCodeGenTest, BinaryOperation)
{
  row_ir::target_info target_info{row_ir::target::TRANSFORM};

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*d32);
    row_ir::code_sink sink;
    row_ir::node op{row_ir::opcode::ADD,
                    std::nullopt,
                    cudf::error_policy::PROPAGATE,
                    row_ir::node{row_ir::input_reference{0}},
                    row_ir::node{row_ir::input_reference{0}}};
    op.instantiate(ctx);
    op.emit_code(ctx, target_info, sink);

    auto expected_code = R"ABI(optional<i32> tmp_0 = in_0;
optional<i32> tmp_1{};
if (tmp_0.validity && tmp_0.validity) {
auto evaluated_tmp_1 = cudf_row_op_ADD_i32_i32(tmp_0.value, tmp_0.value);
tmp_1 = {evaluated_tmp_1, 1};
}
)ABI";

    EXPECT_EQ(sink.get_code(), expected_code);
  }

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*d32);
    row_ir::code_sink sink;
    row_ir::node op{row_ir::opcode::ADD,
                    std::nullopt,
                    cudf::error_policy::PROPAGATE,
                    row_ir::node{row_ir::input_reference{1}},
                    row_ir::node{row_ir::input_reference{1}}};
    op.instantiate(ctx);
    op.emit_code(ctx, target_info, sink);

    auto expected_null_code = R"ABI(optional<decimal32> tmp_0 = in_1;
optional<decimal32> tmp_1{};
if (tmp_0.validity && tmp_0.validity) {
auto evaluated_tmp_1 = cudf_row_op_ADD_decimal32_decimal32(tmp_0.value, tmp_0.value);
tmp_1 = {evaluated_tmp_1, 1};
}
)ABI";

    EXPECT_EQ(sink.get_code(), expected_null_code);
  }
}

TEST_F(RowIRCodeGenTest, BinaryOperationOverflow)
{
  row_ir::target_info target_info{row_ir::target::TRANSFORM};

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*d32);
    row_ir::code_sink sink;
    row_ir::node op{row_ir::opcode::ADD_OVERFLOW,
                    std::nullopt,
                    cudf::error_policy::PROPAGATE,
                    row_ir::node{row_ir::input_reference{0}},
                    row_ir::node{row_ir::input_reference{0}}};
    op.instantiate(ctx);
    op.emit_code(ctx, target_info, sink);

    auto expected_code = R"ABI(optional<i32> tmp_0 = in_0;
optional<i32> tmp_1{};
if (tmp_0.validity && tmp_0.validity) {
auto evaluated_tmp_1 = cudf_row_op_ADD_OVERFLOW_i32_i32(tmp_0.value, tmp_0.value);
if (evaluated_tmp_1.error != success) { return evaluated_tmp_1.error; }
tmp_1.validity = evaluated_tmp_1.error == success;
if (tmp_1.validity) { tmp_1.value = evaluated_tmp_1.value; }
}
)ABI";

    EXPECT_EQ(sink.get_code(), expected_code);
  }

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    ctx.set_has_nulls(true);  // needed for error_policy::NULLIFY
    [[maybe_unused]] auto in0 = ctx.add_input(*i32);
    [[maybe_unused]] auto in1 = ctx.add_input(*d32);
    row_ir::code_sink sink;
    row_ir::node op{row_ir::opcode::ADD_OVERFLOW,
                    std::nullopt,
                    cudf::error_policy::NULLIFY,
                    row_ir::node{row_ir::input_reference{0}},
                    row_ir::node{row_ir::input_reference{0}}};
    op.instantiate(ctx);
    op.emit_code(ctx, target_info, sink);

    auto expected_code = R"ABI(optional<i32> tmp_0 = in_0;
optional<i32> tmp_1{};
if (tmp_0.validity && tmp_0.validity) {
auto evaluated_tmp_1 = cudf_row_op_ADD_OVERFLOW_i32_i32(tmp_0.value, tmp_0.value);
tmp_1.validity = evaluated_tmp_1.error == success;
if (tmp_1.validity) { tmp_1.value = evaluated_tmp_1.value; }
}
)ABI";

    EXPECT_EQ(sink.get_code(), expected_code);
  }
}

TEST_F(RowIRCodeGenTest, VectorLengthOperation)
{
  row_ir::target_info target_info{row_ir::target::TRANSFORM};

  auto length_operation = [&](int32_t input0, int32_t input1, int32_t output) {
    // This function generates the IR for the vector length operation:
    // length(v) = sqrt(x^2 + y^2)
    // where v = (x, y) and v is a 2D vector.
    auto x2 = row_ir::node(row_ir::opcode::MUL,
                           std::nullopt,
                           cudf::error_policy::PROPAGATE,
                           row_ir::node{row_ir::input_reference{input0}},
                           row_ir::node{row_ir::input_reference{input0}});

    auto y2 = row_ir::node(row_ir::opcode::MUL,
                           std::nullopt,
                           cudf::error_policy::PROPAGATE,
                           row_ir::node{row_ir::input_reference{input1}},
                           row_ir::node{row_ir::input_reference{input1}});

    auto sum = row_ir::node(row_ir::opcode::ADD,
                            std::nullopt,
                            cudf::error_policy::PROPAGATE,
                            std::move(x2),
                            std::move(y2));

    auto length = row_ir::node(
      row_ir::opcode::SQRT, std::nullopt, cudf::error_policy::PROPAGATE, std::move(sum));

    return row_ir::node(row_ir::output_reference{0}, std::move(length));
  };

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0  = ctx.add_input(*f64);
    [[maybe_unused]] auto in1  = ctx.add_input(*f64);
    [[maybe_unused]] auto out0 = ctx.add_output();
    row_ir::code_sink sink;

    auto expr_ir = length_operation(0, 1, 0);
    expr_ir.instantiate(ctx);
    expr_ir.emit_code(ctx, target_info, sink);

    auto expected_code = R"ABI(optional<f64> tmp_0 = in_0;
optional<f64> tmp_1{};
if (tmp_0.validity && tmp_0.validity) {
auto evaluated_tmp_1 = cudf_row_op_MUL_f64_f64(tmp_0.value, tmp_0.value);
tmp_1 = {evaluated_tmp_1, 1};
}
optional<f64> tmp_2 = in_1;
optional<f64> tmp_3{};
if (tmp_2.validity && tmp_2.validity) {
auto evaluated_tmp_3 = cudf_row_op_MUL_f64_f64(tmp_2.value, tmp_2.value);
tmp_3 = {evaluated_tmp_3, 1};
}
optional<f64> tmp_4{};
if (tmp_1.validity && tmp_3.validity) {
auto evaluated_tmp_4 = cudf_row_op_ADD_f64_f64(tmp_1.value, tmp_3.value);
tmp_4 = {evaluated_tmp_4, 1};
}
optional<f64> tmp_5{};
if (tmp_4.validity) {
auto evaluated_tmp_5 = cudf_row_op_SQRT_f64(tmp_4.value);
tmp_5 = {evaluated_tmp_5, 1};
}
set_validity(outputs, 0, row, tmp_5.validity);
if (tmp_5.validity) {
auto stored_tmp_6 = set_f64(outputs, 0, row, tmp_5.value);
if (stored_tmp_6 != success) { return stored_tmp_6; }
}
)ABI";

    EXPECT_EQ(sink.get_code(), expected_code);
  }
}

TEST_F(RowIRCodeGenTest, AstConversionBasic)
{
  cudf::ast::tree ast_tree;
  auto forty_two = cudf::numeric_scalar(42);
  auto& column_ref =
    ast_tree.push(cudf::ast::column_reference{0, cudf::ast::table_reference::LEFT});
  auto& forty_two_literal = ast_tree.push(cudf::ast::literal{forty_two});
  auto& add_op            = ast_tree.push(
    cudf::ast::operation{cudf::ast::ast_operator::ADD, forty_two_literal, column_ref});

  auto column = cudf::test::fixed_width_column_wrapper<int32_t>({69, 69, 69, 69, 69, 69}).release();

  auto expected_iter = cuda::constant_iterator{69 + 42};
  auto expected =
    cudf::test::fixed_width_column_wrapper<int32_t>(expected_iter, expected_iter + column->size());

  std::array<std::reference_wrapper<cudf::ast::expression const>, 1> output_expressions{add_op};
  auto transform_args =
    row_ir::ast_converter::compute_table(row_ir::target::TRANSFORM,
                                         output_expressions,
                                         cudf::table_view{{*column}},
                                         cudf::table_view{},
                                         "expression",
                                         cudf::get_default_stream(),
                                         cudf::get_current_device_resource_ref());

  ASSERT_EQ(transform_args.scalar_columns.size(), 1);
  ASSERT_EQ(transform_args.scalar_columns[0]->view().size(), 1);
  EXPECT_EQ(transform_args.source_type, cudf::udf_source_type::CUDA);
  EXPECT_EQ(transform_args.is_null_aware, cudf::null_aware::YES);
  EXPECT_EQ(transform_args.outputs.size(), 1);
  EXPECT_EQ(transform_args.outputs[0].nullability, cudf::output_nullability::ALL_VALID);
  EXPECT_EQ(transform_args.outputs[0].type, cudf::data_type{cudf::type_id::INT32});
  ASSERT_EQ(transform_args.inputs.size(), 2);

  /// The first input should be a scalar value of 42
  ASSERT_TRUE(std::holds_alternative<cudf::scalar_column_view>(transform_args.inputs[0]));
  EXPECT_EQ(std::get<cudf::scalar_column_view>(transform_args.inputs[0]).type(),
            cudf::data_type{cudf::type_id::INT32});
  EXPECT_EQ(std::get<cudf::scalar_column_view>(transform_args.inputs[0]).null_count(), 0);

  /// The input column should be the second column in the transform args
  ASSERT_TRUE(std::holds_alternative<cudf::column_view>(transform_args.inputs[1]));
  ASSERT_EQ(std::get<cudf::column_view>(transform_args.inputs[1]).size(), column->size());
  EXPECT_EQ(std::get<cudf::column_view>(transform_args.inputs[1]).type(), column->type());
  EXPECT_EQ(std::get<cudf::column_view>(transform_args.inputs[1]).null_count(),
            column->null_count());

  auto expected_udf = R"ABI(optional<i32> tmp_0 = in_0;
optional<i32> tmp_1 = in_1;
optional<i32> tmp_2{};
if (tmp_0.validity && tmp_1.validity) {
auto evaluated_tmp_2 = cudf_row_op_ADD_i32_i32(tmp_0.value, tmp_1.value);
tmp_2 = {evaluated_tmp_2, 1};
}
auto stored_tmp_3 = set_i32(outputs, 0, row, tmp_2.value);
if (stored_tmp_3 != success) { return stored_tmp_3; }
)ABI";

  EXPECT_NE(transform_args.udf.find(expected_udf), std::string::npos);
  EXPECT_NE(transform_args.udf.find("status transform_operator"), std::string::npos);

  auto result = cudf::transform(transform_args.udf,
                                transform_args.source_type,
                                transform_args.is_null_aware,
                                transform_args.user_data,
                                transform_args.inputs,
                                transform_args.outputs,
                                std::move(transform_args.string_offsets),
                                transform_args.row_size);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->get_column(0).view());
}

TEST_F(RowIRCodeGenTest, FilterPredicate)
{
  row_ir::target_info target_info{row_ir::target::TRANSFORM};

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*b8);
    row_ir::code_sink sink;
    row_ir::node filter_predicate(row_ir::opcode::PREDICATE,
                                  std::nullopt,
                                  cudf::error_policy::PROPAGATE,
                                  row_ir::node{row_ir::input_reference{0}});
    filter_predicate.instantiate(ctx);
    filter_predicate.emit_code(ctx, target_info, sink);

    auto expected_code = R"ABI(optional<bool8> tmp_0 = in_0;
optional<bool8> tmp_1{};
auto evaluated_tmp_1 = cudf_row_op_PREDICATE_bool8(tmp_0);
tmp_1 = {evaluated_tmp_1, 1};
)ABI";

    EXPECT_EQ(sink.get_code(), expected_code);
  }

  {
    row_ir::instance_context ctx{cudf::get_default_stream(),
                                 cudf::get_current_device_resource_ref()};
    [[maybe_unused]] auto in0 = ctx.add_input(*b8);
    row_ir::code_sink sink;
    row_ir::node filter_predicate(row_ir::opcode::PREDICATE,
                                  std::nullopt,
                                  cudf::error_policy::PROPAGATE,
                                  row_ir::node{row_ir::input_reference{0}});
    ctx.set_has_nulls(true);
    filter_predicate.instantiate(ctx);
    filter_predicate.emit_code(ctx, target_info, sink);

    auto expected_code = R"ABI(optional<bool8> tmp_0 = in_0;
optional<bool8> tmp_1{};
auto evaluated_tmp_1 = cudf_row_op_PREDICATE_bool8(tmp_0);
tmp_1 = {evaluated_tmp_1, 1};
)ABI";

    EXPECT_EQ(sink.get_code(), expected_code);
  }
}

TEST_F(RowIRCodeGenTest, CrossOutputCSE)
{
  auto ref = cudf::ast::column_reference{3};
  auto sum = cudf::ast::operation{cudf::ast::ast_operator::ADD, ref, ref};
  auto mul = cudf::ast::operation{cudf::ast::ast_operator::MUL, sum, ref};
  auto sub = cudf::ast::operation{cudf::ast::ast_operator::SUB, sum, ref};
  std::array<std::reference_wrapper<cudf::ast::expression const>, 2> expressions{mul, sub};

  row_ir::ast_converter converter{
    cudf::get_default_stream(), cudf::get_current_device_resource_ref(), table, {}};
  auto const& [code, null_aware, nullability] =
    converter.generate_code(row_ir::target::TRANSFORM, expressions, "compute_operation");

  auto expected_code = R"ABI(optional<i32> tmp_0 = in_0;
optional<i32> tmp_1{};
if (tmp_0.validity && tmp_0.validity) {
auto evaluated_tmp_1 = cudf_row_op_ADD_i32_i32(tmp_0.value, tmp_0.value);
tmp_1 = {evaluated_tmp_1, 1};
}
optional<i32> tmp_2{};
if (tmp_1.validity && tmp_0.validity) {
auto evaluated_tmp_2 = cudf_row_op_MUL_i32_i32(tmp_1.value, tmp_0.value);
tmp_2 = {evaluated_tmp_2, 1};
}
auto stored_tmp_3 = set_i32(outputs, 0, row, tmp_2.value);
if (stored_tmp_3 != success) { return stored_tmp_3; }
optional<i32> tmp_4{};
if (tmp_1.validity && tmp_0.validity) {
auto evaluated_tmp_4 = cudf_row_op_SUB_i32_i32(tmp_1.value, tmp_0.value);
tmp_4 = {evaluated_tmp_4, 1};
}
auto stored_tmp_5 = set_i32(outputs, 1, row, tmp_4.value);
if (stored_tmp_5 != success) { return stored_tmp_5; }
)ABI";

  EXPECT_NE(code.find(expected_code), std::string::npos);
  EXPECT_NE(code.find("set_i32(outputs, 1"), std::string::npos);
  EXPECT_EQ(null_aware, cudf::null_aware::YES);
  EXPECT_EQ(nullability.size(), 2);
}

TEST_F(RowIRCodeGenTest, ExactFragmentDependencies)
{
  auto ref = cudf::ast::column_reference{0};
  auto sum = cudf::ast::operation{cudf::ast::ast_operator::ADD, ref, ref};
  auto mul = cudf::ast::operation{cudf::ast::ast_operator::MUL, sum, sum};
  std::array<std::reference_wrapper<cudf::ast::expression const>, 2> expressions{sum, mul};
  auto nullable =
    cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}, {true, false, true}).release();
  auto args     = row_ir::ast_converter::compute_table(row_ir::target::TRANSFORM,
                                                   expressions,
                                                   cudf::table_view{{*nullable}},
                                                       {},
                                                   "expression",
                                                   cudf::get_default_stream(),
                                                   cudf::get_current_device_resource_ref());
  auto expected = std::vector<std::string>{"load_i32_nullable",
                                           "cudf_row_op_ADD_i32_i32",
                                           "set_validity",
                                           "set_i32",
                                           "cudf_row_op_MUL_i32_i32"};
  EXPECT_EQ(args.required_symbols, expected);
}

TEST_F(RowIRCodeGenTest, PredicateFragmentDependencies)
{
  auto other     = cudf::test::fixed_width_column_wrapper<int32_t>({3, 2, 1}).release();
  auto lhs       = cudf::ast::column_reference{0};
  auto rhs       = cudf::ast::column_reference{1};
  auto predicate = cudf::ast::operation{cudf::ast::ast_operator::EQUAL, lhs, rhs};
  auto args      = row_ir::ast_converter::filter(row_ir::target::PREDICATE,
                                            predicate,
                                            cudf::table_view{{*i32, *other}},
                                                 {},
                                            "predicate",
                                            cudf::get_default_stream(),
                                            cudf::get_current_device_resource_ref());
  auto expected  = std::vector<std::string>{
    "load_i32", "cudf_row_op_EQUAL_i32_i32", "cudf_row_op_PREDICATE_bool8"};
  EXPECT_EQ(args.required_symbols, expected);
}

TEST_F(RowIRCodeGenTest, DictionaryFragmentDependencies)
{
  auto dictionary = cudf::dictionary::encode(i32->view());
  auto ref        = cudf::ast::column_reference{0};
  auto sum        = cudf::ast::operation{cudf::ast::ast_operator::ADD, ref, ref};
  std::array<std::reference_wrapper<cudf::ast::expression const>, 1> expressions{sum};
  auto args = row_ir::ast_converter::compute_table(row_ir::target::TRANSFORM,
                                                   expressions,
                                                   cudf::table_view{{*dictionary}},
                                                   {},
                                                   "expression",
                                                   cudf::get_default_stream(),
                                                   cudf::get_current_device_resource_ref());
  auto expected =
    std::vector<std::string>{"load_dict_i32_index_i32", "cudf_row_op_ADD_i32_i32", "set_i32"};
  EXPECT_EQ(args.required_symbols, expected);
}

TEST_F(RowIRCodeGenTest, FoldedComparisonFragmentDependencies)
{
  auto strings = cudf::test::strings_column_wrapper({"a", "b", "c"}).release();
  auto ref     = cudf::ast::column_reference{0};
  auto equal   = cudf::ast::operation{cudf::ast::ast_operator::EQUAL, ref, ref};
  std::array<std::reference_wrapper<cudf::ast::expression const>, 1> expressions{equal};
  auto args     = row_ir::ast_converter::compute_table(row_ir::target::TRANSFORM,
                                                   expressions,
                                                   cudf::table_view{{*strings}},
                                                       {},
                                                   "expression",
                                                   cudf::get_default_stream(),
                                                   cudf::get_current_device_resource_ref());
  auto expected = std::vector<std::string>{"load_string_view_offset_i32", "set_bool8"};
  EXPECT_EQ(args.required_symbols, expected);
}

CUDF_TEST_PROGRAM_MAIN()
