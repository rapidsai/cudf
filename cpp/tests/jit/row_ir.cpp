/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir.hpp"

#include "cudf_test/column_wrapper.hpp"

#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/transform.hpp>

#include <algorithm>
#include <cctype>

using namespace cudf;
namespace row_ir = cudf::detail::row_ir;

struct RowIRCudaCodeGenTest : public ::testing::Test {};

TEST_F(RowIRCudaCodeGenTest, GetInput)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  row_ir::var_info inputs[] = {{"in_0", {data_type{type_id::INT32}}},
                               {"in_1", {data_type{type_id::FLOAT32}}}};

  row_ir::instance_info info{inputs, {}};

  {
    row_ir::instance_context ctx{};
    row_ir::get_input get_input_0{0};
    get_input_0.instantiate(ctx, info);
    auto code = get_input_0.generate_code(ctx, target_info, info);

    auto expected_code = "int32_t tmp_0 = in_0;";

    EXPECT_EQ(code, expected_code);
  }

  {
    row_ir::instance_context ctx{};
    row_ir::get_input get_input_1{1};
    get_input_1.instantiate(ctx, info);
    auto null_code = get_input_1.generate_code(ctx, target_info, info);

    auto expected_null_code = "float tmp_0 = in_1;";

    EXPECT_EQ(null_code, expected_null_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, SetOutput)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  row_ir::var_info inputs[] = {{"in_0", {data_type{type_id::INT32}}},
                               {"in_1", {data_type{type_id::FLOAT32}}}};

  row_ir::untyped_var_info outputs[] = {{"out_0"}, {"out_1"}};

  row_ir::instance_info info{inputs, outputs};

  {
    row_ir::instance_context ctx{};
    row_ir::set_output set_output_0{0, std::make_unique<row_ir::get_input>(0)};
    set_output_0.instantiate(ctx, info);
    auto code = set_output_0.generate_code(ctx, target_info, info);

    auto expected_code =
      R"***(int32_t tmp_0 = in_0;
int32_t tmp_1 = tmp_0;
*out_0 = tmp_1;)***";

    EXPECT_EQ(code, expected_code);
  }

  {
    row_ir::instance_context ctx{};
    row_ir::set_output set_output_1{1, std::make_unique<row_ir::get_input>(1)};
    set_output_1.instantiate(ctx, info);
    auto code = set_output_1.generate_code(ctx, target_info, info);

    auto expected_code =
      R"***(float tmp_0 = in_1;
float tmp_1 = tmp_0;
*out_1 = tmp_1;)***";

    EXPECT_EQ(code, expected_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, UnaryOperation)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  row_ir::var_info inputs[] = {{"in_0", {data_type{type_id::INT32}}},
                               {"in_1", {data_type{type_id::DECIMAL32}}}};

  row_ir::untyped_var_info outputs[] = {{"out_0"}, {"out_1"}};

  row_ir::instance_info info{inputs, outputs};

  {
    row_ir::instance_context ctx{};
    row_ir::operation op{row_ir::opcode::IDENTITY,
                         row_ir::operation::operands(row_ir::get_input(0))};
    op.instantiate(ctx, info);
    auto code = op.generate_code(ctx, target_info, info);

    auto expected_code =
      R"***(int32_t tmp_0 = in_0;
int32_t tmp_1 = cudf::ast::detail::operator_functor<cudf::ast::ast_operator::IDENTITY, false>{}(tmp_0);)***";

    EXPECT_EQ(code, expected_code);
  }

  {
    row_ir::instance_context ctx{};
    row_ir::operation op{row_ir::opcode::IDENTITY,
                         row_ir::operation::operands(row_ir::get_input(1))};
    op.instantiate(ctx, info);
    auto null_code = op.generate_code(ctx, target_info, info);

    auto expected_null_code =
      R"***(numeric::decimal32 tmp_0 = in_1;
numeric::decimal32 tmp_1 = cudf::ast::detail::operator_functor<cudf::ast::ast_operator::IDENTITY, false>{}(tmp_0);)***";

    EXPECT_EQ(null_code, expected_null_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, BinaryOperation)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  row_ir::var_info inputs[] = {{"in_0", {data_type{type_id::INT32}}},
                               {"in_1", {data_type{type_id::DECIMAL32}}}};

  row_ir::untyped_var_info outputs[] = {{"out_0"}, {"out_1"}};

  row_ir::instance_info info{inputs, outputs};

  {
    row_ir::instance_context ctx{};
    row_ir::operation op{row_ir::opcode::ADD,
                         row_ir::operation::operands(row_ir::get_input(0), row_ir::get_input(0))};
    op.instantiate(ctx, info);
    auto code = op.generate_code(ctx, target_info, info);

    auto expected_code =
      R"***(int32_t tmp_0 = in_0;
int32_t tmp_1 = in_0;
int32_t tmp_2 = cudf::ast::detail::operator_functor<cudf::ast::ast_operator::ADD, false>{}(tmp_0, tmp_1);)***";

    EXPECT_EQ(code, expected_code);
  }

  {
    row_ir::instance_context ctx{};
    row_ir::operation op{row_ir::opcode::ADD,
                         row_ir::operation::operands(row_ir::get_input(1), row_ir::get_input(1))};
    op.instantiate(ctx, info);
    auto null_code = op.generate_code(ctx, target_info, info);

    auto expected_null_code =
      R"***(numeric::decimal32 tmp_0 = in_1;
numeric::decimal32 tmp_1 = in_1;
numeric::decimal32 tmp_2 = cudf::ast::detail::operator_functor<cudf::ast::ast_operator::ADD, false>{}(tmp_0, tmp_1);)***";

    EXPECT_EQ(null_code, expected_null_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, VectorLengthOperation)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  row_ir::var_info inputs[] = {
    {"in_0", {data_type{type_id::FLOAT64}}},
    {"in_1", {data_type{type_id::FLOAT64}}},
    {"in_2", {data_type{type_id::FLOAT64}}},
    {"in_3", {data_type{type_id::FLOAT64}}},
  };

  row_ir::untyped_var_info outputs[] = {{"out_0"}, {"out_1"}};

  row_ir::instance_info info{inputs, outputs};

  auto length_operation = [&](int32_t input0, int32_t input1, int32_t output) {
    // This function generates the IR for the vector length operation:
    // length(v) = sqrt(x^2 + y^2)
    // where v = (x, y) and v is a 2D vector.
    auto x2 = std::make_unique<row_ir::operation>(
      row_ir::opcode::MUL,
      row_ir::operation::operands(row_ir::get_input(input0), row_ir::get_input(input0)));

    auto y2 = std::make_unique<row_ir::operation>(
      row_ir::opcode::MUL,
      row_ir::operation::operands(row_ir::get_input(input1), row_ir::get_input(input1)));

    auto sum = std::make_unique<row_ir::operation>(
      row_ir::opcode::ADD, row_ir::operation::operands(std::move(x2), std::move(y2)));

    auto length = std::make_unique<row_ir::operation>(row_ir::opcode::SQRT,
                                                      row_ir::operation::operands(std::move(sum)));

    return std::make_unique<row_ir::set_output>(output, std::move(length));
  };

  {
    row_ir::instance_context ctx{};

    auto expr_ir = length_operation(0, 1, 0);
    expr_ir->instantiate(ctx, info);

    auto code = expr_ir->generate_code(ctx, target_info, info);

    auto expected_code =
      R"***(double tmp_0 = in_0;
double tmp_1 = in_0;
double tmp_2 = cudf::ast::detail::operator_functor<cudf::ast::ast_operator::MUL, false>{}(tmp_0, tmp_1);
double tmp_3 = in_1;
double tmp_4 = in_1;
double tmp_5 = cudf::ast::detail::operator_functor<cudf::ast::ast_operator::MUL, false>{}(tmp_3, tmp_4);
double tmp_6 = cudf::ast::detail::operator_functor<cudf::ast::ast_operator::ADD, false>{}(tmp_2, tmp_5);
double tmp_7 = cudf::ast::detail::operator_functor<cudf::ast::ast_operator::SQRT, false>{}(tmp_6);
double tmp_8 = tmp_7;
*out_0 = tmp_8;)***";

    EXPECT_EQ(code, expected_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, AstConversionBasic)
{
  ast::tree ast_tree;
  auto forty_two          = cudf::numeric_scalar(42);
  auto& column_ref        = ast_tree.push(ast::column_reference{0, ast::table_reference::LEFT});
  auto& forty_two_literal = ast_tree.push(ast::literal{forty_two});
  auto& add_op =
    ast_tree.push(ast::operation{ast::ast_operator::ADD, forty_two_literal, column_ref});

  auto column = cudf::test::fixed_width_column_wrapper<int32_t>({69, 69, 69, 69, 69, 69}).release();

  auto expected_iter = detail::make_counting_transform_iterator(0, [](auto i) { return 69 + 42; });
  auto expected =
    cudf::test::fixed_width_column_wrapper<int32_t>(expected_iter, expected_iter + column->size());

  row_ir::ast_args args{.table = cudf::table_view{{column->view()}}};

  auto transform_args =
    row_ir::ast_converter::compute_column(row_ir::target::CUDA,
                                          add_op,
                                          args,
                                          cudf::get_default_stream(),
                                          cudf::get_current_device_resource_ref());

  ASSERT_EQ(transform_args.scalar_columns.size(), 1);
  ASSERT_EQ(transform_args.scalar_columns[0]->view().size(), 1);
  EXPECT_FALSE(transform_args.is_ptx);
  EXPECT_EQ(transform_args.is_null_aware, null_aware::NO);
  EXPECT_EQ(transform_args.output_type, data_type{type_id::INT32});
  ASSERT_EQ(transform_args.columns.size(), 2);

  /// Scalar column should be represented as a column of size 1
  ASSERT_EQ(transform_args.columns[0].size(), 1);
  EXPECT_EQ(transform_args.columns[0].type(), data_type{type_id::INT32});
  EXPECT_EQ(transform_args.columns[0].null_count(), 0);

  /// The input column should be the second column in the transform args
  ASSERT_EQ(transform_args.columns[1].size(), column->size());
  EXPECT_EQ(transform_args.columns[1].type(), column->type());
  EXPECT_EQ(transform_args.columns[1].null_count(), column->null_count());

  auto expected_udf = R"***(
__device__ void expression(int32_t* out_0, int32_t in_0, int32_t in_1)
{
int32_t tmp_0 = in_0;
int32_t tmp_1 = in_1;
int32_t tmp_2 = cudf::ast::detail::operator_functor<cudf::ast::ast_operator::ADD, false>{}(tmp_0, tmp_1);
int32_t tmp_3 = tmp_2;
*out_0 = tmp_3;

return;
}
)***";

  EXPECT_EQ(transform_args.udf, expected_udf);

  auto result = cudf::transform(transform_args.columns,
                                transform_args.udf,
                                transform_args.output_type,
                                transform_args.is_ptx,
                                transform_args.user_data,
                                transform_args.is_null_aware);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

CUDF_TEST_PROGRAM_MAIN()
