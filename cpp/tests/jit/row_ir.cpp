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

#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_factories.hpp>

#include <algorithm>
#include <cctype>

using namespace cudf;
struct RowIRCudaCodeGenTest : public ::testing::Test {};

TEST_F(RowIRCudaCodeGenTest, GetInput)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  row_ir::var_info inputs[] = {{"in_0", {data_type{type_id::INT32}, false}},
                               {"in_1", {data_type{type_id::INT32}, true}}};

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

    auto expected_null_code = "cuda::std::optional<int32_t> tmp_0 = in_1;";

    EXPECT_EQ(null_code, expected_null_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, SetOutput)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  row_ir::var_info inputs[] = {{"in_0", {data_type{type_id::INT32}, false}},
                               {"in_1", {data_type{type_id::INT32}, true}}};

  row_ir::untyped_var_info outputs[] = {{"out_0"}, {"out_1"}};

  row_ir::instance_info info{inputs, outputs};

  {
    row_ir::instance_context ctx{};
    row_ir::set_output set_ouput_0{0, std::make_unique<row_ir::get_input>(0)};
    set_ouput_0.instantiate(ctx, info);
    auto code = set_ouput_0.generate_code(ctx, target_info, info);

    auto expected_code =
      "int32_t tmp_0 = in_0;\n"
      "int32_t tmp_1 = tmp_0;"
      "\n*out_0 = tmp_1;";

    EXPECT_EQ(code, expected_code);
  }

  {
    row_ir::instance_context ctx{};
    row_ir::set_output set_output_1{1, std::make_unique<row_ir::get_input>(1)};
    set_output_1.instantiate(ctx, info);
    auto null_code = set_output_1.generate_code(ctx, target_info, info);

    auto expected_null_code =
      "cuda::std::optional<int32_t> tmp_0 = in_1;\n"
      "cuda::std::optional<int32_t> tmp_1 = tmp_0;\n"
      "*out_1 = tmp_1;";

    EXPECT_EQ(null_code, expected_null_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, UnaryOperation)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  row_ir::var_info inputs[] = {{"in_0", {data_type{type_id::INT32}, false}},
                               {"in_1", {data_type{type_id::INT32}, true}}};

  row_ir::untyped_var_info outputs[] = {{"out_0"}, {"out_1"}};

  row_ir::instance_info info{inputs, outputs};

  {
    row_ir::instance_context ctx{};
    std::vector<std::unique_ptr<row_ir::node>> args;
    args.emplace_back(std::make_unique<row_ir::get_input>(0));
    row_ir::operation op{row_ir::opcode::IDENTITY, std::move(args)};
    op.instantiate(ctx, info);
    auto code = op.generate_code(ctx, target_info, info);

    auto expected_code =
      "int32_t tmp_0 = in_0;\n"
      "int32_t tmp_1 = "
      "cudf::ast::operator_functor<cudf::ast::ast_operator::IDENTITY, "
      "false>(tmp_0);";

    EXPECT_EQ(code, expected_code);
  }

  {
    row_ir::instance_context ctx{};

    std::vector<std::unique_ptr<row_ir::node>> args;
    args.emplace_back(std::make_unique<row_ir::get_input>(1));
    row_ir::operation op{row_ir::opcode::IDENTITY, std::move(args)};
    op.instantiate(ctx, info);
    auto null_code = op.generate_code(ctx, target_info, info);

    auto expected_null_code =
      "cuda::std::optional<int32_t> tmp_0 = in_1;\n"
      "cuda::std::optional<int32_t> tmp_1 = "
      "cudf::ast::operator_functor<cudf::ast::ast_operator::IDENTITY, true>(tmp_0);";

    EXPECT_EQ(null_code, expected_null_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, BinaryOperation)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  row_ir::var_info inputs[] = {{"in_0", {data_type{type_id::INT32}, false}},
                               {"in_1", {data_type{type_id::INT32}, true}}};

  row_ir::untyped_var_info outputs[] = {{"out_0"}, {"out_1"}};

  row_ir::instance_info info{inputs, outputs};

  {
    row_ir::instance_context ctx{};
    std::vector<std::unique_ptr<row_ir::node>> args;
    args.emplace_back(std::make_unique<row_ir::get_input>(0));
    args.emplace_back(std::make_unique<row_ir::get_input>(0));
    row_ir::operation op{row_ir::opcode::ADD, std::move(args)};
    op.instantiate(ctx, info);
    auto code = op.generate_code(ctx, target_info, info);

    auto expected_code =
      "int32_t tmp_0 = in_0;\n"
      "int32_t tmp_1 = in_0;\n"
      "int32_t tmp_2 = "
      "cudf::ast::operator_functor<cudf::ast::ast_operator::ADD, "
      "false>(tmp_0, tmp_1);";

    EXPECT_EQ(code, expected_code);
  }

  {
    row_ir::instance_context ctx{};
    std::vector<std::unique_ptr<row_ir::node>> args;
    args.emplace_back(std::make_unique<row_ir::get_input>(1));
    args.emplace_back(std::make_unique<row_ir::get_input>(1));
    row_ir::operation op{row_ir::opcode::ADD, std::move(args)};
    op.instantiate(ctx, info);
    auto null_code = op.generate_code(ctx, target_info, info);

    auto expected_null_code =
      "cuda::std::optional<int32_t> tmp_0 = in_1;\n"
      "cuda::std::optional<int32_t> tmp_1 = in_1;\n"
      "cuda::std::optional<int32_t> tmp_2 = "
      "cudf::ast::operator_functor<cudf::ast::ast_operator::ADD, true>(tmp_0, tmp_1);";

    EXPECT_EQ(null_code, expected_null_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, VectorLengthOperation)
{
  row_ir::target_info target_info{row_ir::target::CUDA};

  row_ir::var_info inputs[] = {
    {"in_0", {data_type{type_id::FLOAT64}, false}},
    {"in_1", {data_type{type_id::FLOAT64}, false}},
    {"in_2", {data_type{type_id::FLOAT64}, true}},
    {"in_3", {data_type{type_id::FLOAT64}, true}},
  };

  row_ir::untyped_var_info outputs[] = {{"out_0"}, {"out_1"}};

  row_ir::instance_info info{inputs, outputs};

  auto length_operation = [&](int32_t input0, int32_t input1, int32_t output) {
    // This function generates the IR for the vector length operation:
    // length(v) = sqrt(x^2 + y^2)
    // where v = (x, y) is a 2D vector.
    std::vector<std::unique_ptr<row_ir::node>> square0_args;
    square0_args.push_back(std::make_unique<row_ir::get_input>(input0));
    square0_args.push_back(std::make_unique<row_ir::get_input>(input0));

    auto square0 =
      std::make_unique<row_ir::operation>(row_ir::opcode::MUL, std::move(square0_args));

    std::vector<std::unique_ptr<row_ir::node>> square1_args;
    square1_args.push_back(std::make_unique<row_ir::get_input>(input1));
    square1_args.push_back(std::make_unique<row_ir::get_input>(input1));

    auto square1 =
      std::make_unique<row_ir::operation>(row_ir::opcode::MUL, std::move(square1_args));

    std::vector<std::unique_ptr<row_ir::node>> add_args;
    add_args.push_back(std::move(square0));
    add_args.push_back(std::move(square1));

    auto add = std::make_unique<row_ir::operation>(row_ir::opcode::ADD, std::move(add_args));

    std::vector<std::unique_ptr<row_ir::node>> sqrt_args;
    sqrt_args.push_back(std::move(add));

    auto sqrt = std::make_unique<row_ir::operation>(row_ir::opcode::SQRT, std::move(sqrt_args));

    return std::make_unique<row_ir::set_output>(output, std::move(sqrt));
  };

  {
    row_ir::instance_context ctx{};

    auto expr_ir = length_operation(0, 1, 0);
    expr_ir->instantiate(ctx, info);

    auto code = expr_ir->generate_code(ctx, target_info, info);

    auto expected_code =
      "double tmp_0 = in_0;\n"
      "double tmp_1 = in_0;\n"
      "double tmp_2 = "
      "cudf::ast::operator_functor<cudf::ast::ast_operator::MUL, false>(tmp_0, tmp_1);\n"
      "double tmp_3 = in_1;\n"
      "double tmp_4 = in_1;\n"
      "double tmp_5 = "
      "cudf::ast::operator_functor<cudf::ast::ast_operator::MUL, false>(tmp_3, "
      "tmp_4);\n"
      "double tmp_6 = "
      "cudf::ast::operator_functor<cudf::ast::ast_operator::ADD, false>(tmp_2, tmp_5);\n"
      "double tmp_7 = "
      "cudf::ast::operator_functor<cudf::ast::ast_operator::SQRT, false>(tmp_6);\n"
      "double tmp_8 = tmp_7;\n"
      "*out_0 = tmp_8;";

    EXPECT_EQ(code, expected_code);
  }

  {
    row_ir::instance_context ctx{};

    auto expr_ir = length_operation(2, 3, 1);
    expr_ir->instantiate(ctx, info);

    auto null_code = expr_ir->generate_code(ctx, target_info, info);

    auto expected_null_code =
      "cuda::std::optional<double> tmp_0 = in_2;\n"
      "cuda::std::optional<double> tmp_1 = in_2;\n"
      "cuda::std::optional<double> tmp_2 = "
      "cudf::ast::operator_functor<cudf::ast::ast_operator::MUL, true>(tmp_0, tmp_1);\n"
      "cuda::std::optional<double> tmp_3 = in_3;\n"
      "cuda::std::optional<double> tmp_4 = in_3;\n"
      "cuda::std::optional<double> tmp_5 = "
      "cudf::ast::operator_functor<cudf::ast::ast_operator::MUL, true>(tmp_3, tmp_4);\n"
      "cuda::std::optional<double> tmp_6 = "
      "cudf::ast::operator_functor<cudf::ast::ast_operator::ADD, true>(tmp_2, tmp_5);\n"
      "cuda::std::optional<double> tmp_7 = "
      "cudf::ast::operator_functor<cudf::ast::ast_operator::SQRT, true>(tmp_6);\n"
      "cuda::std::optional<double> tmp_8 = tmp_7;\n"
      "*out_1 = tmp_8;";

    EXPECT_EQ(null_code, expected_null_code);
  }
}

TEST_F(RowIRCudaCodeGenTest, AstConversionColumn)
{
  row_ir::ast_converter converter;

  ast::tree ast_tree;
  auto fourty_two          = cudf::numeric_scalar(42);
  auto& column_ref         = ast_tree.push(ast::column_reference{0, ast::table_reference::LEFT});
  auto& fourty_two_literal = ast_tree.push(ast::literal{fourty_two});
  auto& add_op =
    ast_tree.push(ast::operation{ast::ast_operator::ADD, fourty_two_literal, column_ref});

  auto column =
    cudf::make_numeric_column(data_type{type_id::INT32}, 2000, cudf::mask_state::UNALLOCATED);

  row_ir::ast_args args{.table = cudf::table_view{{column->view()}}, .table_column_names = {}};

  auto transform_args = converter.compute_column(row_ir::target::CUDA,
                                                 add_op,
                                                 false,
                                                 args,
                                                 rmm::cuda_stream_view{},
                                                 cudf::get_current_device_resource_ref());

  EXPECT_EQ(transform_args.scalar_columns.size(), 1);
  EXPECT_EQ(transform_args.scalar_columns[0]->view().size(), 1);
  EXPECT_FALSE(transform_args.args.is_ptx);
  EXPECT_EQ(transform_args.args.output_type, data_type{type_id::INT32});
  EXPECT_EQ(transform_args.args.columns.size(), 2);

  /// Scalar column should be represented as a column of size 1
  EXPECT_EQ(transform_args.args.columns[0].size(), 1);
  EXPECT_EQ(transform_args.args.columns[0].type(), data_type{type_id::INT32});
  EXPECT_EQ(transform_args.args.columns[0].null_count(), 0);

  /// The input column should be the second column in the transform args
  EXPECT_EQ(transform_args.args.columns[1].size(), column->size());
  EXPECT_EQ(transform_args.args.columns[1].type(), column->type());
  EXPECT_EQ(transform_args.args.columns[1].null_count(), column->null_count());

  auto expected_udf = R"***(
__device__ void transform(int32_t * out_0, int32_t in_0, int32_t in_1)
{
int32_t tmp_0 = in_0;
int32_t tmp_1 = in_1;
int32_t tmp_2 = cudf::ast::operator_functor<cudf::ast::ast_operator::ADD, false>(tmp_0, tmp_1);
int32_t tmp_3 = tmp_2;
*out_0 = tmp_3;

return;
}
)***";

  EXPECT_EQ(transform_args.args.transform_udf, expected_udf);
}

TEST_F(RowIRCudaCodeGenTest, AstConversionScalar) {}