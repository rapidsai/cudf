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