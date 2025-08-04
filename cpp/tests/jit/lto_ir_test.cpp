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

#include <cudf_test/testing_main.hpp>

#ifdef CUDF_USE_LTO_IR

#include "jit/lto_ir.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

class LtoIrTest : public cudf::test::BaseFixture {};

TEST_F(LtoIrTest, CacheInitialization)
{
  // Test that the LTO-IR cache initializes properly
  auto& cache = cudf::jit::lto_ir_cache::instance();
  
  // Should have some built-in operators
  EXPECT_TRUE(cache.is_lto_ir_available("binary_op", "add"));
  EXPECT_TRUE(cache.is_lto_ir_available("binary_op", "multiply"));
  EXPECT_TRUE(cache.is_lto_ir_available("transform", "sin"));
  EXPECT_TRUE(cache.is_lto_ir_available("transform", "sqrt"));
}

TEST_F(LtoIrTest, OperatorRegistration)
{
  auto& cache = cudf::jit::lto_ir_cache::instance();
  
  // Register a custom operator
  std::vector<std::string> mock_lto_ir_data = {"mock LTO-IR data"};
  cache.register_operator("test_op::custom", mock_lto_ir_data);
  
  // Should be able to find it
  EXPECT_TRUE(cache.is_lto_ir_available("test_op", "custom"));
  
  auto op = cache.get_operator("test_op::custom");
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->name, "test_op::custom");
  EXPECT_EQ(op->data, mock_lto_ir_data);
}

TEST_F(LtoIrTest, FallbackToTraditionalCompilation)
{
  // Test that unsupported operations fall back to traditional compilation
  auto& cache = cudf::jit::lto_ir_cache::instance();
  
  // This should not be available
  EXPECT_FALSE(cache.is_lto_ir_available("custom_op", "unsupported"));
  
  // try_compile_with_lto_ir should return nullptr for unsupported operations
  auto result = cudf::jit::try_compile_with_lto_ir(
    "custom_op", {"unsupported"}, "test_kernel", "mock CUDA source");
  
  EXPECT_EQ(result, nullptr);
}

TEST_F(LtoIrTest, BinaryOperationCompilation)
{
  // Test that LTO-IR compilation is attempted for supported binary operations
  using namespace cudf::test;
  
  auto lhs = fixed_width_column_wrapper<int32_t>({1, 2, 3, 4});
  auto rhs = fixed_width_column_wrapper<int32_t>({5, 6, 7, 8});
  
  // This should attempt LTO-IR compilation first, then fall back if needed
  // Since we don't have actual LTO-IR data, it will fall back but the call should succeed
  auto result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, cudf::data_type{cudf::type_id::INT32});
  
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->size(), 4);
}

TEST_F(LtoIrTest, TransformOperationCompilation)
{
  // Test that LTO-IR compilation is attempted for transform operations
  using namespace cudf::test;
  
  auto input = fixed_width_column_wrapper<float>({1.0, 2.0, 3.0, 4.0});
  
  // Create a simple transform expression (x + 1.0)
  std::string cuda_source = R"(
    __device__ inline void GENERIC_TRANSFORM_OP(float& output, float input) {
      output = input + 1.0f;
    }
  )";
  
  // This should attempt LTO-IR compilation first, then fall back
  auto result = cudf::transform(input, cuda_source, cudf::data_type{cudf::type_id::FLOAT32}, false);
  
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->size(), 4);
}

#endif // CUDF_USE_LTO_IR

// Always provide a basic test even when LTO-IR is disabled
#ifndef CUDF_USE_LTO_IR
TEST(LtoIrDisabled, CompileTest)
{
  // Just verify that the code compiles when LTO-IR is disabled
  EXPECT_TRUE(true);
}
#endif