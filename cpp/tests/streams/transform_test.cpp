/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/transform.hpp>

class TransformTest : public cudf::test::BaseFixture {};

TEST_F(TransformTest, ComputeColumn)
{
  auto c_0   = cudf::test::fixed_width_column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = cudf::test::fixed_width_column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>{13, 27, 21, 50};
  auto result   = cudf::compute_column(table, expression, cudf::test::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}
