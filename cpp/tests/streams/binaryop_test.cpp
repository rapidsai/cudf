/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/binaryop.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

class BinaryopTest : public cudf::test::BaseFixture {};

TEST_F(BinaryopTest, ColumnColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> lhs{10, 20, 30, 40, 50};
  cudf::test::fixed_width_column_wrapper<int32_t> rhs{15, 25, 35, 45, 55};

  cudf::binary_operation(lhs,
                         rhs,
                         cudf::binary_operator::ADD,
                         cudf::data_type(cudf::type_to_id<int32_t>()),
                         cudf::test::get_default_stream());
}

TEST_F(BinaryopTest, ColumnScalar)
{
  cudf::test::fixed_width_column_wrapper<int32_t> lhs{10, 20, 30, 40, 50};
  cudf::numeric_scalar<int32_t> rhs{23, true, cudf::test::get_default_stream()};

  cudf::binary_operation(lhs,
                         rhs,
                         cudf::binary_operator::ADD,
                         cudf::data_type(cudf::type_to_id<int32_t>()),
                         cudf::test::get_default_stream());
}

TEST_F(BinaryopTest, ScalarColumn)
{
  cudf::numeric_scalar<int32_t> lhs{42, true, cudf::test::get_default_stream()};
  cudf::test::fixed_width_column_wrapper<int32_t> rhs{15, 25, 35, 45, 55};

  cudf::binary_operation(lhs,
                         rhs,
                         cudf::binary_operator::ADD,
                         cudf::data_type(cudf::type_to_id<int32_t>()),
                         cudf::test::get_default_stream());
}
