/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/unary.hpp>

class UnaryTest : public cudf::test::BaseFixture {};

TEST_F(UnaryTest, UnaryOperation)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const column{10, 20, 30, 40, 50};

  cudf::unary_operation(column, cudf::unary_operator::ABS, cudf::test::get_default_stream());
}

TEST_F(UnaryTest, IsNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const column{10, 20, 30, 40, 50};

  cudf::is_null(column, cudf::test::get_default_stream());
}

TEST_F(UnaryTest, IsValid)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const column{10, 20, 30, 40, 50};

  cudf::is_valid(column, cudf::test::get_default_stream());
}

TEST_F(UnaryTest, Cast)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const column{10, 20, 30, 40, 50};

  cudf::cast(column, cudf::data_type{cudf::type_id::INT64}, cudf::test::get_default_stream());
}

TEST_F(UnaryTest, IsNan)
{
  cudf::test::fixed_width_column_wrapper<float> const column{10, 20, 30, 40, 50};

  cudf::is_nan(column, cudf::test::get_default_stream());
}

TEST_F(UnaryTest, IsNotNan)
{
  cudf::test::fixed_width_column_wrapper<float> const column{10, 20, 30, 40, 50};

  cudf::is_not_nan(column, cudf::test::get_default_stream());
}
