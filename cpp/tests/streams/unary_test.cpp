/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

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

CUDF_TEST_PROGRAM_MAIN()
