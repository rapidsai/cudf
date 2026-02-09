/*
 * SPDX-FileCopyrightText: Copyright 2018-2019 BlazingDB, Inc.
 * SPDX-FileCopyrightText: Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/binaryop.hpp>

struct BinaryOperationNullTest : public cudf::test::BaseFixture {};

TEST_F(BinaryOperationNullTest, Scalar_Null_Vector_Valid)
{
  auto lhs = cudf::scalar_type_t<int32_t>(0);
  lhs.set_valid_async(false);
  auto rhs = cudf::test::fixed_width_column_wrapper<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                             cudf::test::iterators::no_nulls());

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::INT32));

  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>(
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, cudf::test::iterators::all_nulls());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected);
}

TEST_F(BinaryOperationNullTest, Scalar_Valid_Vector_NonNullable)
{
  auto lhs = cudf::scalar_type_t<int32_t>(1);
  auto rhs = cudf::test::fixed_width_column_wrapper<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::INT32));

  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected);
}

TEST_F(BinaryOperationNullTest, Scalar_Null_Vector_NonNullable)
{
  auto lhs = cudf::scalar_type_t<int32_t>(0);
  lhs.set_valid_async(false);
  auto rhs = cudf::test::fixed_width_column_wrapper<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::INT32));

  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>(
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, cudf::test::iterators::all_nulls());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected);
}

TEST_F(BinaryOperationNullTest, Vector_Null_Scalar_Valid)
{
  auto lhs = cudf::scalar_type_t<int32_t>(1);
  auto rhs = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                             cudf::test::iterators::all_nulls());

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::INT32));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), rhs);
}

TEST_F(BinaryOperationNullTest, Vector_Null_Vector_Valid)
{
  auto lhs = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                             cudf::test::iterators::all_nulls());
  auto rhs = cudf::test::fixed_width_column_wrapper<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                             cudf::test::iterators::no_nulls());

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::INT32));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), lhs);
}

TEST_F(BinaryOperationNullTest, Vector_Null_Vector_NonNullable)
{
  auto lhs = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                             cudf::test::iterators::all_nulls());
  auto rhs = cudf::test::fixed_width_column_wrapper<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::INT32));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), lhs);
}

TEST_F(BinaryOperationNullTest, Vector_Valid_Vector_NonNullable)
{
  auto lhs = cudf::test::fixed_width_column_wrapper<int32_t>({9, 8, 7, 6, 5, 4, 3, 2, 1, 0},
                                                             cudf::test::iterators::no_nulls());
  auto rhs = cudf::test::fixed_width_column_wrapper<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::INT32));

  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>(
    {9, 9, 9, 9, 9, 9, 9, 9, 9, 9}, cudf::test::iterators::no_nulls());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected);
}

TEST_F(BinaryOperationNullTest, Vector_NonNullable_Vector_NonNullable)
{
  auto lhs = cudf::test::fixed_width_column_wrapper<int32_t>({9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
  auto rhs = cudf::test::fixed_width_column_wrapper<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ADD, cudf::data_type(cudf::type_id::INT32));

  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>({9, 9, 9, 9, 9, 9, 9, 9, 9, 9});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected);
}
