/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <binaryop/compiled/binary_ops.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>

namespace cudf::test {
template <typename T>
struct TypedBinopStructCompare : BaseFixture {
};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(TypedBinopStructCompare, NumericTypesNotBool);
TYPED_TEST(TypedBinopStructCompare, binopcompare_no_nulls)
{
  using T = TypeParam;

  auto col1 = fixed_width_column_wrapper<T>{26, 0, 14, 116, 89, 62, 63, 0, 121};
  auto col2 = fixed_width_column_wrapper<T>{117, 34, 23, 29, 2, 37, 63, 0, 121};

  auto strings1 = strings_column_wrapper{"0a", "1c", "2d", "3b", "5c", "6", "7d", "9g", "0h"};
  auto strings2 = strings_column_wrapper{"0b", "0c", "2d", "3a", "4c", "6", "8e", "9f", "0h"};

  std::vector<std::unique_ptr<column>> lhs_columns;
  lhs_columns.push_back(col1.release());
  lhs_columns.push_back(strings1.release());
  auto lhs_col = cudf::make_structs_column(9, std::move(lhs_columns), 0, rmm::device_buffer{});
  std::vector<std::unique_ptr<column>> rhs_columns;
  rhs_columns.push_back(col2.release());
  rhs_columns.push_back(strings2.release());
  auto rhs_col = cudf::make_structs_column(9, std::move(rhs_columns), 0, rmm::device_buffer{});

  auto lhs     = lhs_col->view();
  auto rhs     = rhs_col->view();
  data_type dt = cudf::data_type(type_id::BOOL8);

  auto res_eq  = cudf::binops::compiled::struct_binary_op(lhs, rhs, binary_operator::EQUAL, dt);
  auto res_neq = cudf::binops::compiled::struct_binary_op(lhs, rhs, binary_operator::NOT_EQUAL, dt);
  auto res_lt  = cudf::binops::compiled::struct_binary_op(lhs, rhs, binary_operator::LESS, dt);
  auto res_gteq =
    cudf::binops::compiled::struct_binary_op(lhs, rhs, binary_operator::GREATER_EQUAL, dt);
  auto res_gt = cudf::binops::compiled::struct_binary_op(lhs, rhs, binary_operator::GREATER, dt);
  auto res_lteq =
    cudf::binops::compiled::struct_binary_op(lhs, rhs, binary_operator::LESS_EQUAL, dt);

  auto expected_eq   = fixed_width_column_wrapper<bool>{0, 0, 0, 0, 0, 0, 0, 0, 1};
  auto expected_neq  = fixed_width_column_wrapper<bool>{1, 1, 1, 1, 1, 1, 1, 1, 0};
  auto expected_lt   = fixed_width_column_wrapper<bool>{1, 1, 1, 0, 0, 0, 1, 0, 0};
  auto expected_gteq = fixed_width_column_wrapper<bool>{0, 0, 0, 1, 1, 1, 0, 1, 1};
  auto expected_gt   = fixed_width_column_wrapper<bool>{0, 0, 0, 1, 1, 1, 0, 1, 0};
  auto expected_lteq = fixed_width_column_wrapper<bool>{1, 1, 1, 0, 0, 0, 1, 0, 1};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_eq, expected_eq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_neq, expected_neq);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lt, expected_lt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gteq, expected_gteq);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gt, expected_gt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lteq, expected_lteq);
}

TYPED_TEST(TypedBinopStructCompare, binopcompare_with_nulls)
{
  using T = TypeParam;

  auto col1 = fixed_width_column_wrapper<T>{
    {26, 0, 14, 116, 89, 62, 63, 0, 121, 26, 0, 14, 116, 89, 62, 63, 0, 121, 1, 1, 1},
    {0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1}};
  auto col2 = fixed_width_column_wrapper<T>{
    {117, 34, 23, 29, 2, 37, 63, 0, 121, 117, 34, 23, 29, 2, 37, 63, 0, 121, 1, 1, 1},
    {1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1}};

  auto strings1 =
    strings_column_wrapper{{"0b", "",   "1c", "2a", "",   "5d", "6e", "8f", "",   "0a", "1c",
                            "2d", "3b", "5c", "6",  "7d", "9g", "0h", "1f", "2g", "3h"},
                           {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1}};
  auto strings2 =
    strings_column_wrapper{{"0a", "",   "1d", "2a", "3c", "4",  "7d", "9",  "",   "0b", "0c",
                            "2d", "3a", "4c", "6",  "8e", "9f", "0h", "1f", "2g", "3h"},
                           {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1}};

  std::vector<std::unique_ptr<column>> lhs_columns;
  lhs_columns.push_back(col1.release());
  lhs_columns.push_back(strings1.release());
  auto lhs_col = cudf::make_structs_column(21, std::move(lhs_columns), 0, rmm::device_buffer{});
  auto const lhs_nulls = thrust::host_vector<bool>(
    std::vector<bool>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0});
  lhs_col->set_null_mask(cudf::test::detail::make_null_mask(lhs_nulls.begin(), lhs_nulls.end()));

  std::vector<std::unique_ptr<column>> rhs_columns;
  rhs_columns.push_back(col2.release());
  rhs_columns.push_back(strings2.release());
  auto rhs_col = cudf::make_structs_column(21, std::move(rhs_columns), 0, rmm::device_buffer{});
  auto const rhs_nulls = thrust::host_vector<bool>(
    std::vector<bool>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0});
  rhs_col->set_null_mask(cudf::test::detail::make_null_mask(rhs_nulls.begin(), rhs_nulls.end()));

  auto lhs     = lhs_col->view();
  auto rhs     = rhs_col->view();
  data_type dt = cudf::data_type(cudf::type_id::BOOL8);

  auto res_eq  = cudf::binops::compiled::struct_binary_op(lhs, rhs, binary_operator::EQUAL, dt);
  auto res_neq = cudf::binops::compiled::struct_binary_op(lhs, rhs, binary_operator::NOT_EQUAL, dt);
  auto res_lt  = cudf::binops::compiled::struct_binary_op(lhs, rhs, binary_operator::LESS, dt);
  auto res_gteq =
    cudf::binops::compiled::struct_binary_op(lhs, rhs, binary_operator::GREATER_EQUAL, dt);
  auto res_gt = cudf::binops::compiled::struct_binary_op(lhs, rhs, binary_operator::GREATER, dt);
  auto res_lteq =
    cudf::binops::compiled::struct_binary_op(lhs, rhs, binary_operator::LESS_EQUAL, dt);

  auto expected_eq = fixed_width_column_wrapper<bool>{
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  auto expected_neq = fixed_width_column_wrapper<bool>{
    {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  auto expected_lt = fixed_width_column_wrapper<bool>{
    {1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  auto expected_gteq = fixed_width_column_wrapper<bool>{
    {0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  auto expected_gt = fixed_width_column_wrapper<bool>{
    {0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  auto expected_lteq = fixed_width_column_wrapper<bool>{
    {1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_eq, expected_eq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_neq, expected_neq);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lt, expected_lt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gteq, expected_gteq);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gt, expected_gt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lteq, expected_lteq);
}

TYPED_TEST(TypedBinopStructCompare, binopcompare_nested_structs)
{
  using T = TypeParam;

  auto col1 = fixed_width_column_wrapper<T>{
    104, 40, 105, 1, 86, 128, 25, 47, 39, 117, 125, 92, 101, 59, 69, 48, 36, 50};
  auto col2 = fixed_width_column_wrapper<T>{
    {104, 40, 105, 1, 86, 128, 25, 47, 39, 117, 125, 92, 101, 59, 69, 48, 36, 50},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  auto col3 = fixed_width_column_wrapper<T>{
    {26, 0, 14, 116, 89, 62, 63, 0, 121, 26, 0, 14, 116, 89, 62, 63, 0, 121},
    {0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  auto col4 = fixed_width_column_wrapper<T>{
    {117, 34, 23, 29, 2, 37, 63, 0, 121, 117, 34, 23, 29, 2, 37, 63, 0, 121},
    {1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

  auto strings1 = strings_column_wrapper{{"0b",
                                          "",
                                          "1c",
                                          "2a",
                                          "",
                                          "5d",
                                          "6e",
                                          "8f",
                                          "",
                                          "0a",
                                          "1c",
                                          "2d",
                                          "3b",
                                          "5c",
                                          "6",
                                          "7d",
                                          "9g",
                                          "0h"},
                                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0}};
  auto strings2 = strings_column_wrapper{{"0a",
                                          "",
                                          "1d",
                                          "2a",
                                          "3c",
                                          "4",
                                          "7d",
                                          "9",
                                          "",
                                          "0b",
                                          "0c",
                                          "2d",
                                          "3a",
                                          "4c",
                                          "6",
                                          "8e",
                                          "9f",
                                          "0h"},
                                         {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0}};

  auto struct_col1 = structs_column_wrapper{col3, strings1};
  auto nested_col1 = structs_column_wrapper{col1, struct_col1}.release();
  auto struct_col2 = structs_column_wrapper{col4, strings2};
  auto nested_col2 = structs_column_wrapper{col2, struct_col2}.release();

  data_type dt = cudf::data_type(cudf::type_id::BOOL8);

  auto res_eq = cudf::binops::compiled::struct_binary_op(
    *nested_col1, *nested_col2, binary_operator::EQUAL, dt);
  auto res_neq = cudf::binops::compiled::struct_binary_op(
    *nested_col1, *nested_col2, binary_operator::NOT_EQUAL, dt);
  auto res_lt =
    cudf::binops::compiled::struct_binary_op(*nested_col1, *nested_col2, binary_operator::LESS, dt);
  auto res_gteq = cudf::binops::compiled::struct_binary_op(
    *nested_col1, *nested_col2, binary_operator::GREATER_EQUAL, dt);
  auto res_gt = cudf::binops::compiled::struct_binary_op(
    *nested_col1, *nested_col2, binary_operator::GREATER, dt);
  auto res_lteq = cudf::binops::compiled::struct_binary_op(
    *nested_col1, *nested_col2, binary_operator::LESS_EQUAL, dt);

  auto expected_eq =
    fixed_width_column_wrapper<bool>{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};
  auto expected_neq =
    fixed_width_column_wrapper<bool>{1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0};
  auto expected_lt =
    fixed_width_column_wrapper<bool>{1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0};
  auto expected_gteq =
    fixed_width_column_wrapper<bool>{0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1};
  auto expected_gt =
    fixed_width_column_wrapper<bool>{0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
  auto expected_lteq =
    fixed_width_column_wrapper<bool>{1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_eq, expected_eq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_neq, expected_neq);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lt, expected_lt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gteq, expected_gteq);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gt, expected_gt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lteq, expected_lteq);
}

}  // namespace cudf::test
