/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cudf_test/type_lists.hpp>

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

  auto res_eq   = binary_operation(lhs, rhs, binary_operator::EQUAL, dt);
  auto res_neq  = binary_operation(lhs, rhs, binary_operator::NOT_EQUAL, dt);
  auto res_lt   = binary_operation(lhs, rhs, binary_operator::LESS, dt);
  auto res_lteq = binary_operation(lhs, rhs, binary_operator::LESS_EQUAL, dt);
  auto res_gt   = binary_operation(lhs, rhs, binary_operator::GREATER, dt);
  auto res_gteq = binary_operation(lhs, rhs, binary_operator::GREATER_EQUAL, dt);

  auto expected_eq   = fixed_width_column_wrapper<bool>{0, 0, 0, 0, 0, 0, 0, 0, 1};
  auto expected_neq  = fixed_width_column_wrapper<bool>{1, 1, 1, 1, 1, 1, 1, 1, 0};
  auto expected_lt   = fixed_width_column_wrapper<bool>{1, 1, 1, 0, 0, 0, 1, 0, 0};
  auto expected_lteq = fixed_width_column_wrapper<bool>{1, 1, 1, 0, 0, 0, 1, 0, 1};
  auto expected_gt   = fixed_width_column_wrapper<bool>{0, 0, 0, 1, 1, 1, 0, 1, 0};
  auto expected_gteq = fixed_width_column_wrapper<bool>{0, 0, 0, 1, 1, 1, 0, 1, 1};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_eq, expected_eq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_neq, expected_neq);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lt, expected_lt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lteq, expected_lteq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gt, expected_gt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gteq, expected_gteq);
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

  auto res_eq   = binary_operation(lhs, rhs, binary_operator::EQUAL, dt);
  auto res_neq  = binary_operation(lhs, rhs, binary_operator::NOT_EQUAL, dt);
  auto res_lt   = binary_operation(lhs, rhs, binary_operator::LESS, dt);
  auto res_lteq = binary_operation(lhs, rhs, binary_operator::LESS_EQUAL, dt);
  auto res_gt   = binary_operation(lhs, rhs, binary_operator::GREATER, dt);
  auto res_gteq = binary_operation(lhs, rhs, binary_operator::GREATER_EQUAL, dt);

  auto expected_eq = fixed_width_column_wrapper<bool>{
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  auto expected_neq = fixed_width_column_wrapper<bool>{
    {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  auto expected_lt = fixed_width_column_wrapper<bool>{
    {1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  auto expected_lteq = fixed_width_column_wrapper<bool>{
    {1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  auto expected_gt = fixed_width_column_wrapper<bool>{
    {0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  auto expected_gteq = fixed_width_column_wrapper<bool>{
    {0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_eq, expected_eq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_neq, expected_neq);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lt, expected_lt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lteq, expected_lteq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gt, expected_gt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gteq, expected_gteq);
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

  auto s1 = strings_column_wrapper{{"0b",
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
  auto s2 = strings_column_wrapper{{"0a",
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

  auto struct_col1 = structs_column_wrapper{col3, s1};
  auto nested_col1 = structs_column_wrapper{col1, struct_col1}.release();
  auto struct_col2 = structs_column_wrapper{col4, s2};
  auto nested_col2 = structs_column_wrapper{col2, struct_col2}.release();

  auto lhs     = nested_col1->view();
  auto rhs     = nested_col2->view();
  data_type dt = cudf::data_type(cudf::type_id::BOOL8);

  auto res_eq   = binary_operation(lhs, rhs, binary_operator::EQUAL, dt);
  auto res_neq  = binary_operation(lhs, rhs, binary_operator::NOT_EQUAL, dt);
  auto res_lt   = binary_operation(lhs, rhs, binary_operator::LESS, dt);
  auto res_lteq = binary_operation(lhs, rhs, binary_operator::LESS_EQUAL, dt);
  auto res_gt   = binary_operation(lhs, rhs, binary_operator::GREATER, dt);
  auto res_gteq = binary_operation(lhs, rhs, binary_operator::GREATER_EQUAL, dt);

  auto expected_eq =
    fixed_width_column_wrapper<bool>{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};
  auto expected_neq =
    fixed_width_column_wrapper<bool>{1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0};
  auto expected_lt =
    fixed_width_column_wrapper<bool>{1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0};
  auto expected_lteq =
    fixed_width_column_wrapper<bool>{1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1};
  auto expected_gt =
    fixed_width_column_wrapper<bool>{0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
  auto expected_gteq =
    fixed_width_column_wrapper<bool>{0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_eq, expected_eq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_neq, expected_neq);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lt, expected_lt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lteq, expected_lteq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gt, expected_gt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gteq, expected_gteq);
}

TYPED_TEST(TypedBinopStructCompare, binopcompare_scalars)
{
  using T = TypeParam;

  auto col1 =
    fixed_width_column_wrapper<T>{40, 105, 68, 25, 86, 68, 25, 127, 68, 68, 68, 68, 68, 68, 68};
  auto col2 =
    fixed_width_column_wrapper<T>{{26, 0, 14, 116, 89, 62, 63, 0, 121, 5, 115, 18, 0, 88, 18},
                                  {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0}};
  auto s1 = strings_column_wrapper{
    {"6S", "5G", "4a", "5G", "", "5Z", "5e", "9a", "5G", "5", "5Gs", "5G", "", "5G2", "5G"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  auto struct_col1 = structs_column_wrapper{col2, s1};
  auto nested_col1 = structs_column_wrapper{col1, struct_col1}.release();
  auto col_val     = nested_col1->view();

  cudf::test::fixed_width_column_wrapper<T> col3{68};
  cudf::test::fixed_width_column_wrapper<T> col4{{18}, {0}};
  auto strings2    = strings_column_wrapper{"5G"};
  auto struct_col2 = structs_column_wrapper{col4, strings2};
  cudf::table_view tbl({col3, struct_col2});
  cudf::struct_scalar struct_val(tbl);
  data_type dt = cudf::data_type(cudf::type_id::BOOL8);

  auto res_eq   = binary_operation(col_val, struct_val, binary_operator::EQUAL, dt);
  auto res_neq  = binary_operation(col_val, struct_val, binary_operator::NOT_EQUAL, dt);
  auto res_lt   = binary_operation(col_val, struct_val, binary_operator::LESS, dt);
  auto res_gt   = binary_operation(col_val, struct_val, binary_operator::GREATER, dt);
  auto res_gteq = binary_operation(col_val, struct_val, binary_operator::GREATER_EQUAL, dt);
  auto res_lteq = binary_operation(col_val, struct_val, binary_operator::LESS_EQUAL, dt);

  auto expected_eq  = fixed_width_column_wrapper<bool>{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
  auto expected_neq = fixed_width_column_wrapper<bool>{1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1};
  auto expected_lt  = fixed_width_column_wrapper<bool>{1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1};
  auto expected_lteq =
    fixed_width_column_wrapper<bool>{1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1};
  auto expected_gt = fixed_width_column_wrapper<bool>{0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0};
  auto expected_gteq =
    fixed_width_column_wrapper<bool>{0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_eq, expected_eq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_neq, expected_neq);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lt, expected_lt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lteq, expected_lteq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gt, expected_gt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gteq, expected_gteq);
}

struct BinopStructCompareNAN : public cudf::test::BaseFixture {
};

TEST_F(BinopStructCompareNAN, float_nans)
{
  cudf::test::fixed_width_column_wrapper<float> lhs{
    -NAN, -NAN, -NAN, NAN, NAN, NAN, 1.0f, 0.0f, -54.3f};
  cudf::test::fixed_width_column_wrapper<float> rhs{
    -32.5f, -NAN, NAN, -0.0f, -NAN, NAN, 111.0f, -NAN, NAN};
  data_type dt = cudf::data_type(cudf::type_id::BOOL8);

  auto expected_eq   = binary_operation(lhs, rhs, binary_operator::EQUAL, dt);
  auto expected_neq  = binary_operation(lhs, rhs, binary_operator::NOT_EQUAL, dt);
  auto expected_lt   = binary_operation(lhs, rhs, binary_operator::LESS, dt);
  auto expected_gt   = binary_operation(lhs, rhs, binary_operator::GREATER, dt);
  auto expected_gteq = binary_operation(lhs, rhs, binary_operator::GREATER_EQUAL, dt);
  auto expected_lteq = binary_operation(lhs, rhs, binary_operator::LESS_EQUAL, dt);

  auto struct_lhs = structs_column_wrapper{lhs};
  auto struct_rhs = structs_column_wrapper{rhs};
  auto res_eq     = binary_operation(struct_lhs, struct_rhs, binary_operator::EQUAL, dt);
  auto res_neq    = binary_operation(struct_lhs, struct_rhs, binary_operator::NOT_EQUAL, dt);
  auto res_lt     = binary_operation(struct_lhs, struct_rhs, binary_operator::LESS, dt);
  auto res_gt     = binary_operation(struct_lhs, struct_rhs, binary_operator::GREATER, dt);
  auto res_gteq   = binary_operation(struct_lhs, struct_rhs, binary_operator::GREATER_EQUAL, dt);
  auto res_lteq   = binary_operation(struct_lhs, struct_rhs, binary_operator::LESS_EQUAL, dt);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_eq, *expected_eq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_neq, *expected_neq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lt, *expected_lt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lteq, *expected_lteq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gt, *expected_gt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gteq, *expected_gteq);
};

TEST_F(BinopStructCompareNAN, double_nans)
{
  cudf::test::fixed_width_column_wrapper<double> lhs{
    -NAN, -NAN, -NAN, NAN, NAN, NAN, 1.0f, 0.0f, -54.3f};
  cudf::test::fixed_width_column_wrapper<double> rhs{
    -32.5f, -NAN, NAN, -0.0f, -NAN, NAN, 111.0f, -NAN, NAN};
  data_type dt = cudf::data_type(cudf::type_id::BOOL8);

  auto expected_eq   = binary_operation(lhs, rhs, binary_operator::EQUAL, dt);
  auto expected_neq  = binary_operation(lhs, rhs, binary_operator::NOT_EQUAL, dt);
  auto expected_lt   = binary_operation(lhs, rhs, binary_operator::LESS, dt);
  auto expected_gt   = binary_operation(lhs, rhs, binary_operator::GREATER, dt);
  auto expected_gteq = binary_operation(lhs, rhs, binary_operator::GREATER_EQUAL, dt);
  auto expected_lteq = binary_operation(lhs, rhs, binary_operator::LESS_EQUAL, dt);

  auto struct_lhs = structs_column_wrapper{lhs};
  auto struct_rhs = structs_column_wrapper{rhs};
  auto res_eq     = binary_operation(struct_lhs, struct_rhs, binary_operator::EQUAL, dt);
  auto res_neq    = binary_operation(struct_lhs, struct_rhs, binary_operator::NOT_EQUAL, dt);
  auto res_lt     = binary_operation(struct_lhs, struct_rhs, binary_operator::LESS, dt);
  auto res_gt     = binary_operation(struct_lhs, struct_rhs, binary_operator::GREATER, dt);
  auto res_gteq   = binary_operation(struct_lhs, struct_rhs, binary_operator::GREATER_EQUAL, dt);
  auto res_lteq   = binary_operation(struct_lhs, struct_rhs, binary_operator::LESS_EQUAL, dt);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_eq, *expected_eq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_neq, *expected_neq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lt, *expected_lt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_lteq, *expected_lteq);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gt, *expected_gt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*res_gteq, *expected_gteq);
};

struct BinopStructCompareFailures : public cudf::test::BaseFixture {
  void attempt_struct_binop(binary_operator op,
                            data_type dt = cudf::data_type(cudf::type_id::BOOL8))
  {
    auto col        = fixed_width_column_wrapper<uint32_t>{0, 89, 121};
    auto struct_col = structs_column_wrapper{col};
    binary_operation(struct_col, struct_col, op, dt);
  }
};

TEST_F(BinopStructCompareFailures, binopcompare_lists)
{
  auto list_col   = lists_column_wrapper<uint32_t>{{0, 0}, {127, 3, 55}, {7, 3}};
  auto struct_col = structs_column_wrapper{list_col};
  auto dt         = cudf::data_type(cudf::type_id::BOOL8);

  EXPECT_THROW(binary_operation(struct_col, struct_col, binary_operator::EQUAL, dt),
               cudf::logic_error);
  EXPECT_THROW(binary_operation(struct_col, struct_col, binary_operator::NOT_EQUAL, dt),
               cudf::logic_error);
  EXPECT_THROW(binary_operation(struct_col, struct_col, binary_operator::LESS, dt),
               cudf::logic_error);
  EXPECT_THROW(binary_operation(struct_col, struct_col, binary_operator::GREATER, dt),
               cudf::logic_error);
  EXPECT_THROW(binary_operation(struct_col, struct_col, binary_operator::GREATER_EQUAL, dt),
               cudf::logic_error);
  EXPECT_THROW(binary_operation(struct_col, struct_col, binary_operator::LESS_EQUAL, dt),
               cudf::logic_error);
}

TEST_F(BinopStructCompareFailures, binopcompare_unsupported_ops)
{
  EXPECT_THROW(attempt_struct_binop(binary_operator::ADD), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::SUB), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::MUL), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::DIV), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::TRUE_DIV), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::FLOOR_DIV), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::MOD), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::PMOD), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::PYMOD), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::POW), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::LOG_BASE), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::ATAN2), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::SHIFT_LEFT), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::SHIFT_RIGHT), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::SHIFT_RIGHT_UNSIGNED), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::BITWISE_AND), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::BITWISE_OR), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::BITWISE_XOR), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::LOGICAL_AND), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::LOGICAL_OR), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::NULL_EQUALS), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::NULL_MAX), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::NULL_MIN), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::GENERIC_BINARY), cudf::logic_error);
  EXPECT_THROW(attempt_struct_binop(binary_operator::INVALID_BINARY), cudf::logic_error);
}

}  // namespace cudf::test
