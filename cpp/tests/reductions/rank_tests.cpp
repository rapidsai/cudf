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

#include "scan_tests.hpp"

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/reduction.hpp>

using aggregation = cudf::aggregation;
using cudf::null_policy;
using cudf::scan_type;
using namespace cudf::test::iterators;

template <typename T>
struct TypedRankScanTest : BaseScanTest<T> {
  inline void test_ungrouped_rank_scan(cudf::column_view const& input,
                                       cudf::column_view const& expect_vals,
                                       std::unique_ptr<aggregation> const& agg,
                                       null_policy null_handling)
  {
    auto col_out = cudf::scan(input, agg, scan_type::INCLUSIVE, null_handling);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_vals, col_out->view());
  }
};

using RankTypes = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                     cudf::test::FloatingPointTypes,
                                     cudf::test::FixedPointTypes,
                                     cudf::test::ChronoTypes,
                                     cudf::test::StringTypes>;

TYPED_TEST_SUITE(TypedRankScanTest, RankTypes);

TYPED_TEST(TypedRankScanTest, Rank)
{
  auto const v = [] {
    if (std::is_signed<TypeParam>::value)
      return make_vector<TypeParam>({-120, -120, -120, -16, -16, 5, 6, 6, 6, 6, 34, 113});
    return make_vector<TypeParam>({5, 5, 5, 6, 6, 9, 11, 11, 11, 11, 14, 34});
  }();
  auto col = this->make_column(v);

  auto const expected_dense_vals =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5, 6};
  auto const expected_rank_vals =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 1, 4, 4, 6, 7, 7, 7, 7, 11, 12};
  this->test_ungrouped_rank_scan(
    *col, expected_dense_vals, cudf::make_dense_rank_aggregation(), null_policy::INCLUDE);
  this->test_ungrouped_rank_scan(
    *col, expected_rank_vals, cudf::make_rank_aggregation(), null_policy::INCLUDE);
}

TYPED_TEST(TypedRankScanTest, RankWithNulls)
{
  auto const v = [] {
    if (std::is_signed<TypeParam>::value)
      return make_vector<TypeParam>({-120, -120, -120, -16, -16, 5, 6, 6, 6, 6, 34, 113});
    return make_vector<TypeParam>({5, 5, 5, 6, 6, 9, 11, 11, 11, 11, 14, 34});
  }();
  auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0});
  auto col     = this->make_column(v, b);

  auto const expected_dense_vals =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 1, 2, 3, 4, 5, 5, 6, 6, 7, 8};
  auto const expected_rank_vals =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 1, 4, 5, 6, 7, 7, 9, 9, 11, 12};
  this->test_ungrouped_rank_scan(
    *col, expected_dense_vals, cudf::make_dense_rank_aggregation(), null_policy::INCLUDE);
  this->test_ungrouped_rank_scan(
    *col, expected_rank_vals, cudf::make_rank_aggregation(), null_policy::INCLUDE);
}

TYPED_TEST(TypedRankScanTest, MixedStructs)
{
  auto const v = [] {
    if (std::is_signed<TypeParam>::value)
      return make_vector<TypeParam>({-1, -1, -4, -4, -4, 5, 7, 7, 7, 9, 9, 9});
    return make_vector<TypeParam>({0, 0, 4, 4, 4, 5, 7, 7, 7, 9, 9, 9});
  }();
  auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1});
  auto col     = this->make_column(v, b);
  auto strings = cudf::test::strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};
  std::vector<std::unique_ptr<cudf::column>> vector_of_columns;
  vector_of_columns.push_back(std::move(col));
  vector_of_columns.push_back(strings.release());
  auto struct_col = cudf::test::structs_column_wrapper{std::move(vector_of_columns)}.release();

  auto expected_dense_vals =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 8};
  auto expected_rank_vals =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 3, 3, 5, 6, 7, 7, 9, 10, 10, 12};

  this->test_ungrouped_rank_scan(
    *struct_col, expected_dense_vals, cudf::make_dense_rank_aggregation(), null_policy::INCLUDE);
  this->test_ungrouped_rank_scan(
    *struct_col, expected_rank_vals, cudf::make_rank_aggregation(), null_policy::INCLUDE);
}

TYPED_TEST(TypedRankScanTest, NestedStructs)
{
  auto const v = [] {
    if (std::is_signed<TypeParam>::value)
      return make_vector<TypeParam>({-1, -1, -4, -4, -4, 5, 7, 7, 7, 9, 9, 9});
    return make_vector<TypeParam>({0, 0, 4, 4, 4, 5, 7, 7, 7, 9, 9, 9});
  }();
  auto const b  = thrust::host_vector<bool>(std::vector<bool>{1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1});
  auto col1     = this->make_column(v, b);
  auto col2     = this->make_column(v, b);
  auto col3     = this->make_column(v, b);
  auto col4     = this->make_column(v, b);
  auto strings1 = cudf::test::strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};
  auto strings2 = cudf::test::strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};

  std::vector<std::unique_ptr<cudf::column>> struct_columns;
  struct_columns.push_back(std::move(col1));
  struct_columns.push_back(strings1.release());
  auto struct_col = cudf::test::structs_column_wrapper{std::move(struct_columns)};
  std::vector<std::unique_ptr<cudf::column>> nested_columns;
  nested_columns.push_back(struct_col.release());
  nested_columns.push_back(std::move(col2));
  auto nested_col = cudf::test::structs_column_wrapper{std::move(nested_columns)};
  std::vector<std::unique_ptr<cudf::column>> flat_columns;
  flat_columns.push_back(std::move(col3));
  flat_columns.push_back(strings2.release());
  flat_columns.push_back(std::move(col4));
  auto flat_col = cudf::test::structs_column_wrapper{std::move(flat_columns)};

  auto dense_out = cudf::scan(
    nested_col, cudf::make_dense_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto dense_expected = cudf::scan(
    flat_col, cudf::make_dense_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto rank_out = cudf::scan(
    nested_col, cudf::make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto rank_expected =
    cudf::scan(flat_col, cudf::make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(dense_out->view(), dense_expected->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(rank_out->view(), rank_expected->view());
}

TYPED_TEST(TypedRankScanTest, structsWithNullPushdown)
{
  auto const v = [] {
    if (std::is_signed<TypeParam>::value)
      return make_vector<TypeParam>({-1, -1, -4, -4, -4, 5, 7, 7, 7, 9, 9, 9});
    return make_vector<TypeParam>({0, 0, 4, 4, 4, 5, 7, 7, 7, 9, 9, 9});
  }();
  auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1});
  auto col     = this->make_column(v, b);
  auto strings = cudf::test::strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};
  std::vector<std::unique_ptr<cudf::column>> struct_columns;
  struct_columns.push_back(std::move(col));
  struct_columns.push_back(strings.release());

  auto struct_col =
    cudf::make_structs_column(12, std::move(struct_columns), 0, rmm::device_buffer{});

  struct_col->set_null_mask(create_null_mask(12, cudf::mask_state::ALL_NULL));
  auto expected_null_result =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto dense_null_out = cudf::scan(
    *struct_col, cudf::make_dense_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto rank_null_out = cudf::scan(
    *struct_col, cudf::make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(dense_null_out->view(), expected_null_result);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(rank_null_out->view(), expected_null_result);

  auto const struct_nulls =
    thrust::host_vector<bool>(std::vector<bool>{1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  struct_col->set_null_mask(
    cudf::test::detail::make_null_mask(struct_nulls.begin(), struct_nulls.end()));
  auto expected_dense_vals =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 2, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9};
  auto expected_rank_vals =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 2, 2, 4, 5, 6, 7, 7, 9, 10, 10, 12};
  auto dense_out = cudf::scan(
    *struct_col, cudf::make_dense_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto rank_out = cudf::scan(
    *struct_col, cudf::make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(dense_out->view(), expected_dense_vals);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(rank_out->view(), expected_rank_vals);
}

struct RankScanTest : public cudf::test::BaseFixture {
};

TEST(RankScanTest, BoolRank)
{
  cudf::test::fixed_width_column_wrapper<bool> vals{0, 0, 0, 6, 6, 9, 11, 11, 11, 11, 14, 34};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_dense_vals{
    1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_rank_vals{
    1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4};

  auto dense_out = cudf::scan(
    vals, cudf::make_dense_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto rank_out =
    cudf::scan(vals, cudf::make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_dense_vals, dense_out->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_rank_vals, rank_out->view());
}

TEST(RankScanTest, BoolRankWithNull)
{
  cudf::test::fixed_width_column_wrapper<bool> vals{{0, 0, 0, 6, 6, 9, 11, 11, 11, 11, 14, 34},
                                                    {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}};
  cudf::table_view order_table{std::vector<cudf::column_view>{vals}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_dense_vals{
    1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_rank_vals{
    1, 1, 1, 4, 4, 4, 4, 4, 9, 9, 9, 9};

  auto nullable_dense_out = cudf::scan(
    vals, cudf::make_dense_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto nullable_rank_out =
    cudf::scan(vals, cudf::make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_dense_vals, nullable_dense_out->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_rank_vals, nullable_rank_out->view());
}

TEST(RankScanTest, ExclusiveScan)
{
  cudf::test::fixed_width_column_wrapper<uint32_t> vals{3, 4, 5};
  cudf::test::fixed_width_column_wrapper<uint32_t> order_col{3, 3, 1};
  cudf::table_view order_table{std::vector<cudf::column_view>{order_col}};

  CUDF_EXPECT_THROW_MESSAGE(
    cudf::scan(
      vals, cudf::make_dense_rank_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE),
    "Unsupported dense rank aggregation operator for exclusive scan");
  CUDF_EXPECT_THROW_MESSAGE(
    cudf::scan(vals, cudf::make_rank_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE),
    "Unsupported rank aggregation operator for exclusive scan");
}
