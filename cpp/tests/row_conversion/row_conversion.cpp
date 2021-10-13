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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/row_conversion.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <limits>

struct ColumnToRowTests : public cudf::test::BaseFixture {
};
struct RowToColumnTests : public cudf::test::BaseFixture {
};

TEST_F(ColumnToRowTests, Single)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1});
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Simple)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Tall)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  cudf::test::fixed_width_column_wrapper<int32_t> a(r, r + (size_t)4096);
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Wide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({rand()}));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, SingleByteWide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int8_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int8_t>({rand()}));
    views.push_back(cols.back());

    schema.push_back(cudf::data_type{cudf::type_id::INT8});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Non2Power)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  constexpr auto num_rows = 6 * 1024 + 557;
  for (int i = 0; i < 131; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    for (int j = 0; j < old_tbl->num_columns(); ++j) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(old_tbl->get_column(j), new_tbl->get_column(j));
    }

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Big)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 28 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 28; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    for (int j = 0; j < old_tbl->num_columns(); ++j) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(old_tbl->get_column(j), new_tbl->get_column(j));
    }

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Bigger)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 128 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    for (int j = 0; j < old_tbl->num_columns(); ++j) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(old_tbl->get_column(j), new_tbl->get_column(j));
    }

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Biggest)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 128 columns of 2 million rows
  constexpr auto num_rows = 2 * 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    for (int j = 0; j < old_tbl->num_columns(); ++j) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(old_tbl->get_column(j), new_tbl->get_column(j));
    }

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Single)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1});
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = cudf::convert_to_rows(in);
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32}};
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Simple)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32}};
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Tall)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  cudf::test::fixed_width_column_wrapper<int32_t> a(r, r + (size_t)4096);
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Wide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({i}));  // rand()}));
    views.push_back(cols.back());
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, SingleByteWide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int8_t>> cols;
  std::vector<cudf::column_view> views;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int8_t>({rand()}));
    views.push_back(cols.back());
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, AllTypes)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT64},
                                      cudf::data_type{cudf::type_id::FLOAT64},
                                      cudf::data_type{cudf::type_id::INT8},
                                      cudf::data_type{cudf::type_id::BOOL8},
                                      cudf::data_type{cudf::type_id::FLOAT32},
                                      cudf::data_type{cudf::type_id::INT8},
                                      cudf::data_type{cudf::type_id::INT32},
                                      cudf::data_type{cudf::type_id::INT64}};

  cudf::test::fixed_width_column_wrapper<int64_t> c0({3, 9, 4, 2, 20, 0}, {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<double> c1({5.0, 9.5, 0.9, 7.23, 2.8, 0.0},
                                                    {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<int8_t> c2({5, 1, 0, 2, 7, 0}, {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> c3({true, false, false, true, false, false},
                                                  {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<float> c4({1.0f, 3.5f, 5.9f, 7.1f, 9.8f, 0.0f},
                                                   {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<int8_t> c5({2, 3, 4, 5, 9, 0}, {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_point_column_wrapper<int32_t> c6(
    {-300, 500, 950, 90, 723, 0}, {1, 1, 1, 1, 1, 1, 1, 0}, numeric::scale_type{-2});
  cudf::test::fixed_point_column_wrapper<int64_t> c7(
    {-80, 30, 90, 20, 200, 0}, {1, 1, 1, 1, 1, 1, 0}, numeric::scale_type{-1});

  cudf::table_view in({c0, c1, c2, c3, c4, c5, c6, c7});

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, AllTypesLarge)
{
  std::vector<cudf::column> cols;
  std::vector<cudf::data_type> schema{};

  // 10 columns of each type with 1024 entries
  constexpr int num_rows{1024};

  std::default_random_engine re;
  std::uniform_real_distribution<double> rand_double(std::numeric_limits<double>::min(),
                                                     std::numeric_limits<double>::max());
  std::uniform_int_distribution<int64_t> rand_int64(std::numeric_limits<int64_t>::min(),
                                                    std::numeric_limits<int64_t>::max());
  auto r = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) -> int64_t { return rand_int64(re); });
  auto d = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) -> double { return rand_double(re); });

  for (int i = 0; i < 10; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<int8_t>(r, r + num_rows).release().release());
    schema.push_back(cudf::data_type{cudf::type_id::INT8});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<int16_t>(r, r + num_rows).release().release());
    schema.push_back(cudf::data_type{cudf::type_id::INT16});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<int32_t>(r, r + num_rows).release().release());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<float>(d, d + num_rows).release().release());
    schema.push_back(cudf::data_type{cudf::type_id::FLOAT32});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<double>(d, d + num_rows).release().release());
    schema.push_back(cudf::data_type{cudf::type_id::FLOAT64});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<bool>(r, r + num_rows).release().release());
    schema.push_back(cudf::data_type{cudf::type_id::BOOL8});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(
         r, r + num_rows)
         .release()
         .release());
    schema.push_back(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>(
         r, r + num_rows)
         .release()
         .release());
    schema.push_back(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(
      *cudf::test::fixed_point_column_wrapper<int32_t>(r, r + num_rows, numeric::scale_type{-2})
         .release()
         .release());
    schema.push_back(cudf::data_type{cudf::type_id::DECIMAL32});
  }

  for (int i = 0; i < 10; ++i) {
    cols.push_back(
      *cudf::test::fixed_point_column_wrapper<int64_t>(r, r + num_rows, numeric::scale_type{-1})
         .release()
         .release());
    schema.push_back(cudf::data_type{cudf::type_id::DECIMAL64});
  }

  std::vector<cudf::column_view> views(cols.begin(), cols.end());
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = cudf::convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Non2Power)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  constexpr auto num_rows = 6 * 1024 + 557;
  for (int i = 0; i < 131; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Big)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 28 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 28; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Bigger)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 28 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Biggest)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 28 columns of 1 million rows
  constexpr auto num_rows = 5 * 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = cudf::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl =
      cudf::convert_from_rows_fixed_width_optimized(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}
