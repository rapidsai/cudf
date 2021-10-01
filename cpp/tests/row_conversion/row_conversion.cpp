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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <cudf/row_conversion.hpp>
#include "cudf/lists/lists_column_view.hpp"
#include "cudf/types.hpp"

struct ColumnToRowTests : public cudf::test::BaseFixture {
};
struct RowToColumnTests : public cudf::test::BaseFixture {
};

TEST_F(ColumnToRowTests, Single)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1});
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = cudf::old_convert_to_rows(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }

  for (uint i = 0; i < old_rows.size(); i++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*old_rows[i], *new_rows[i]);
  }
}

TEST_F(ColumnToRowTests, Simple)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = cudf::old_convert_to_rows(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }

  for (uint i = 0; i < old_rows.size(); i++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*old_rows[i], *new_rows[i]);
  }
}

TEST_F(ColumnToRowTests, Tall)
{
  auto r =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t { return rand(); });
  cudf::test::fixed_width_column_wrapper<int32_t> a(r, r + (size_t)4096);
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = cudf::old_convert_to_rows(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }

  for (uint i = 0; i < old_rows.size(); i++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*old_rows[i], *new_rows[i]);
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

  auto old_rows = cudf::old_convert_to_rows(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }

  for (uint i = 0; i < old_rows.size(); i++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*old_rows[i], *new_rows[i]);
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

  auto old_rows = cudf::old_convert_to_rows(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }

  for (uint i = 0; i < old_rows.size(); i++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*old_rows[i], *new_rows[i]);
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

  auto old_rows = cudf::old_convert_to_rows(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    for (int j = 0; j < old_tbl->num_columns(); ++j) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(old_tbl->get_column(j), new_tbl->get_column(j));
    }

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }

  for (uint i = 0; i < old_rows.size(); i++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*old_rows[i], *new_rows[i]);
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

  auto old_rows = cudf::old_convert_to_rows(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); i++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*old_rows[i], *new_rows[i]);
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

  auto old_rows = cudf::old_convert_to_rows(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); i++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*old_rows[i], *new_rows[i]);
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

  auto old_rows = cudf::old_convert_to_rows(in);
  auto new_rows = cudf::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); i++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*old_rows[i], *new_rows[i]);
  }
}

TEST_F(RowToColumnTests, Single)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1});
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = cudf::convert_to_rows(in);
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32}};
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Simple)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = cudf::old_convert_to_rows(in);
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32}};
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
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

  auto old_rows = cudf::old_convert_to_rows(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
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

  auto old_rows = cudf::old_convert_to_rows(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
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

  auto old_rows = cudf::old_convert_to_rows(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

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

  auto old_rows = cudf::old_convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
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

  auto old_rows = cudf::old_convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
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

  auto old_rows = cudf::old_convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
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

  auto old_rows = cudf::old_convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = cudf::old_convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl = cudf::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}
