/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/equality.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_buffer.hpp>

#include <limits>
#include <memory>
#include <stdexcept>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

using s_col_wrapper       = cudf::test::strings_column_wrapper;
using lists_col_wrapper   = cudf::test::lists_column_wrapper<int32_t>;
using structs_col_wrapper = cudf::test::structs_column_wrapper;

using CVector     = std::vector<std::unique_ptr<cudf::column>>;
using column      = cudf::column;
using column_view = cudf::column_view;
using TView       = cudf::table_view;
using Table       = cudf::table;

struct TableTest : public cudf::test::BaseFixture {};

TEST_F(TableTest, EmptyColumnedTable)
{
  std::vector<column_view> cols{};

  TView input(cols);
  cudf::size_type expected = 0;

  EXPECT_EQ(input.num_columns(), expected);
}

TEST_F(TableTest, ValidateConstructorTableViewToTable)
{
  column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  column_wrapper<int8_t> col2{{1, 2, 3, 4}};

  CVector cols;
  cols.push_back(col1.release());
  cols.push_back(col2.release());

  Table input_table(std::move(cols));

  Table out_table(input_table.view());

  EXPECT_EQ(input_table.num_columns(), out_table.num_columns());
  EXPECT_EQ(input_table.num_rows(), out_table.num_rows());
}

TEST_F(TableTest, GetTableWithSelectedColumns)
{
  column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  column_wrapper<int16_t> col2{{1, 2, 3, 4}};
  column_wrapper<int32_t> col3{{4, 5, 6, 7}};
  column_wrapper<int64_t> col4{{4, 5, 6, 7}};

  CVector cols;
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  cols.push_back(col3.release());
  cols.push_back(col4.release());

  Table t(std::move(cols));

  cudf::table_view selected_tview = t.select(std::vector<cudf::size_type>{2, 3});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(t.view().column(2), selected_tview.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(t.view().column(3), selected_tview.column(1));
}

TEST_F(TableTest, SelectingOutOfBounds)
{
  column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  column_wrapper<int16_t> col2{{1, 2, 3, 4}};

  CVector cols;
  cols.push_back(col1.release());
  cols.push_back(col2.release());

  Table t(std::move(cols));

  EXPECT_THROW(cudf::table_view selected_tview = t.select(std::vector<cudf::size_type>{0, 1, 2}),
               std::out_of_range);
}

TEST_F(TableTest, SelectingNoColumns)
{
  column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  column_wrapper<int16_t> col2{{1, 2, 3, 4}};

  CVector cols;
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  Table t(std::move(cols));
  TView selected_table = t.select(std::vector<cudf::size_type>{});

  EXPECT_EQ(selected_table.num_columns(), 0);
}

TEST_F(TableTest, CreateFromViewVector)
{
  column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  column_wrapper<int16_t> col2{{1, 2, 3, 4}};

  std::vector<TView> views;
  views.emplace_back(std::vector<column_view>{col1});
  views.emplace_back(std::vector<column_view>{col2});
  TView final_view{views};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(final_view.column(0), views[0].column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(final_view.column(1), views[1].column(0));
}

TEST_F(TableTest, CreateFromViewVectorRowsMismatch)
{
  column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  column_wrapper<int16_t> col2{{1, 2, 3}};

  std::vector<TView> views;
  views.emplace_back(std::vector<column_view>{col1});
  views.emplace_back(std::vector<column_view>{col2});
  EXPECT_THROW(TView{views}, cudf::logic_error);
}

TEST_F(TableTest, CreateFromViewVectorEmptyTables)
{
  std::vector<TView> views;
  views.emplace_back(std::vector<column_view>{});
  views.emplace_back(std::vector<column_view>{});
  TView final_view{views};
  EXPECT_EQ(final_view.num_columns(), 0);
}

TEST_F(TableTest, AllocSize)
{
  column_wrapper<int32_t> col1{{1, 2, 3, 4}};
  column_wrapper<int16_t> col2{{1, 2, 3, 4}};

  CVector cols;
  cols.push_back(col1.release());
  cols.push_back(col2.release());

  Table t(std::move(cols));
  EXPECT_EQ(t.alloc_size(), 24);
}

TEST_F(TableTest, AllocSizeWithNulls)
{
  column_wrapper<int32_t> col1{{1, 2, 3, 4}, {1, 0, 1, 0}};
  column_wrapper<int16_t> col2{{1, 2, 3, 4}, {1, 0, 1, 0}};

  CVector cols;
  cols.push_back(col1.release());
  cols.push_back(col2.release());

  Table t(std::move(cols));
  EXPECT_EQ(t.alloc_size(), 152);  // bitmask has padding
}

TEST_F(TableTest, TablesEqual)
{
  column_wrapper<int32_t> left_col0{{1, 2, 3}};
  column_wrapper<double> left_col1{{4.0, 5.0, 6.0}};
  column_wrapper<int32_t> right_col0{{1, 2, 3}};
  column_wrapper<double> right_col1{{4.0, 5.0, 6.0}};

  EXPECT_TRUE(cudf::tables_equal(cudf::table_view{{left_col0, left_col1}},
                                 cudf::table_view{{right_col0, right_col1}}));
}

TEST_F(TableTest, TablesEqualValueMismatch)
{
  column_wrapper<int32_t> left{{1, 2, 3}};
  column_wrapper<int32_t> right{{1, 4, 3}};

  EXPECT_FALSE(cudf::tables_equal(cudf::table_view{{left}}, cudf::table_view{{right}}));
}

TEST_F(TableTest, TablesEqualShapeAndTypeMismatch)
{
  column_wrapper<int32_t> left{{1, 2, 3}};
  column_wrapper<int32_t> shorter{{1, 2}};
  column_wrapper<int32_t> extra{{1, 2, 3}};
  column_wrapper<int64_t> different_type{{1, 2, 3}};

  EXPECT_FALSE(cudf::tables_equal(cudf::table_view{{left}}, cudf::table_view{{shorter}}));
  EXPECT_FALSE(cudf::tables_equal(cudf::table_view{{left}}, cudf::table_view{{left, extra}}));
  EXPECT_FALSE(cudf::tables_equal(cudf::table_view{{left}}, cudf::table_view{{different_type}}));
}

TEST_F(TableTest, TablesEqualNullEquality)
{
  column_wrapper<int32_t> left{{1, 2, 3}, {1, 0, 1}};
  column_wrapper<int32_t> right{{1, 4, 3}, {1, 0, 1}};

  EXPECT_TRUE(cudf::tables_equal(
    cudf::table_view{{left}}, cudf::table_view{{right}}, cudf::null_equality::EQUAL));
  EXPECT_FALSE(cudf::tables_equal(
    cudf::table_view{{left}}, cudf::table_view{{right}}, cudf::null_equality::UNEQUAL));
}

TEST_F(TableTest, TablesEqualNaNsCompareEqual)
{
  column_wrapper<double> left{{std::numeric_limits<double>::quiet_NaN(), 1.0}};
  column_wrapper<double> right{{std::numeric_limits<double>::quiet_NaN(), 1.0}};

  EXPECT_TRUE(cudf::tables_equal(cudf::table_view{{left}}, cudf::table_view{{right}}));
}

TEST_F(TableTest, TablesEqualStructColumns)
{
  column_wrapper<int32_t> left_id{{1, 2, 3}};
  column_wrapper<int16_t> left_inner_value{{10, 20, 30}};
  column_wrapper<double> left_deep_leaf{{1.25, 2.5, 3.75}};
  structs_col_wrapper left_inner{{left_inner_value, left_deep_leaf}};
  structs_col_wrapper left_outer{{left_id, left_inner}};

  column_wrapper<int32_t> right_id{{1, 2, 3}};
  column_wrapper<int16_t> right_inner_value{{10, 20, 30}};
  column_wrapper<double> right_deep_leaf{{1.25, 2.5, 3.75}};
  structs_col_wrapper right_inner{{right_inner_value, right_deep_leaf}};
  structs_col_wrapper right_outer{{right_id, right_inner}};

  EXPECT_TRUE(cudf::tables_equal(cudf::table_view{{left_outer}}, cudf::table_view{{right_outer}}));
}

TEST_F(TableTest, TablesEqualStructColumnsDeepLeafMismatch)
{
  column_wrapper<int32_t> left_id{{1, 2, 3}};
  column_wrapper<int16_t> left_inner_value{{10, 20, 30}};
  column_wrapper<double> left_deep_leaf{{1.25, 2.5, 3.75}};
  structs_col_wrapper left_inner{{left_inner_value, left_deep_leaf}};
  structs_col_wrapper left_outer{{left_id, left_inner}};

  column_wrapper<int32_t> right_id{{1, 2, 3}};
  column_wrapper<int16_t> right_inner_value{{10, 20, 30}};
  column_wrapper<double> right_deep_leaf{{1.25, 2.5, 99.0}};
  structs_col_wrapper right_inner{{right_inner_value, right_deep_leaf}};
  structs_col_wrapper right_outer{{right_id, right_inner}};

  EXPECT_FALSE(cudf::tables_equal(cudf::table_view{{left_outer}}, cudf::table_view{{right_outer}}));
}

TEST_F(TableTest, TablesEqualThrowsForNonEqualityComparableTypes)
{
  auto left =
    column{cudf::data_type{cudf::type_id::EMPTY}, 3, rmm::device_buffer{}, rmm::device_buffer{}, 0};
  auto right =
    column{cudf::data_type{cudf::type_id::EMPTY}, 3, rmm::device_buffer{}, rmm::device_buffer{}, 0};

  EXPECT_THROW(
    std::ignore = cudf::tables_equal(cudf::table_view{{left}}, cudf::table_view{{right}}),
    cudf::logic_error);
}

TEST_F(TableTest, TablesEqualListColumns)
{
  lists_col_wrapper left{{1, 2}, {3}, {}};
  lists_col_wrapper right{{1, 2}, {3}, {}};
  lists_col_wrapper different_values{{1, 2}, {4}, {}};
  lists_col_wrapper different_offsets{{1}, {2, 3}, {}};

  EXPECT_TRUE(cudf::tables_equal(cudf::table_view{{left}}, cudf::table_view{{right}}));
  EXPECT_FALSE(cudf::tables_equal(cudf::table_view{{left}}, cudf::table_view{{different_values}}));
  EXPECT_FALSE(cudf::tables_equal(cudf::table_view{{left}}, cudf::table_view{{different_offsets}}));
}

TEST_F(TableTest, TablesEqualStructColumnsWithLists)
{
  column_wrapper<int32_t> left_id{{1, 2, 3}};
  lists_col_wrapper left_list{{1, 2}, {3}, {}};
  structs_col_wrapper left{{left_id, left_list}};

  column_wrapper<int32_t> right_id{{1, 2, 3}};
  lists_col_wrapper right_list{{1, 2}, {3}, {}};
  structs_col_wrapper right{{right_id, right_list}};

  column_wrapper<int32_t> different_id{{1, 2, 3}};
  lists_col_wrapper different_list{{1, 2}, {4}, {}};
  structs_col_wrapper different{{different_id, different_list}};

  EXPECT_TRUE(cudf::tables_equal(cudf::table_view{{left}}, cudf::table_view{{right}}));
  EXPECT_FALSE(cudf::tables_equal(cudf::table_view{{left}}, cudf::table_view{{different}}));
}

CUDF_TEST_PROGRAM_MAIN()
