/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/stream_compaction.hpp>
#include <cudf/copying.hpp>

#include <utilities/error_utils.hpp>

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/column_wrapper_factory.hpp>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>

#include <sstream>

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

struct DropNullsErrorTest : GdfTest {};

TEST_F(DropNullsErrorTest, EmptyInput)
{
  gdf_column bad_input{};
  gdf_column_view(&bad_input, 0, 0, 0, GDF_INT32);

  // zero size, so expect no error, just empty output column
  cudf::table output;
  CUDF_EXPECT_NO_THROW(output = cudf::drop_nulls({&bad_input}, {&bad_input}));
  EXPECT_EQ(output.num_columns(), 1);
  EXPECT_EQ(output.num_rows(), 0);
  EXPECT_EQ(output.get_column(0)->null_count, 0);

  bad_input.valid = reinterpret_cast<gdf_valid_type*>(0x0badf00d);
  bad_input.null_count = 1;
  bad_input.size = 2; 

  // nonzero, with non-null valid mask, so non-null input expected
  CUDF_EXPECT_THROW_MESSAGE(cudf::drop_nulls({&bad_input}, {&bad_input}),
                            "Null input data");

  gdf_column bad_keys{};
  gdf_column_view(&bad_input, 0, 0, 0, GDF_INT32);
  bad_keys.valid = reinterpret_cast<gdf_valid_type*>(0x0badf00d);
  bad_keys.null_count = 1;
  bad_keys.size = 1;

  // keys size smaller than table size
  CUDF_EXPECT_THROW_MESSAGE(cudf::drop_nulls({&bad_input}, {&bad_keys}),
                            "Column size mismatch");
}

/*
 * Runs drop_nulls checking for errors, and compares the result column 
 * to the specified expected result column.
 */
template <typename T>
void DropNulls(column_wrapper<T> const& source,
               column_wrapper<T> const& expected)
{
  cudf::table result;
  cudf::table source_table{const_cast<gdf_column*>(source.get())};
  EXPECT_NO_THROW(result = cudf::drop_nulls(source_table, source_table));
  gdf_column *res = result.get_column(0);
  EXPECT_EQ(res->null_count, 0);
  bool columns_equal{false};
  EXPECT_TRUE(columns_equal = (expected == *res));

  if (!columns_equal) {
    std::cout << "expected\n";
    expected.print();
    std::cout << expected.get()->null_count << "\n";
    std::cout << "result\n";
    print_gdf_column(res);
    std::cout << res->null_count << "\n";
  }

  result.destroy();
}

constexpr gdf_size_type column_size{100};

template <typename T>
struct DropNullsTest : GdfTest 
{
  cudf::test::column_wrapper_factory<T> factory;
};

using test_types =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double,
                     cudf::timestamp, cudf::date32, cudf::date64,
                     cudf::bool8, cudf::nvstring_category>;
TYPED_TEST_CASE(DropNullsTest, test_types);

TYPED_TEST(DropNullsTest, Identity)
{
  auto col = this->factory.make(column_size,
    [](gdf_index_type row) { return row; },
    [](gdf_index_type row) { return true; });
  DropNulls<TypeParam>(col, col);
}

TYPED_TEST(DropNullsTest, AllNull)
{
  DropNulls<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return false; }),
    column_wrapper<TypeParam>(0, false));
}

TYPED_TEST(DropNullsTest, EvensNull)
{
  DropNulls<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return row % 2 == 1; }),
    this->factory.make(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1; },
      [](gdf_index_type row) { return true; }));
}

TYPED_TEST(DropNullsTest, NonalignedGap)
{
  const int start{1}, end{column_size / 4};

  DropNulls<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [&](gdf_index_type row) { return (row < start) || (row >= end); }),
    this->factory.make(column_size - (end - start),
      [&](gdf_index_type row) { 
        return (row < start) ? row : row + end - start;
      },
      [](gdf_index_type row) { return true; }));
}

TYPED_TEST(DropNullsTest, NoNullMask)
{
  DropNulls<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; }),
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; }));
}

struct DropNullsTableTest : GdfTest {};

static cudf::test::column_wrapper_factory<cudf::nvstring_category> string_factory;

/*
 * Runs drop_nulls checking for errors, and compares the result column 
 * to the specified expected result column.
 */
void DropNullsTable(cudf::table const &source,
                    cudf::table const &keys,
                    cudf::table const &expected,
                    gdf_size_type keep_thresh = -1)
{
  cudf::table result;
  if (keep_thresh >= 0)
    EXPECT_NO_THROW(result = cudf::drop_nulls(source, keys, keep_thresh));
  else 
    EXPECT_NO_THROW(result = cudf::drop_nulls(source, keys));

  for (int c = 0; c < result.num_columns(); c++) {
    gdf_column *res = result.get_column(c);
    gdf_column const *exp = expected.get_column(c);
    EXPECT_EQ(res->null_count, exp->null_count);
    bool columns_equal{false};
    EXPECT_TRUE(columns_equal = gdf_equal_columns(*res, *exp));

    if (!columns_equal) {
      std:: cout << "Column " << c << "\n";
      std::cout << "expected\n";
      print_gdf_column(exp);
      std::cout << exp->null_count << "\n";
      std::cout << "result\n";
      print_gdf_column(res);
      std::cout << res->null_count << "\n";
    }
  }
  result.destroy();
}

TEST_F(DropNullsTableTest, Identity)
{
  cudf::test::column_wrapper<int32_t> int_column(
      column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<float> float_column(
      column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<cudf::bool8> bool_column(
      column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<cudf::nvstring_category> string_column =
    string_factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; });


  std::vector<gdf_column*> cols;
  cols.push_back(int_column.get());
  cols.push_back(float_column.get());
  cols.push_back(bool_column.get());
  cols.push_back(string_column.get());
  cudf::table table_source(cols);
  cudf::table table_expected(cols);

  cudf::table table_keys(cols);

  DropNullsTable(table_source, table_source, table_expected);
}

TEST_F(DropNullsTableTest, AllNull)
{
  cudf::test::column_wrapper<int32_t> int_column(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return false; });
  cudf::test::column_wrapper<float> float_column(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return false; });
  cudf::test::column_wrapper<cudf::bool8> bool_column(column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return false; });
  cudf::test::column_wrapper<cudf::nvstring_category> string_column =
    string_factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return false; });
    
  std::vector<gdf_column*> cols;
  cols.push_back(int_column.get());
  cols.push_back(float_column.get());
  cols.push_back(bool_column.get());
  cols.push_back(string_column.get());
  cudf::table table_source(cols);
  cudf::table table_expected(0, column_dtypes(table_source),
                             column_dtype_infos(table_source), true, false);

  DropNullsTable(table_source, table_source, table_expected);
}

TEST_F(DropNullsTableTest, EvensNull)
{
  cudf::test::column_wrapper<int32_t> int_column(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return row % 2 != 0; });
  cudf::test::column_wrapper<float> float_column(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return row % 2 != 0; });
  cudf::test::column_wrapper<cudf::bool8> bool_column(column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return row % 2 != 0; });
  cudf::test::column_wrapper<cudf::nvstring_category> string_column =
    string_factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return row % 2 != 0; });

  std::vector<gdf_column*> cols;
  cols.push_back(int_column.get());
  cols.push_back(float_column.get());
  cols.push_back(bool_column.get());
  cols.push_back(string_column.get());
  cudf::table table_source(cols);

  cudf::test::column_wrapper<int32_t> int_expected(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1;  },
      [](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<float> float_expected(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1;  },
      [](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<cudf::bool8> bool_expected(column_size / 2,
      [](gdf_index_type row) { return cudf::bool8{true};  },
      [](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<cudf::nvstring_category> string_expected =
    string_factory.make(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1; },
      [](gdf_index_type row) { return true; });
  
  std::vector<gdf_column*> cols_expected;
  cols_expected.push_back(int_expected.get());
  cols_expected.push_back(float_expected.get());
  cols_expected.push_back(bool_expected.get());
  cols_expected.push_back(string_expected.get());
  cudf::table table_expected(cols_expected);

  DropNullsTable(table_source, table_source, table_expected);
}

TEST_F(DropNullsTableTest, OneColumnEvensNull)
{
  cudf::test::column_wrapper<int32_t> int_column(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<float> float_column(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return row % 2 != 0; });
  cudf::test::column_wrapper<cudf::bool8> bool_column(column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<cudf::nvstring_category> string_column =
    string_factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; });

  std::vector<gdf_column*> cols;
  cols.push_back(int_column.get());
  cols.push_back(float_column.get());
  cols.push_back(bool_column.get());
  cols.push_back(string_column.get());
  cudf::table table_source(cols);

  cudf::test::column_wrapper<int32_t> int_expected(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1;  },
      [](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<float> float_expected(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1;  },
      [](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<cudf::bool8> bool_expected(column_size / 2,
      [](gdf_index_type row) { return cudf::bool8{true};  },
      [](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<cudf::nvstring_category> string_expected =
    string_factory.make(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1; },
      [](gdf_index_type row) { return true; });
  
  std::vector<gdf_column*> cols_expected;
  cols_expected.push_back(int_expected.get());
  cols_expected.push_back(float_expected.get());
  cols_expected.push_back(bool_expected.get());
  cols_expected.push_back(string_expected.get());
  cudf::table table_expected(cols_expected);

  DropNullsTable(table_source, table_source, table_expected);

  // nothing dropped if 0 is used for keep_threshold since all columns
  // must be NULL to drop a row.
  DropNullsTable(table_source, table_source, table_source, 0);
}

TEST_F(DropNullsTableTest, SomeNullMasks)
{
  cudf::test::column_wrapper<int32_t> int_column(column_size,
      [](gdf_index_type row) { return row; }, false);
  cudf::test::column_wrapper<float> float_column(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return row % 2 != 0; });
  cudf::test::column_wrapper<cudf::bool8> bool_column(column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; }, false);
  cudf::test::column_wrapper<cudf::nvstring_category> string_column =
    string_factory.make(column_size,
      [](gdf_index_type row) { return row; });

  std::vector<gdf_column*> cols;
  cols.push_back(int_column.get());
  cols.push_back(float_column.get());
  cols.push_back(bool_column.get());
  cols.push_back(string_column.get());
  cudf::table table_source(cols);

  cudf::test::column_wrapper<int32_t> int_expected(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1;  }, false);
  cudf::test::column_wrapper<float> float_expected(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1;  },
      [](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<cudf::bool8> bool_expected(column_size / 2,
      [](gdf_index_type row) { return cudf::bool8{true};  }, false);
  cudf::test::column_wrapper<cudf::nvstring_category> string_expected =
    string_factory.make(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1; });
  
  std::vector<gdf_column*> cols_expected;
  cols_expected.push_back(int_expected.get());
  cols_expected.push_back(float_expected.get());
  cols_expected.push_back(bool_expected.get());
  cols_expected.push_back(string_expected.get());
  cudf::table table_expected(cols_expected);

  DropNullsTable(table_source, table_source, table_expected);

  // nothing dropped if 0 threshold is used since all columns must be NULL to drop
  DropNullsTable(table_source, table_source, table_source, 0);

  // test threshold -- at least one valid value to keep, so all valid
  DropNullsTable(table_source, table_source, table_source, 1);

  // test threshold -- at least 4 valid values, so should match default
  DropNullsTable(table_source, table_source, table_expected, 4);

  // Test a subset of columns
  cudf::table subset_columns{float_column, bool_column};

  DropNullsTable(table_source, subset_columns, table_expected);
  // nothing dropped if 0 is used for keep_threshold since all columns must be NULL to drop
  DropNullsTable(table_source, subset_columns, table_source, 0);

  // thresh of 1 valid means we keep all rows when we only consider float_column
  DropNullsTable(table_source, subset_columns, table_source, 1);

  // thresh of 1 valid means we drop half the rows when we only consider float_column
  cudf::table subset_columns2{float_column}; // 50% nulls
  DropNullsTable(table_source, subset_columns2, table_expected, 1);
}

TEST_F(DropNullsTableTest, NonalignedGap)
{
  const int start{1}, end{column_size / 4};

  cudf::test::column_wrapper<int32_t> int_column(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return (row < start) || (row >= end); });
  cudf::test::column_wrapper<float> float_column(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return (row < start) || (row >= end); });
  cudf::test::column_wrapper<cudf::bool8> bool_column(column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return (row < start) || (row >= end); });
  cudf::test::column_wrapper<cudf::nvstring_category> string_column =
    string_factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return (row < start) || (row >= end); });

  std::vector<gdf_column*> cols;
  cols.push_back(int_column.get());
  cols.push_back(float_column.get());
  cols.push_back(bool_column.get());
  cols.push_back(string_column.get());
  cudf::table table_source(cols);

  cudf::test::column_wrapper<int32_t> int_expected(column_size - (end - start),
      [](gdf_index_type row) { return (row < start) ? row : row + end - start; },
      [&](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<float> float_expected(column_size - (end - start),
      [](gdf_index_type row) { return (row < start) ? row : row + end - start; },
      [&](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<cudf::bool8> bool_expected(column_size - (end - start),
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [&](gdf_index_type row) { return true; });
  cudf::test::column_wrapper<cudf::nvstring_category> string_expected =
    string_factory.make(column_size - (end - start),
      [](gdf_index_type row) { return (row < start) ? row : row + end - start; },
      [&](gdf_index_type row) { return true; });
  
  std::vector<gdf_column*> cols_expected;
  cols_expected.push_back(int_expected.get());
  cols_expected.push_back(float_expected.get());
  cols_expected.push_back(bool_expected.get());
  cols_expected.push_back(string_expected.get());
  cudf::table table_expected(cols_expected);

  DropNullsTable(table_source, table_source, table_expected);
}

TEST_F(DropNullsTableTest, NoNullMask)
{
  cudf::test::column_wrapper<int32_t> int_column(column_size,
      [](gdf_index_type row) { return row; }, false);
  cudf::test::column_wrapper<float> float_column(column_size,
      [](gdf_index_type row) { return row; }, false);
  cudf::test::column_wrapper<cudf::bool8> bool_column(column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; }, false);
  cudf::test::column_wrapper<cudf::nvstring_category> string_column =
    string_factory.make(column_size,
      [](gdf_index_type row) { return row; });

  std::vector<gdf_column*> cols;
  cols.push_back(int_column.get());
  cols.push_back(float_column.get());
  cols.push_back(bool_column.get());
  cols.push_back(string_column.get());
  cudf::table table_source(cols);
  cudf::table table_expected(cols);

  DropNullsTable(table_source, table_source, table_expected);
}
