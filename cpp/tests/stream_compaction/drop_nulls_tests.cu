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
  gdf_column output{};
  CUDF_EXPECT_NO_THROW(output = cudf::drop_nulls(bad_input));
  EXPECT_EQ(output.size, 0);
  EXPECT_EQ(output.null_count, 0);
  EXPECT_EQ(output.data, nullptr);
  EXPECT_EQ(output.valid, nullptr);

  bad_input.valid = reinterpret_cast<gdf_valid_type*>(0x0badf00d);
  bad_input.null_count = 1;
  bad_input.size = 2; 
  // nonzero, with non-null valid mask, so non-null input expected
  CUDF_EXPECT_THROW_MESSAGE(cudf::drop_nulls(bad_input), "Null input data");
}

/*
 * Runs drop_nulls checking for errors, and compares the result column 
 * to the specified expected result column.
 */
template <typename T>
void DropNulls(column_wrapper<T> const& source,
               column_wrapper<T> const& expected)
{
  gdf_column result{};
  EXPECT_NO_THROW(result = cudf::drop_nulls(source));
  EXPECT_EQ(result.null_count, 0);
  bool columns_equal{false};
  EXPECT_TRUE(columns_equal = (expected == result));

  if (!columns_equal) {
    std::cout << "expected\n";
    expected.print();
    std::cout << expected.get()->null_count << "\n";
    std::cout << "result\n";
    print_gdf_column(&result);
    std::cout << result.null_count << "\n";
  }

  gdf_column_free(&result);
}

constexpr gdf_size_type column_size{100000};

template <typename T>
struct DropNullsTest : GdfTest 
{
  cudf::test::column_wrapper_factory<T> factory;
};

using test_types =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double,
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
                    std::vector<gdf_index_type> const &column_indices,
                    cudf::table const &expected,
                    cudf::any_or_all drop_if = cudf::ANY)
{
  cudf::table result;
  EXPECT_NO_THROW(result = cudf::drop_nulls(source, column_indices, drop_if));

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

  std::vector<gdf_index_type> column_indices{0, 1, 2, 3};

  DropNullsTable(table_source, column_indices, table_expected);
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
  cudf::table table_expected(0, column_dtypes(table_source), true, false);

  std::vector<gdf_index_type> column_indices{0, 1, 2, 3};

  DropNullsTable(table_source, column_indices, table_expected);
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

  std::vector<gdf_index_type> column_indices{0, 1, 2, 3};

  DropNullsTable(table_source, column_indices, table_expected);
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

  std::vector<gdf_index_type> column_indices{0, 1, 2, 3};

  DropNullsTable(table_source, column_indices, table_expected, cudf::ANY);

  // nothing dropped if cudf::ALL is used for drop criteria since all columns
  // must be NULL to drop a row.
  DropNullsTable(table_source, column_indices, table_source, cudf::ALL);
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

  std::vector<gdf_index_type> column_indices{0, 1, 2, 3};

  DropNullsTable(table_source, column_indices, table_expected, cudf::ANY);

  // nothing dropped if cudf::ALL is used since all columns must be NULL to drop
  DropNullsTable(table_source, column_indices, table_source, cudf::ALL);

  // Test a subset of columns
  std::vector<gdf_index_type> subset_columns{1, 2};

  DropNullsTable(table_source, subset_columns, table_expected, cudf::ANY);
  // nothing dropped if cudf::ALL is used since all columns must be NULL to drop
  DropNullsTable(table_source, subset_columns, table_source, cudf::ALL);
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

  std::vector<gdf_index_type> column_indices{0, 1, 2, 3};

  DropNullsTable(table_source, column_indices, table_expected);
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

  std::vector<gdf_index_type> column_indices{0, 1, 2, 3};

  DropNullsTable(table_source, column_indices, table_expected);
}
