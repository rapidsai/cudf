/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Christian Noboa Mardini <christian@blazingdb.com>
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

#include "gtest/gtest.h"
#include "copying.hpp"
#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/copying/copying_test_helper.hpp"

struct SplitInputTest : GdfTest {};

TEST_F(SplitInputTest, IndexesNull) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create input column
  auto input_column = create_random_column<ColumnType>(SIZE);

  // Create indexes
  std::vector<gdf_index_type> indexes_host{SIZE / 2};

  // Create output
  std::vector<std::shared_ptr<cudf::test::column_wrapper<ColumnType>>> output_columns;
  auto source_columns =
      allocate_split_output_columns<ColumnType>(output_columns, indexes_host, SIZE);
  cudf::column_array column_array(source_columns.data(), source_columns.size());

  // Perform test
  ASSERT_ANY_THROW(cudf::split(input_column.get(), nullptr, &column_array));
}

TEST_F(SplitInputTest, InputColumnNull) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create indexes
  std::vector<gdf_index_type> indexes_host{SIZE / 2};
  cudf::test::column_wrapper<gdf_index_type> indexes(indexes_host);

  // Create output
  std::vector<std::shared_ptr<cudf::test::column_wrapper<ColumnType>>> output_columns;
  auto source_columns =
      allocate_split_output_columns<ColumnType>(output_columns, indexes_host, SIZE);
  cudf::column_array column_array(source_columns.data(), source_columns.size());

  // Perform test
  ASSERT_ANY_THROW(cudf::split(nullptr, indexes.get(), &column_array));
}

TEST_F(SplitInputTest, OutputColumnNull) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create input column
  auto input_column = create_random_column<ColumnType>(SIZE);

  // Create indexes
  std::vector<gdf_index_type> indexes_host{SIZE / 2};
  cudf::test::column_wrapper<gdf_index_type> indexes(indexes_host);

  // Perform test
  ASSERT_ANY_THROW(cudf::split(input_column.get(), indexes.get(), nullptr));
}

TEST_F(SplitInputTest, IndexesSizeNull) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create input column
  auto input_column = create_random_column<ColumnType>(SIZE);

  // Create indexes
  gdf_column indexes;
  indexes.size = 0;

  // Create output
  gdf_column column;
  gdf_column* source_columns[1] = { &column };
  cudf::column_array column_array(source_columns, 1);

  // Perform test
  ASSERT_NO_THROW(cudf::split(input_column.get(), &indexes, &column_array));
}

TEST_F(SplitInputTest, InputColumnSizeNull) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create input column
  gdf_column input_column;
  input_column.size = 0;

  // Create indexes
  std::vector<gdf_index_type> indexes_host{SIZE / 2};
  cudf::test::column_wrapper<gdf_index_type> indexes(indexes_host);

  // Create output
  gdf_column column;
  gdf_column* source_columns[1] = { &column };
  cudf::column_array column_array(source_columns, 1);

  // Perform test
  ASSERT_NO_THROW(cudf::split(&input_column, indexes.get(), &column_array));
}

TEST_F(SplitInputTest, IndexesDataNull) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create input column
  auto input_column = create_random_column<ColumnType>(SIZE);

  // Create indexes
  std::vector<gdf_index_type> indexes_host{SIZE / 2};
  cudf::test::column_wrapper<gdf_index_type> indexes(indexes_host);
  gdf_column* indexes_test = indexes.get();
  indexes_test->data = nullptr;

  // Create output
  std::vector<std::shared_ptr<cudf::test::column_wrapper<ColumnType>>> output_columns;
  auto source_columns =
      allocate_split_output_columns<ColumnType>(output_columns, indexes_host, SIZE);
  cudf::column_array column_array(source_columns.data(), source_columns.size());

  // Perform test
  ASSERT_ANY_THROW(cudf::split(input_column.get(), indexes_test, &column_array));
}

TEST_F(SplitInputTest, InputColumnDataNull) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create input column
  auto input_column = create_random_column<ColumnType>(SIZE);
  gdf_column* input_column_test = input_column.get();
  input_column_test->data = nullptr;

  // Create indexes
  std::vector<gdf_index_type> indexes_host{SIZE / 2};
  cudf::test::column_wrapper<gdf_index_type> indexes(indexes_host);

  // Create output
  std::vector<std::shared_ptr<cudf::test::column_wrapper<ColumnType>>> output_columns;
  auto source_columns =
      allocate_split_output_columns<ColumnType>(output_columns, indexes_host, SIZE);
  cudf::column_array column_array(source_columns.data(), source_columns.size());

  // Perform test
  ASSERT_ANY_THROW(cudf::split(input_column_test, indexes.get(), &column_array));
}

TEST_F(SplitInputTest, InputColumnBitmaskNull) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create input column
  auto input_column = create_random_column<ColumnType>(SIZE);
  gdf_column* input_column_test = input_column.get();
  input_column_test->valid = nullptr;

  // Create indexes
  std::vector<gdf_index_type> indexes_host{SIZE / 2};
  cudf::test::column_wrapper<gdf_index_type> indexes(indexes_host);

  // Create output
  std::vector<std::shared_ptr<cudf::test::column_wrapper<ColumnType>>> output_columns;
  auto source_columns =
      allocate_split_output_columns<ColumnType>(output_columns, indexes_host, SIZE);
  cudf::column_array column_array(source_columns.data(), source_columns.size());

  // Perform test
  ASSERT_ANY_THROW(cudf::split(input_column_test, indexes.get(), &column_array));
}

TEST_F(SplitInputTest, OutputColumnsAndIndexesSizeMismatch) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create input column
  auto input_column = create_random_column<ColumnType>(SIZE);

  // Create indexes
  std::vector<gdf_index_type> indexes_host{SIZE / 2};
  cudf::test::column_wrapper<gdf_index_type> indexes(indexes_host);

  // Create indexes for test
  std::vector<gdf_index_type> indexes_host_test{SIZE / 4, SIZE / 2};
  cudf::test::column_wrapper<gdf_index_type> indexes_test(indexes_host_test);

  // Create output
  std::vector<std::shared_ptr<cudf::test::column_wrapper<ColumnType>>> output_columns;
  auto source_columns =
      allocate_split_output_columns<ColumnType>(output_columns, indexes_host, SIZE);
  cudf::column_array column_array(source_columns.data(), source_columns.size());

  // Perform test
  ASSERT_ANY_THROW(cudf::split(input_column.get(), indexes_test.get(), &column_array));
}


template <typename ColumnType>
struct SplitTest : GdfTest {};

using test_types =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;
TYPED_TEST_CASE(SplitTest, test_types);

/**
 * It performs a parameterized type test, where the array of indices contains
 * multiple values.
 *
 * It tests:
 * when the indices contain limit values (0 and INPUT_SIZE).
 * when the indices are the same.
 * when is less than 16 or greater than 64.
 */
TYPED_TEST(SplitTest, MultipleSplits) {
  // Create input column
  auto input_column = create_random_column<TypeParam>(INPUT_SIZE);

  // Create indexes
  std::vector<gdf_index_type> indexes_host{0, 13, 31, 31, 32, INPUT_SIZE};
  cudf::test::column_wrapper<gdf_index_type> indexes(indexes_host);

  // Create output
  std::vector<std::shared_ptr<cudf::test::column_wrapper<TypeParam>>> output_columns;
  auto source_columns =
      allocate_split_output_columns<TypeParam>(output_columns, indexes_host, INPUT_SIZE);
  cudf::column_array column_array(source_columns.data(), source_columns.size());

  // Perform operation
  ASSERT_NO_THROW(cudf::split(input_column.get(), indexes.get(), &column_array));

  // Transfer input column to host
  auto input_column_host = makeHelperColumn<TypeParam>(input_column);

  // Transfer output columns to host
  auto output_column_host = makeHelperColumn<TypeParam>(output_columns);

  // Perform split in cpu
  auto output_column_cpu = split_columns<TypeParam>(input_column_host,
                                                    indexes_host,
                                                    INPUT_SIZE);

  // Verify the operation
  for (std::size_t i = 0; i < output_column_host.size(); ++i) {
    verify<TypeParam>(output_column_cpu[i], output_column_host[i]);
  }
}

/**
 * It performs a parameterized type and a parameterized value test.
 * The indices array contains only one value and that value goes from
 * 0 to the size of the input column.
 */
TYPED_TEST(SplitTest, RangeIndexPosition) {
  // Test parameters
  constexpr gdf_index_type init_index{0};
  constexpr gdf_index_type final_index{INPUT_SIZE};

  // Create input column
  auto input_column = create_random_column<TypeParam>(INPUT_SIZE);
  
  for (gdf_index_type index = init_index; index < final_index; ++index) {
    // Create indexes
    std::vector<gdf_index_type> indexes_host{index};
    cudf::test::column_wrapper<gdf_index_type> indexes(indexes_host);
    
    // Create output
    std::vector<std::shared_ptr<cudf::test::column_wrapper<TypeParam>>> output_columns;
    auto source_columns =
        allocate_split_output_columns<TypeParam>(output_columns, indexes_host, INPUT_SIZE);
    cudf::column_array column_array(source_columns.data(), source_columns.size());
    
    // Perform operation
    ASSERT_NO_THROW(cudf::split(input_column.get(), indexes.get(), &column_array));

    // Transfer input column to host
    auto input_column_host = makeHelperColumn<TypeParam>(input_column);

    // Transfer output columns to host
    auto output_column_host = makeHelperColumn<TypeParam>(output_columns);

    // Perform split in cpu
    auto output_column_cpu = split_columns<TypeParam>(input_column_host,
                                                      indexes_host,
                                                      INPUT_SIZE);

    // Verify columns
    for (std::size_t i = 0; i < output_column_host.size(); ++i) {
      verify<TypeParam>(output_column_cpu[i], output_column_host[i]);
    }
  }
}
