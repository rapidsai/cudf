/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cudf/interop.hpp>

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>

#include <memory>
#include <utility>

struct ArrowColumnTest : public cudf::test::BaseFixture {};

template <typename T>
auto export_to_arrow(T& obj)
{
  // Now we can extract an ArrowDeviceArray from the arrow_column
  auto schema = std::make_unique<ArrowSchema>();
  obj.to_arrow_schema(schema.get());
  auto array = std::make_unique<ArrowDeviceArray>();
  obj.to_arrow(array.get(), ARROW_DEVICE_CUDA);
  return std::make_pair(std::move(schema), std::move(array));
}

TEST_F(ArrowColumnTest, TwoWayConversion)
{
  cudf::test::fixed_width_column_wrapper<int32_t> int_col{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  auto col                           = cudf::column(int_col);
  auto arrow_column_from_cudf_column = cudf::arrow_column(std::move(col));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col, *arrow_column_from_cudf_column.view());

  auto [arrow_schema_from_cudf_column, arrow_array_from_arrow_column] =
    export_to_arrow(arrow_column_from_cudf_column);
  arrow_column_from_cudf_column.to_arrow_schema(arrow_schema_from_cudf_column.get());
  arrow_column_from_cudf_column.to_arrow(arrow_array_from_arrow_column.get(), ARROW_DEVICE_CUDA);

  auto arrow_column_from_arrow_array =
    cudf::arrow_column(arrow_schema_from_cudf_column.get(), arrow_array_from_arrow_column.get());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col, *arrow_column_from_arrow_array.view());
}

TEST_F(ArrowColumnTest, LifetimeManagement)
{
  cudf::test::fixed_width_column_wrapper<int32_t> int_col{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  auto col                           = std::make_unique<cudf::column>(int_col);
  auto arrow_column_from_cudf_column = std::make_unique<cudf::arrow_column>(std::move(*col));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col, *arrow_column_from_cudf_column->view());

  auto [schema1, array1] = export_to_arrow(*arrow_column_from_cudf_column);
  auto [schema2, array2] = export_to_arrow(*arrow_column_from_cudf_column);

  // Delete the original owner of the data, then reimport and ensure that we
  // are still referencing the same valid original data.
  arrow_column_from_cudf_column.reset();
  auto col1 = std::make_unique<cudf::arrow_column>(schema1.get(), array1.get());
  auto col2 = std::make_unique<cudf::arrow_column>(schema2.get(), array2.get());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col, *col1->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*col1->view(), *col2->view());
}

struct ArrowTableTest : public cudf::test::BaseFixture {};

TEST_F(ArrowTableTest, TwoWayConversion)
{
  cudf::test::fixed_width_column_wrapper<int32_t> int_col{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  cudf::test::fixed_width_column_wrapper<float> float_col{
    {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.}};
  auto original_view = cudf::table_view{{int_col, float_col}};
  cudf::table table{cudf::table_view{{int_col, float_col}}};
  auto arrow_table_from_cudf_table = cudf::arrow_table(std::move(table));

  CUDF_TEST_EXPECT_TABLES_EQUAL(original_view, *arrow_table_from_cudf_table.view());

  auto [arrow_schema_from_cudf_table, arrow_array_from_arrow_table] =
    export_to_arrow(arrow_table_from_cudf_table);
  arrow_table_from_cudf_table.to_arrow_schema(arrow_schema_from_cudf_table.get());
  arrow_table_from_cudf_table.to_arrow(arrow_array_from_arrow_table.get(), ARROW_DEVICE_CUDA);

  // auto arrow_table_from_arrow_array =
  //     cudf::arrow_table(arrow_schema_from_cudf_table.get(),
  //             arrow_array_from_arrow_table.get());
  // CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col, *arrow_table_from_arrow_array.view());
}
