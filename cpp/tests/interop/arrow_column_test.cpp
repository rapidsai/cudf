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

#include "nanoarrow/common/inline_types.h"
#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow.hpp"
#include "nanoarrow/nanoarrow_device.h"
#include "tests/interop/nanoarrow_utils.hpp"

#include <tests/interop/arrow_utils.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/filling.hpp>
#include <cudf/interop.hpp>
#include <cudf/types.hpp>

// #include <cudf_test/column_utilities.hpp>
// #include <cudf_test/column_wrapper.hpp>
// #include <cudf_test/table_utilities.hpp>
// #include <cudf_test/type_lists.hpp>
//
// #include <cudf/column/column.hpp>
// #include <cudf/column/column_view.hpp>
// #include <cudf/copying.hpp>
// #include <cudf/detail/iterator.cuh>
// #include <cudf/dictionary/encode.hpp>
// #include <cudf/interop.hpp>
// #include <cudf/table/table.hpp>
// #include <cudf/table/table_view.hpp>
// #include <cudf/types.hpp>
//
// #include <thrust/iterator/counting_iterator.h>
//
// #include <arrow/c/bridge.h>
//
// std::unique_ptr<cudf::table> get_cudf_table()
//{
//   std::vector<std::unique_ptr<cudf::column>> columns;
//   columns.emplace_back(cudf::test::fixed_width_column_wrapper<int32_t>(
//                          {1, 2, 5, 2, 7}, {true, false, true, true, true})
//                          .release());
//   columns.emplace_back(cudf::test::fixed_width_column_wrapper<int64_t>({1, 2, 3, 4,
//   5}).release()); columns.emplace_back(cudf::test::strings_column_wrapper({"fff", "aaa", "",
//   "fff", "ccc"},
//                                                           {true, true, true, false, true})
//                          .release());
//   auto col4 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 5, 2, 7},
//                                                               {true, false, true, true, true});
//   columns.emplace_back(cudf::dictionary::encode(col4));
//   columns.emplace_back(cudf::test::fixed_width_column_wrapper<bool>(
//                          {true, false, true, false, true}, {true, false, true, true, false})
//                          .release());
//   columns.emplace_back(cudf::test::strings_column_wrapper(
//                          {
//                            "",
//                            "abc",
//                            "def",
//                            "1",
//                            "2",
//                          },
//                          {0, 1, 1, 1, 1})
//                          .release());
//   // columns.emplace_back(cudf::test::lists_column_wrapper<int>({{1, 2}, {3, 4}, {}, {6}, {7, 8,
//   // 9}}).release());
//   return std::make_unique<cudf::table>(std::move(columns));
// }
//
// std::shared_ptr<arrow::LargeStringArray> get_arrow_large_string_array(
//   std::vector<std::string> const& data, std::vector<uint8_t> const& mask = {})
//{
//   std::shared_ptr<arrow::LargeStringArray> large_string_array;
//   arrow::LargeStringBuilder large_string_builder;
//
//   CUDF_EXPECTS(large_string_builder.AppendValues(data, mask.data()).ok(),
//                "Failed to append values to string builder");
//   CUDF_EXPECTS(large_string_builder.Finish(&large_string_array).ok(),
//                "Failed to create arrow string array");
//
//   return large_string_array;
// }
//
// struct FromArrowTest : public cudf::test::BaseFixture {};
//
// std::optional<std::unique_ptr<cudf::table>> export_table(std::shared_ptr<arrow::Table>
// arrow_table)
//{
//   ArrowSchema schema;
//   if (!arrow::ExportSchema(*arrow_table->schema(), &schema).ok()) { return std::nullopt; }
//   auto batch = arrow_table->CombineChunksToBatch().ValueOrDie();
//   ArrowArray arr;
//   if (!arrow::ExportRecordBatch(*batch, &arr).ok()) { return std::nullopt; }
//   auto ret = cudf::from_arrow(&schema, &arr);
//   arr.release(&arr);
//   schema.release(&schema);
//   return {std::move(ret)};
// }
//
// TEST_F(FromArrowTest, ChunkedArray)
//{
//   auto int64array     = get_arrow_array<int64_t>({1, 2, 3, 4, 5});
//   auto int32array_1   = get_arrow_array<int32_t>({1, 2}, {1, 0});
//   auto int32array_2   = get_arrow_array<int32_t>({5, 2, 7}, {1, 1, 1});
//   auto string_array_1 = get_arrow_array<cudf::string_view>({
//     "fff",
//     "aaa",
//     "",
//   });
//   auto string_array_2 = get_arrow_array<cudf::string_view>(
//     {
//       "fff",
//       "ccc",
//     },
//     {0, 1});
//   auto large_string_array_1 = get_arrow_large_string_array(
//     {
//       "",
//       "abc",
//       "def",
//       "1",
//       "2",
//     },
//     {0, 1, 1, 1, 1});
//   auto dict_array1 = get_arrow_dict_array({1, 2, 5, 7}, {0, 1, 2}, {1, 0, 1});
//   auto dict_array2 = get_arrow_dict_array({1, 2, 5, 7}, {1, 3});
//
//   auto int64_chunked_array = std::make_shared<arrow::ChunkedArray>(int64array);
//   auto int32_chunked_array = std::make_shared<arrow::ChunkedArray>(
//     std::vector<std::shared_ptr<arrow::Array>>{int32array_1, int32array_2});
//   auto string_chunked_array = std::make_shared<arrow::ChunkedArray>(
//     std::vector<std::shared_ptr<arrow::Array>>{string_array_1, string_array_2});
//   auto dict_chunked_array = std::make_shared<arrow::ChunkedArray>(
//     std::vector<std::shared_ptr<arrow::Array>>{dict_array1, dict_array2});
//   auto boolean_array =
//     get_arrow_array<bool>({true, false, true, false, true}, {true, false, true, true, false});
//   auto boolean_chunked_array      = std::make_shared<arrow::ChunkedArray>(boolean_array);
//   auto large_string_chunked_array = std::make_shared<arrow::ChunkedArray>(
//     std::vector<std::shared_ptr<arrow::Array>>{large_string_array_1});
//
//   std::vector<std::shared_ptr<arrow::Field>> schema_vector(
//     {arrow::field("a", int32_chunked_array->type()),
//      arrow::field("b", int64array->type()),
//      arrow::field("c", string_array_1->type()),
//      arrow::field("d", dict_chunked_array->type()),
//      arrow::field("e", boolean_chunked_array->type()),
//      arrow::field("f", large_string_array_1->type())});
//   auto schema = std::make_shared<arrow::Schema>(schema_vector);
//
//   auto arrow_table = arrow::Table::Make(schema,
//                                         {int32_chunked_array,
//                                          int64_chunked_array,
//                                          string_chunked_array,
//                                          dict_chunked_array,
//                                          boolean_chunked_array,
//                                          large_string_chunked_array});
//
//   auto expected_cudf_table = get_cudf_table();
//
//   auto got_cudf_table = export_table(arrow_table);
//   ASSERT_TRUE(got_cudf_table.has_value());
//
//   CUDF_TEST_EXPECT_TABLES_EQUAL(expected_cudf_table->view(), got_cudf_table.value()->view());
// }

struct ArrowColumnTest : public cudf::test::BaseFixture {};

TEST_F(ArrowColumnTest, TwoWayConversion)
{
  cudf::test::fixed_width_column_wrapper<int32_t> int_col{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  auto col                           = cudf::column(int_col);
  auto arrow_column_from_cudf_column = cudf::arrow_column(std::move(col));

  // Now we can extract an ArrowDeviceArray from the arrow_column
  ArrowSchema arrow_schema_from_cudf_column;
  arrow_column_from_cudf_column.to_arrow_schema(&arrow_schema_from_cudf_column);
  ArrowDeviceArray arrow_array_from_arrow_column;
  arrow_column_from_cudf_column.to_arrow(&arrow_array_from_arrow_column, ARROW_DEVICE_CUDA);

  // Now let's convert it back to an arrow_column
  auto arrow_column_from_arrow_array =
    cudf::arrow_column(&arrow_schema_from_cudf_column, &arrow_array_from_arrow_column);

  // Now do some assertions
  auto view = arrow_column_from_arrow_array.view();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col.view(), *arrow_column_from_cudf_column.view());

  //// Should be able to create an arrow_column from an ArrowDeviceArray.
  // auto tmp1 = cudf::arrow_column(&arr);
  //
  //// Should be able to create an arrow_column from cudf::column. It always takes ownership.
  // cudf::column col;

  //// Should be able to create an arrow_table from an ArrowDeviceArray.
  // ArrowDeviceArray arr2;
  // cudf::arrow_table(&arr2);
  //
  //// Should be able to create an arrow_table from cudf::table. It always takes ownership.
  // cudf::table tbl;
  // cudf::arrow_table(std::move(tbl));
}
