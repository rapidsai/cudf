/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <tests/interop/arrow_utils.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

std::unique_ptr<cudf::table> get_cudf_table()
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.emplace_back(
    cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 5, 2, 7}, {1, 0, 1, 1, 1}).release());
  columns.emplace_back(cudf::test::fixed_width_column_wrapper<int64_t>({1, 2, 3, 4, 5}).release());
  columns.emplace_back(
    cudf::test::strings_column_wrapper({"fff", "aaa", "", "fff", "ccc"}, {1, 1, 1, 0, 1})
      .release());
  auto col4 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 5, 2, 7}, {1, 0, 1, 1, 1});
  columns.emplace_back(std::move(cudf::dictionary::encode(col4)));
  columns.emplace_back(
    cudf::test::fixed_width_column_wrapper<bool>({true, false, true, false, true}, {1, 0, 1, 1, 0})
      .release());
  // columns.emplace_back(cudf::test::lists_column_wrapper<int>({{1, 2}, {3, 4}, {}, {6}, {7, 8,
  // 9}}).release());
  return std::make_unique<cudf::table>(std::move(columns));
}

struct FromArrowTest : public cudf::test::BaseFixture {
};

TEST_F(FromArrowTest, EmptyTable)
{
  auto tables = get_tables(0);

  auto expected_cudf_table = tables.first->view();
  auto arrow_table         = tables.second;

  auto got_cudf_table = cudf::from_arrow(*arrow_table);

  cudf::test::expect_tables_equal(expected_cudf_table, got_cudf_table->view());
}

TEST_F(FromArrowTest, DateTimeTable)
{
  auto data = {1, 2, 3, 4, 5, 6};

  auto col = cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms>(data);

  cudf::table_view expected_table_view({col});

  std::shared_ptr<arrow::Array> arr;
  arrow::TimestampBuilder timestamp_builder(timestamp(arrow::TimeUnit::type::MILLI),
                                            arrow::default_memory_pool());
  timestamp_builder.AppendValues(std::vector<int64_t>{1, 2, 3, 4, 5, 6});
  CUDF_EXPECTS(timestamp_builder.Finish(&arr).ok(), "Failed to build array");

  std::vector<std::shared_ptr<arrow::Field>> schema_vector({arrow::field("a", arr->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto arrow_table = arrow::Table::Make(schema, {arr});

  auto got_cudf_table = cudf::from_arrow(*arrow_table);

  cudf::test::expect_tables_equal(expected_table_view, got_cudf_table->view());
}

TEST_F(FromArrowTest, NestedList)
{
  auto valids =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return i == 2 ? false : true; });
  auto col = cudf::test::lists_column_wrapper<int64_t>({{{1, 2}, {3, 4}, {5}}, {{6}, {7, 8, 9}}});
  cudf::table_view expected_table_view({col});

  auto list_arr = get_arrow_list_array<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 2, 4, 5, 6, 9});
  std::vector<int32_t> offset{0, 3, 5};
  auto nested_list_arr = std::make_shared<arrow::ListArray>(
    arrow::list(list(arrow::int64())), offset.size() - 1, arrow::Buffer::Wrap(offset), list_arr);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector(
    {arrow::field("a", nested_list_arr->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto arrow_table = arrow::Table::Make(schema, {nested_list_arr});

  auto got_cudf_table = cudf::from_arrow(*arrow_table);
  cudf::test::expect_tables_equal(expected_table_view, got_cudf_table->view());
}

TEST_F(FromArrowTest, DictionaryIndicesType)
{
  auto array1 =
    get_arrow_dict_array<int64_t, int8_t>({1, 2, 5, 7}, {0, 1, 2, 1, 3}, {1, 0, 1, 1, 1});
  auto array2 =
    get_arrow_dict_array<int64_t, int16_t>({1, 2, 5, 7}, {0, 1, 2, 1, 3}, {1, 0, 1, 1, 1});
  auto array3 =
    get_arrow_dict_array<int64_t, int64_t>({1, 2, 5, 7}, {0, 1, 2, 1, 3}, {1, 0, 1, 1, 1});

  std::vector<std::shared_ptr<arrow::Field>> schema_vector({arrow::field("a", array1->type()),
                                                            arrow::field("b", array2->type()),
                                                            arrow::field("c", array3->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto arrow_table = arrow::Table::Make(schema, {array1, array2, array3});

  std::vector<std::unique_ptr<cudf::column>> columns;
  auto col = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 5, 2, 7}, {1, 0, 1, 1, 1});
  columns.emplace_back(std::move(cudf::dictionary::encode(col)));
  columns.emplace_back(std::move(cudf::dictionary::encode(col)));
  columns.emplace_back(std::move(cudf::dictionary::encode(col)));

  cudf::table expected_table(std::move(columns));

  auto got_cudf_table = cudf::from_arrow(*arrow_table);

  cudf::test::expect_tables_equal(expected_table.view(), got_cudf_table->view());
}

TEST_F(FromArrowTest, ChunkedArray)
{
  auto int64array     = get_arrow_array<int64_t>({1, 2, 3, 4, 5});
  auto int32array_1   = get_arrow_array<int32_t>({1, 2}, {1, 0});
  auto int32array_2   = get_arrow_array<int32_t>({5, 2, 7}, {1, 1, 1});
  auto string_array_1 = get_arrow_array<cudf::string_view>({
    "fff",
    "aaa",
    "",
  });
  auto string_array_2 = get_arrow_array<cudf::string_view>(
    {
      "fff",
      "ccc",
    },
    {0, 1});
  auto dict_array1 = get_arrow_dict_array({1, 2, 5, 7}, {0, 1, 2}, {1, 0, 1});
  auto dict_array2 = get_arrow_dict_array({1, 2, 5, 7}, {1, 3});

  auto int64_chunked_array = std::make_shared<arrow::ChunkedArray>(int64array);
  auto int32_chunked_array = std::make_shared<arrow::ChunkedArray>(
    std::vector<std::shared_ptr<arrow::Array>>{int32array_1, int32array_2});
  auto string_chunked_array = std::make_shared<arrow::ChunkedArray>(
    std::vector<std::shared_ptr<arrow::Array>>{string_array_1, string_array_2});
  auto dict_chunked_array = std::make_shared<arrow::ChunkedArray>(
    std::vector<std::shared_ptr<arrow::Array>>{dict_array1, dict_array2});
  auto boolean_array = get_arrow_array<bool>({true, false, true, false, true}, {1, 0, 1, 1, 0});
  auto boolean_chunked_array = std::make_shared<arrow::ChunkedArray>(boolean_array);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector(
    {arrow::field("a", int32_chunked_array->type()),
     arrow::field("b", int64array->type()),
     arrow::field("c", string_array_1->type()),
     arrow::field("d", dict_chunked_array->type()),
     arrow::field("e", boolean_chunked_array->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto arrow_table = arrow::Table::Make(schema,
                                        {int32_chunked_array,
                                         int64_chunked_array,
                                         string_chunked_array,
                                         dict_chunked_array,
                                         boolean_chunked_array});

  auto expected_cudf_table = get_cudf_table();

  auto got_cudf_table = cudf::from_arrow(*arrow_table);

  cudf::test::expect_tables_equal(expected_cudf_table->view(), got_cudf_table->view());
}

struct FromArrowTestSlice
  : public FromArrowTest,
    public ::testing::WithParamInterface<std::tuple<cudf::size_type, cudf::size_type>> {
};

TEST_P(FromArrowTestSlice, SliceTest)
{
  auto tables          = get_tables(10000);
  auto cudf_table_view = tables.first->view();
  auto arrow_table     = tables.second;
  auto start           = std::get<0>(GetParam());
  auto end             = std::get<1>(GetParam());

  auto sliced_cudf_table = cudf::slice(cudf_table_view, {start, end})[0];
  cudf::table expected_cudf_table{sliced_cudf_table};
  auto sliced_arrow_table = arrow_table->Slice(start, end - start);
  auto got_cudf_table     = cudf::from_arrow(*sliced_arrow_table);

  cudf::test::expect_tables_equal(expected_cudf_table.view(), got_cudf_table->view());
}

INSTANTIATE_TEST_CASE_P(FromArrowTest,
                        FromArrowTestSlice,
                        ::testing::Values(std::make_tuple(0, 10000),
                                          std::make_tuple(2912, 2915),
                                          std::make_tuple(100, 3000),
                                          std::make_tuple(0, 0),
                                          std::make_tuple(0, 3000),
                                          std::make_tuple(10000, 10000)));
