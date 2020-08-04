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
#include <tests/utilities/type_lists.hpp>

struct CUDFToArrowTest : public cudf::test::BaseFixture {
};

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

std::shared_ptr<arrow::Table> get_arrow_table()
{
  auto int64array = get_arrow_array<int64_t>({1, 2, 3, 4, 5});
  auto int32array = get_arrow_array<int32_t>({1, 2, 5, 2, 7}, {1, 0, 1, 1, 1});
  auto string_array =
    get_arrow_array<cudf::string_view>({"fff", "aaa", "", "fff", "ccc"}, {1, 1, 1, 0, 1});
  auto dict_array = get_arrow_dict_array({1, 2, 5, 7}, {0, 1, 2, 1, 3}, {1, 0, 1, 1, 1});
  auto boolarray  = get_arrow_array<bool>({true, false, true, false, true}, {1, 0, 1, 1, 0});
  // auto list_array = get_arrow_list_array({1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 2, 4, 5, 6, 9}, {1, 1,
  // 0, 1, 1});

  // std::vector <std::shared_ptr<arrow::Field>> schema_vector = {arrow::field("a",
  // int32array->type()), arrow::field("b", int64array->type()), arrow::field("c",
  // string_array->type()), arrow::field("d", dict_array->type()), arrow::field("e",
  // list_array->type())};
  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
    arrow::field("a", int32array->type()),
    arrow::field("b", int64array->type()),
    arrow::field("c", string_array->type()),
    arrow::field("d", dict_array->type()),
    arrow::field("e", boolarray->type())};

  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  // return arrow::Table::Make(schema, {int32array, int64array, string_array, dict_array,
  // list_array});
  return arrow::Table::Make(schema, {int32array, int64array, string_array, dict_array, boolarray});
}

TEST_F(CUDFToArrowTest, DateTimeTable)
{
  auto data = {1, 2, 3, 4, 5, 6};

  auto col = cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms>(data);

  cudf::table_view input_view({col});

  std::shared_ptr<arrow::Array> arr;
  arrow::TimestampBuilder timestamp_builder(timestamp(arrow::TimeUnit::type::MILLI),
                                            arrow::default_memory_pool());
  timestamp_builder.AppendValues(std::vector<int64_t>{1, 2, 3, 4, 5, 6});
  CUDF_EXPECTS(timestamp_builder.Finish(&arr).ok(), "Failed to build array");

  std::vector<std::shared_ptr<arrow::Field>> schema_vector({arrow::field("a", arr->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto expected_arrow_table = arrow::Table::Make(schema, {arr});

  auto got_arrow_table = cudf::to_arrow(input_view, {"a"});

  ASSERT_EQ(expected_arrow_table->Equals(*got_arrow_table, true), true);
}

TEST_F(CUDFToArrowTest, NestedList)
{
  auto col = cudf::test::lists_column_wrapper<int64_t>({{{1, 2}, {3, 4}, {5}}, {{6}, {7, 8, 9}}});
  cudf::table_view input_view({col});

  auto list_arr = get_arrow_list_array<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 2, 4, 5, 6, 9});
  std::vector<int32_t> offset{0, 3, 5};
  auto nested_list_arr = std::make_shared<arrow::ListArray>(
    arrow::list(list(arrow::int64())), offset.size() - 1, arrow::Buffer::Wrap(offset), list_arr);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector(
    {arrow::field("a", nested_list_arr->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto expected_arrow_table = arrow::Table::Make(schema, {nested_list_arr});
  auto got_arrow_table      = cudf::to_arrow(input_view, {"a"});

  ASSERT_TRUE(expected_arrow_table->Equals(*got_arrow_table, true));
}

struct CUDFToArrowTestSlice
  : public CUDFToArrowTest,
    public ::testing::WithParamInterface<std::tuple<cudf::size_type, cudf::size_type>> {
};

TEST_P(CUDFToArrowTestSlice, SliceTest)
{
  auto cudf_table  = get_cudf_table();
  auto arrow_table = get_arrow_table();
  auto start       = std::get<0>(GetParam());
  auto end         = std::get<1>(GetParam());

  auto sliced_cudf_table    = cudf::slice(cudf_table->view(), {start, end})[0];
  auto expected_arrow_table = arrow_table->Slice(start, end - start);
  auto got_arrow_table      = cudf::to_arrow(sliced_cudf_table, {"a", "b", "c", "d", "e"});

  ASSERT_EQ(expected_arrow_table->Equals(*got_arrow_table, true), true);
}

INSTANTIATE_TEST_CASE_P(CUDFToArrowTest,
                        CUDFToArrowTestSlice,
                        ::testing::Values(std::make_tuple(0, 5),
                                          std::make_tuple(1, 3),
                                          std::make_tuple(0, 0),
                                          std::make_tuple(0, 2),
                                          std::make_tuple(4, 4)));

CUDF_TEST_PROGRAM_MAIN()
