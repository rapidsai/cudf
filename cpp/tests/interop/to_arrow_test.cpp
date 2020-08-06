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

std::pair<std::unique_ptr<cudf::table>, std::shared_ptr<arrow::Table>> get_tables(
  cudf::size_type length)
{
  std::vector<int64_t> int64_data(length);
  std::vector<bool> bool_data(length);
  std::vector<std::string> string_data(length);
  std::vector<uint8_t> validity(length);
  std::vector<bool> bool_validity(length);

  std::vector<std::unique_ptr<cudf::column>> columns;

  std::transform(int64_data.cbegin(), int64_data.cend(), int64_data.begin(), [](auto val) {
    return rand() % 500000;
  });
  std::transform(bool_data.cbegin(), bool_data.cend(), bool_data.begin(), [](auto val) {
    return rand() % 7 != 0 ? true : false;
  });
  std::transform(string_data.cbegin(), string_data.cend(), string_data.begin(), [](auto val) {
    return rand() % 7 != 0 ? "CUDF" : "Rocks";
  });
  std::transform(validity.cbegin(), validity.cend(), validity.begin(), [](auto val) {
    return rand() % 7 != 0 ? 1 : 0;
  });
  std::transform(bool_validity.cbegin(), bool_validity.cend(), bool_validity.begin(), [](auto val) {
    return rand() % 7 != 0 ? true : false;
  });

  columns.emplace_back(cudf::test::fixed_width_column_wrapper<int64_t>(
                         int64_data.begin(), int64_data.end(), validity.begin())
                         .release());
  columns.emplace_back(
    cudf::test::strings_column_wrapper(string_data.begin(), string_data.end(), validity.begin())
      .release());
  auto col4 = cudf::test::fixed_width_column_wrapper<int64_t>(
    int64_data.begin(), int64_data.end(), validity.begin());
  auto dict_col = cudf::dictionary::encode(col4);
  columns.emplace_back(std::move(cudf::dictionary::encode(col4)));
  columns.emplace_back(cudf::test::fixed_width_column_wrapper<bool>(
                         bool_data.begin(), bool_data.end(), bool_validity.begin())
                         .release());

  auto int64array   = get_arrow_array<int64_t>(int64_data, validity);
  auto string_array = get_arrow_array<cudf::string_view>(string_data, validity);
  cudf::dictionary_column_view view(dict_col->view());
  auto keys       = cudf::test::to_host<int64_t>(view.keys()).first;
  auto indices    = cudf::test::to_host<int32_t>(view.indices()).first;
  auto dict_array = get_arrow_dict_array(std::vector<int64_t>(keys.begin(), keys.end()),
                                         std::vector<int32_t>(indices.begin(), indices.end()),
                                         validity);
  auto boolarray  = get_arrow_array<bool>(bool_data, bool_validity);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
    arrow::field("a", int64array->type()),
    arrow::field("b", string_array->type()),
    arrow::field("c", dict_array->type()),
    arrow::field("d", boolarray->type())};

  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  return std::make_pair(
    std::make_unique<cudf::table>(std::move(columns)),
    arrow::Table::Make(schema, {int64array, string_array, dict_array, boolarray}));
}

struct ToArrowTest : public cudf::test::BaseFixture {
};

TEST_F(ToArrowTest, EmptyTable)
{
  auto tables = get_tables(0);

  auto cudf_table_view      = tables.first->view();
  auto expected_arrow_table = tables.second;

  auto got_arrow_table = cudf::to_arrow(cudf_table_view, {"a", "b", "c", "d"});

  ASSERT_EQ(expected_arrow_table->Equals(*got_arrow_table, true), true);
}

TEST_F(ToArrowTest, DateTimeTable)
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

TEST_F(ToArrowTest, NestedList)
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

struct ToArrowTestSlice
  : public ToArrowTest,
    public ::testing::WithParamInterface<std::tuple<cudf::size_type, cudf::size_type>> {
};

TEST_P(ToArrowTestSlice, SliceTest)
{
  auto tables          = get_tables(10000);
  auto cudf_table_view = tables.first->view();
  auto arrow_table     = tables.second;
  auto start           = std::get<0>(GetParam());
  auto end             = std::get<1>(GetParam());

  auto sliced_cudf_table    = cudf::slice(cudf_table_view, {start, end})[0];
  auto expected_arrow_table = arrow_table->Slice(start, end - start);
  auto got_arrow_table      = cudf::to_arrow(sliced_cudf_table, {"a", "b", "c", "d"});

  ASSERT_EQ(expected_arrow_table->Equals(*got_arrow_table, true), true);
}

INSTANTIATE_TEST_CASE_P(ToArrowTest,
                        ToArrowTestSlice,
                        ::testing::Values(std::make_tuple(0, 10000),
                                          std::make_tuple(100, 3000),
                                          std::make_tuple(0, 0),
                                          std::make_tuple(0, 3000),
                                          std::make_tuple(10000, 10000)));

CUDF_TEST_PROGRAM_MAIN()
