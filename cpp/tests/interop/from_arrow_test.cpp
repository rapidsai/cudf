/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <tests/interop/arrow_utils.hpp>

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

template <typename T>
struct FromArrowTestDurationsTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(FromArrowTestDurationsTest, cudf::test::DurationTypes);

TEST_F(FromArrowTest, EmptyTable)
{
  auto tables = get_tables(0);

  auto expected_cudf_table = tables.first->view();
  auto arrow_table         = tables.second;

  auto got_cudf_table = cudf::from_arrow(*arrow_table);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_cudf_table, got_cudf_table->view());
}

TEST_F(FromArrowTest, DateTimeTable)
{
  auto data = {1, 2, 3, 4, 5, 6};

  auto col =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(data);

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

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, got_cudf_table->view());
}

TYPED_TEST(FromArrowTestDurationsTest, DurationTable)
{
  using T = TypeParam;

  auto data = {T{1}, T{2}, T{3}, T{4}, T{5}, T{6}};
  auto col  = cudf::test::fixed_width_column_wrapper<T>(data);

  std::shared_ptr<arrow::Array> arr;
  cudf::table_view expected_table_view({col});
  arrow::TimeUnit::type arrow_unit;

  switch (cudf::type_to_id<TypeParam>()) {
    case cudf::type_id::DURATION_SECONDS: arrow_unit = arrow::TimeUnit::type::SECOND; break;
    case cudf::type_id::DURATION_MILLISECONDS: arrow_unit = arrow::TimeUnit::type::MILLI; break;
    case cudf::type_id::DURATION_MICROSECONDS: arrow_unit = arrow::TimeUnit::type::MICRO; break;
    case cudf::type_id::DURATION_NANOSECONDS: arrow_unit = arrow::TimeUnit::type::NANO; break;
    case cudf::type_id::DURATION_DAYS: return;
    default: CUDF_FAIL("Unsupported duration unit in arrow");
  }
  arrow::DurationBuilder duration_builder(duration(arrow_unit), arrow::default_memory_pool());
  duration_builder.AppendValues(std::vector<int64_t>{1, 2, 3, 4, 5, 6});
  CUDF_EXPECTS(duration_builder.Finish(&arr).ok(), "Failed to build array");

  std::vector<std::shared_ptr<arrow::Field>> schema_vector({arrow::field("a", arr->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto arrow_table = arrow::Table::Make(schema, {arr});

  auto got_cudf_table = cudf::from_arrow(*arrow_table);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, got_cudf_table->view());
}

TEST_F(FromArrowTest, NestedList)
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 3 != 0; });
  auto col = cudf::test::lists_column_wrapper<int64_t>(
    {{{{{1, 2}, valids}, {{3, 4}, valids}, {5}}, {{6}, {{7, 8, 9}, valids}}}, valids});
  cudf::table_view expected_table_view({col});

  auto list_arr = get_arrow_list_array<int64_t>({6, 7, 8, 9}, {0, 1, 4}, {1, 0, 1, 1});
  std::vector<int32_t> offset{0, 0, 2};
  auto mask_buffer     = arrow::internal::BytesToBits({0, 1}).ValueOrDie();
  auto nested_list_arr = std::make_shared<arrow::ListArray>(arrow::list(list(arrow::int64())),
                                                            offset.size() - 1,
                                                            arrow::Buffer::Wrap(offset),
                                                            list_arr,
                                                            mask_buffer);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector(
    {arrow::field("a", nested_list_arr->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto arrow_table = arrow::Table::Make(schema, {nested_list_arr});

  auto got_cudf_table = cudf::from_arrow(*arrow_table);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, got_cudf_table->view());
}

TEST_F(FromArrowTest, StructColumn)
{
  using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;

  // Create cudf table
  auto nested_type_field_names =
    std::vector<std::vector<std::string>>{{"string", "integral", "bool", "nested_list", "struct"}};
  auto str_col =
    cudf::test::strings_column_wrapper{
      "Samuel Vimes", "Carrot Ironfoundersson", "Angua von Uberwald"}
      .release();
  auto str_col2 =
    cudf::test::strings_column_wrapper{{"CUDF", "ROCKS", "EVERYWHERE"}, {0, 1, 0}}.release();
  int num_rows{str_col->size()};
  auto int_col = cudf::test::fixed_width_column_wrapper<int32_t, int32_t>{{48, 27, 25}}.release();
  auto int_col2 =
    cudf::test::fixed_width_column_wrapper<int32_t, int32_t>{{12, 24, 47}, {1, 0, 1}}.release();
  auto bool_col = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}}.release();
  auto list_col =
    cudf::test::lists_column_wrapper<int64_t>({{{1, 2}, {3, 4}, {5}}, {{{6}}}, {{7}, {8, 9}}})
      .release();
  vector_of_columns cols2;
  cols2.push_back(std::move(str_col2));
  cols2.push_back(std::move(int_col2));
  auto mask =
    cudf::bools_to_mask(cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}});
  auto sub_struct_col = cudf::make_structs_column(
    num_rows, std::move(cols2), cudf::UNKNOWN_NULL_COUNT, std::move(*(mask.first)));
  vector_of_columns cols;
  cols.push_back(std::move(str_col));
  cols.push_back(std::move(int_col));
  cols.push_back(std::move(bool_col));
  cols.push_back(std::move(list_col));
  cols.push_back(std::move(sub_struct_col));

  auto struct_col = cudf::make_structs_column(num_rows, std::move(cols), 0, {});
  cudf::table_view expected_cudf_table({struct_col->view()});

  // Create Arrow table
  std::vector<std::string> str{"Samuel Vimes", "Carrot Ironfoundersson", "Angua von Uberwald"};
  std::vector<std::string> str2{"CUDF", "ROCKS", "EVERYWHERE"};
  auto str_array  = get_arrow_array<cudf::string_view>(str);
  auto int_array  = get_arrow_array<int32_t>({48, 27, 25});
  auto str2_array = get_arrow_array<cudf::string_view>(str2, {0, 1, 0});
  auto int2_array = get_arrow_array<int32_t>({12, 24, 47}, {1, 0, 1});
  auto bool_array = get_arrow_array<bool>({true, true, false});
  auto list_arr = get_arrow_list_array<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 2, 4, 5, 6, 7, 9});
  std::vector<int32_t> offset{0, 3, 4, 6};
  auto nested_list_arr = std::make_shared<arrow::ListArray>(
    arrow::list(list(arrow::int64())), offset.size() - 1, arrow::Buffer::Wrap(offset), list_arr);

  std::vector<std::shared_ptr<arrow::Array>> child_arrays2({str2_array, int2_array});
  auto fields2 = std::vector<std::shared_ptr<arrow::Field>>{
    std::make_shared<arrow::Field>("string2", str2_array->type(), str2_array->null_count() > 0),
    std::make_shared<arrow::Field>("integral2", int2_array->type(), int2_array->null_count() > 0)};
  std::shared_ptr<arrow::Buffer> mask_buffer = arrow::internal::BytesToBits({1, 1, 0}).ValueOrDie();
  auto dtype2                                = std::make_shared<arrow::StructType>(fields2);
  auto struct_array2                         = std::make_shared<arrow::StructArray>(
    dtype2, static_cast<int64_t>(expected_cudf_table.num_rows()), child_arrays2, mask_buffer);

  std::vector<std::shared_ptr<arrow::Array>> child_arrays(
    {str_array, int_array, bool_array, nested_list_arr, struct_array2});
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::transform(child_arrays.cbegin(),
                 child_arrays.cend(),
                 nested_type_field_names[0].cbegin(),
                 std::back_inserter(fields),
                 [](auto const array, auto const name) {
                   return std::make_shared<arrow::Field>(
                     name, array->type(), array->null_count() > 0);
                 });
  auto dtype = std::make_shared<arrow::StructType>(fields);

  auto struct_array = std::make_shared<arrow::StructArray>(
    dtype, static_cast<int64_t>(expected_cudf_table.num_rows()), child_arrays);
  std::vector<std::shared_ptr<arrow::Field>> schema_vector(
    {arrow::field("a", struct_array->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  auto input  = arrow::Table::Make(schema, {struct_array});

  auto got_cudf_table = cudf::from_arrow(*input);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_cudf_table, got_cudf_table->view());
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

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table.view(), got_cudf_table->view());
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

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_cudf_table->view(), got_cudf_table->view());
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

  // This has been added to take-care of empty string column issue with no children
  if (got_cudf_table->num_rows() == 0 and expected_cudf_table.num_rows() == 0) {
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_cudf_table.view(), got_cudf_table->view());
  } else {
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_cudf_table.view(), got_cudf_table->view());
  }
}

INSTANTIATE_TEST_CASE_P(FromArrowTest,
                        FromArrowTestSlice,
                        ::testing::Values(std::make_tuple(0, 10000),
                                          std::make_tuple(2912, 2915),
                                          std::make_tuple(100, 3000),
                                          std::make_tuple(0, 0),
                                          std::make_tuple(0, 3000),
                                          std::make_tuple(10000, 10000)));
