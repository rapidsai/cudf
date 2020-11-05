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
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>
#include <tests/interop/arrow_utils.hpp>

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;

std::pair<std::unique_ptr<cudf::table>, std::shared_ptr<arrow::Table>> get_tables(
  cudf::size_type length)
{
  std::vector<int64_t> int64_data(length);
  std::vector<bool> bool_data(length);
  std::vector<std::string> string_data(length);
  std::vector<uint8_t> validity(length);
  std::vector<bool> bool_validity(length);
  std::vector<uint8_t> bool_data_validity;
  cudf::size_type length_of_individual_list = 3;
  cudf::size_type length_of_list            = length_of_individual_list * length;
  std::vector<int64_t> list_int64_data(length_of_list);
  std::vector<uint8_t> list_int64_data_validity(length_of_list);
  std::vector<int32_t> list_offsets(length + 1);

  std::vector<std::unique_ptr<cudf::column>> columns;

  std::generate(int64_data.begin(), int64_data.end(), []() { return rand() % 500000; });
  std::generate(list_int64_data.begin(), list_int64_data.end(), []() { return rand() % 500000; });
  auto validity_generator = []() { return rand() % 7 != 0; };
  std::generate(
    list_int64_data_validity.begin(), list_int64_data_validity.end(), validity_generator);
  // cudf::size_type n = 0;
  std::generate(
    list_offsets.begin(), list_offsets.end(), [length_of_individual_list, n = 0]() mutable {
      return (n++) * length_of_individual_list;
    });
  std::generate(bool_data.begin(), bool_data.end(), validity_generator);
  std::generate(
    string_data.begin(), string_data.end(), []() { return rand() % 7 != 0 ? "CUDF" : "Rocks"; });
  std::generate(validity.begin(), validity.end(), validity_generator);
  std::generate(bool_validity.begin(), bool_validity.end(), validity_generator);

  std::transform(bool_validity.cbegin(),
                 bool_validity.cend(),
                 std::back_inserter(bool_data_validity),
                 [](auto val) { return static_cast<uint8_t>(val); });

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
  auto list_child_column = cudf::test::fixed_width_column_wrapper<int64_t>(
    list_int64_data.begin(), list_int64_data.end(), list_int64_data_validity.begin());
  auto list_offsets_column =
    cudf::test::fixed_width_column_wrapper<int32_t>(list_offsets.begin(), list_offsets.end());
  auto list_mask = cudf::bools_to_mask(cudf::test::fixed_width_column_wrapper<bool>(
    bool_data_validity.begin(), bool_data_validity.end()));
  columns.emplace_back(cudf::make_lists_column(length,
                                               list_offsets_column.release(),
                                               list_child_column.release(),
                                               cudf::UNKNOWN_NULL_COUNT,
                                               std::move(*(list_mask.first))));
  auto int_column = cudf::test::fixed_width_column_wrapper<int64_t>(
                      int64_data.begin(), int64_data.end(), validity.begin())
                      .release();
  auto str_column =
    cudf::test::strings_column_wrapper(string_data.begin(), string_data.end(), validity.begin())
      .release();
  vector_of_columns cols;
  cols.push_back(move(int_column));
  cols.push_back(move(str_column));
  auto mask = cudf::bools_to_mask(cudf::test::fixed_width_column_wrapper<bool>(
    bool_data_validity.begin(), bool_data_validity.end()));
  columns.emplace_back(cudf::make_structs_column(
    length, std::move(cols), cudf::UNKNOWN_NULL_COUNT, std::move(*(mask.first))));

  auto int64array = get_arrow_array<int64_t>(int64_data, validity);

  auto string_array = get_arrow_array<cudf::string_view>(string_data, validity);
  cudf::dictionary_column_view view(dict_col->view());
  auto keys       = cudf::test::to_host<int64_t>(view.keys()).first;
  auto indices    = cudf::test::to_host<uint32_t>(view.indices()).first;
  auto dict_array = get_arrow_dict_array(std::vector<int64_t>(keys.begin(), keys.end()),
                                         std::vector<int32_t>(indices.begin(), indices.end()),
                                         validity);
  auto boolarray  = get_arrow_array<bool>(bool_data, bool_validity);
  auto list_array = get_arrow_list_array<int64_t>(
    list_int64_data, list_offsets, list_int64_data_validity, bool_data_validity);

  arrow::ArrayVector child_arrays({int64array, string_array});
  std::vector<std::shared_ptr<arrow::Field>> fields = {
    arrow::field("integral", int64array->type(), int64array->null_count() > 0),
    arrow::field("string", string_array->type(), string_array->null_count() > 0)};
  auto dtype = std::make_shared<arrow::StructType>(fields);
  std::shared_ptr<arrow::Buffer> mask_buffer =
    arrow::internal::BytesToBits(static_cast<std::vector<uint8_t>>(bool_data_validity))
      .ValueOrDie();
  auto struct_array =
    std::make_shared<arrow::StructArray>(dtype, length, child_arrays, mask_buffer);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
    arrow::field("a", int64array->type()),
    arrow::field("b", string_array->type()),
    arrow::field("c", dict_array->type()),
    arrow::field("d", boolarray->type()),
    arrow::field("e", list_array->type()),
    arrow::field("f", struct_array->type())};

  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  return std::make_pair(
    std::make_unique<cudf::table>(std::move(columns)),
    arrow::Table::Make(
      schema, {int64array, string_array, dict_array, boolarray, list_array, struct_array}));
}

struct ToArrowTest : public cudf::test::BaseFixture {
};

template <typename T>
struct ToArrowTestDurationsTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(ToArrowTestDurationsTest, cudf::test::DurationTypes);

TEST_F(ToArrowTest, EmptyTable)
{
  auto tables = get_tables(0);

  auto cudf_table_view      = tables.first->view();
  auto expected_arrow_table = tables.second;
  auto struct_meta          = cudf::column_metadata{"f"};
  struct_meta.children_meta = {{"integral"}, {"string"}};

  auto got_arrow_table =
    cudf::to_arrow(cudf_table_view, {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, struct_meta});

  ASSERT_EQ(expected_arrow_table->Equals(*got_arrow_table, true), true);
}

TEST_F(ToArrowTest, DateTimeTable)
{
  auto data = {1, 2, 3, 4, 5, 6};

  auto col =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(data);

  cudf::table_view input_view({col});

  std::shared_ptr<arrow::Array> arr;
  arrow::TimestampBuilder timestamp_builder(timestamp(arrow::TimeUnit::type::MILLI),
                                            arrow::default_memory_pool());
  CUDF_EXPECTS(timestamp_builder.AppendValues(std::vector<int64_t>{1, 2, 3, 4, 5, 6}).ok(),
               "Failed to append values");
  CUDF_EXPECTS(timestamp_builder.Finish(&arr).ok(), "Failed to build array");

  std::vector<std::shared_ptr<arrow::Field>> schema_vector({arrow::field("a", arr->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto expected_arrow_table = arrow::Table::Make(schema, {arr});

  auto got_arrow_table = cudf::to_arrow(input_view, {{"a"}});

  ASSERT_EQ(expected_arrow_table->Equals(*got_arrow_table, true), true);
}

TYPED_TEST(ToArrowTestDurationsTest, DurationTable)
{
  using T = TypeParam;

  auto data = {T{1}, T{2}, T{3}, T{4}, T{5}, T{6}};
  auto col  = cudf::test::fixed_width_column_wrapper<T>(data);

  cudf::table_view input_view({col});

  std::shared_ptr<arrow::Array> arr;
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
  CUDF_EXPECTS(duration_builder.AppendValues(std::vector<int64_t>{1, 2, 3, 4, 5, 6}).ok(),
               "Failed to append values");
  CUDF_EXPECTS(duration_builder.Finish(&arr).ok(), "Failed to build array");

  std::vector<std::shared_ptr<arrow::Field>> schema_vector({arrow::field("a", arr->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto expected_arrow_table = arrow::Table::Make(schema, {arr});

  auto got_arrow_table = cudf::to_arrow(input_view, {{"a"}});

  ASSERT_EQ(expected_arrow_table->Equals(*got_arrow_table, true), true);
}

TEST_F(ToArrowTest, NestedList)
{
  auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 3 != 0; });
  auto col    = cudf::test::lists_column_wrapper<int64_t>(
    {{{{{1, 2}, valids}, {{3, 4}, valids}, {5}}, {{6}, {{7, 8, 9}, valids}}}, valids});
  cudf::table_view input_view({col});

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

  auto expected_arrow_table = arrow::Table::Make(schema, {nested_list_arr});
  auto got_arrow_table      = cudf::to_arrow(input_view, {{"a"}});

  ASSERT_TRUE(expected_arrow_table->Equals(*got_arrow_table, true));
}

TEST_F(ToArrowTest, StructColumn)
{
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
  cudf::table_view input_view({struct_col->view()});

  // Create name metadata
  auto sub_metadata          = cudf::column_metadata{"struct"};
  sub_metadata.children_meta = {{"string2"}, {"integral2"}};
  auto metadata              = cudf::column_metadata{"a"};
  metadata.children_meta     = {{"string"}, {"integral"}, {"bool"}, {"nested_list"}, sub_metadata};

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
  auto dtype2                                = std::make_shared<arrow::StructType>(fields2);
  std::shared_ptr<arrow::Buffer> mask_buffer = arrow::internal::BytesToBits({1, 1, 0}).ValueOrDie();
  auto struct_array2                         = std::make_shared<arrow::StructArray>(
    dtype2, static_cast<int64_t>(input_view.num_rows()), child_arrays2, mask_buffer);

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
    dtype, static_cast<int64_t>(input_view.num_rows()), child_arrays);
  std::vector<std::shared_ptr<arrow::Field>> schema_vector(
    {arrow::field("a", struct_array->type())});
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto expected_arrow_table = arrow::Table::Make(schema, {struct_array});

  auto got_arrow_table = cudf::to_arrow(input_view, {metadata});

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
  auto struct_meta          = cudf::column_metadata{"f"};
  struct_meta.children_meta = {{"integral"}, {"string"}};
  auto got_arrow_table =
    cudf::to_arrow(sliced_cudf_table, {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, struct_meta});

  ASSERT_EQ(expected_arrow_table->Equals(*got_arrow_table, true), true);
}

INSTANTIATE_TEST_CASE_P(ToArrowTest,
                        ToArrowTestSlice,
                        ::testing::Values(std::make_tuple(0, 10000),
                                          std::make_tuple(100, 3000),
                                          std::make_tuple(0, 0),
                                          std::make_tuple(0, 3000)));

CUDF_TEST_PROGRAM_MAIN()
