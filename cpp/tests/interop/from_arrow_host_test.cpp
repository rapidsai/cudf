/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "nanoarrow_utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <thrust/iterator/counting_iterator.h>

// create a cudf::table and equivalent arrow table with host memory
std::tuple<std::unique_ptr<cudf::table>, nanoarrow::UniqueSchema, nanoarrow::UniqueArray>
get_nanoarrow_host_tables(cudf::size_type length)
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
  auto [list_mask, list_nulls] = cudf::bools_to_mask(cudf::test::fixed_width_column_wrapper<bool>(
    bool_data_validity.begin(), bool_data_validity.end()));
  columns.emplace_back(cudf::make_lists_column(length,
                                               list_offsets_column.release(),
                                               list_child_column.release(),
                                               list_nulls,
                                               std::move(*list_mask)));
  auto int_column = cudf::test::fixed_width_column_wrapper<int64_t>(
                      int64_data.begin(), int64_data.end(), validity.begin())
                      .release();
  auto str_column =
    cudf::test::strings_column_wrapper(string_data.begin(), string_data.end(), validity.begin())
      .release();
  vector_of_columns cols;
  cols.push_back(move(int_column));
  cols.push_back(move(str_column));
  auto [null_mask, null_count] = cudf::bools_to_mask(cudf::test::fixed_width_column_wrapper<bool>(
    bool_data_validity.begin(), bool_data_validity.end()));
  columns.emplace_back(
    cudf::make_structs_column(length, std::move(cols), null_count, std::move(*null_mask)));

  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(schema.get(), 6));

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema->children[0], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[0], "a"));
  if (columns[0]->null_count() > 0) {
    schema->children[0]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[0]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema->children[1], NANOARROW_TYPE_STRING));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[1], "b"));
  if (columns[1]->null_count() > 0) {
    schema->children[1]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[1]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema->children[2], NANOARROW_TYPE_UINT32));
  NANOARROW_THROW_NOT_OK(ArrowSchemaAllocateDictionary(schema->children[2]));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(schema->children[2]->dictionary, NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[2], "c"));
  if (columns[2]->null_count() > 0) {
    schema->children[2]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[2]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema->children[3], NANOARROW_TYPE_BOOL));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[3], "d"));
  if (columns[3]->null_count() > 0) {
    schema->children[3]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[3]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema->children[4], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(schema->children[4]->children[0], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[4]->children[0], "element"));
  if (columns[4]->child(1).null_count() > 0) {
    schema->children[4]->children[0]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[4]->children[0]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[4], "e"));
  if (columns[4]->has_nulls()) {
    schema->children[4]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[4]->flags = 0;
  }

  ArrowSchemaInit(schema->children[5]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(schema->children[5], 2));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(schema->children[5]->children[0], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[5]->children[0], "integral"));
  if (columns[5]->child(0).has_nulls()) {
    schema->children[5]->children[0]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[5]->children[0]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(schema->children[5]->children[1], NANOARROW_TYPE_STRING));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[5]->children[1], "string"));
  if (columns[5]->child(1).has_nulls()) {
    schema->children[5]->children[1]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[5]->children[1]->flags = 0;
  }

  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[5], "f"));
  if (columns[5]->has_nulls()) {
    schema->children[5]->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->children[5]->flags = 0;
  }

  auto int64_array  = get_nanoarrow_array<int64_t>(int64_data, validity);
  auto string_array = get_nanoarrow_array<cudf::string_view>(string_data, validity);
  cudf::dictionary_column_view view(dict_col->view());
  auto keys       = cudf::test::to_host<int64_t>(view.keys()).first;
  auto indices    = cudf::test::to_host<uint32_t>(view.indices()).first;
  auto dict_array = get_nanoarrow_dict_array(std::vector<int64_t>(keys.begin(), keys.end()),
                                             std::vector<int32_t>(indices.begin(), indices.end()),
                                             validity);
  auto boolarray  = get_nanoarrow_array<bool>(bool_data, bool_validity);
  auto list_array = get_nanoarrow_list_array<int64_t>(
    list_int64_data, list_offsets, list_int64_data_validity, bool_data_validity);

  nanoarrow::UniqueArray arrow;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(arrow.get(), schema.get(), nullptr));
  arrow->length = length;

  int64_array.move(arrow->children[0]);
  string_array.move(arrow->children[1]);
  dict_array.move(arrow->children[2]);
  boolarray.move(arrow->children[3]);
  list_array.move(arrow->children[4]);

  int64_array  = get_nanoarrow_array<int64_t>(int64_data, validity);
  string_array = get_nanoarrow_array<cudf::string_view>(string_data, validity);
  int64_array.move(arrow->children[5]->children[0]);
  string_array.move(arrow->children[5]->children[1]);

  ArrowBitmap struct_validity;
  ArrowBitmapInit(&struct_validity);
  NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(&struct_validity, length));
  ArrowBitmapAppendInt8Unsafe(
    &struct_validity, reinterpret_cast<const int8_t*>(bool_data_validity.data()), length);
  arrow->children[5]->length = length;
  ArrowArraySetValidityBitmap(arrow->children[5], &struct_validity);
  arrow->children[5]->null_count =
    length - ArrowBitCountSet(ArrowArrayValidityBitmap(arrow->children[5])->buffer.data, 0, length);

  ArrowError error;
  if (ArrowArrayFinishBuilding(arrow.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, &error) !=
      NANOARROW_OK) {
    std::cerr << ArrowErrorMessage(&error) << std::endl;
    CUDF_FAIL("failed to build example arrays");
  }

  return std::make_tuple(
    std::make_unique<cudf::table>(std::move(columns)), std::move(schema), std::move(arrow));
}

struct FromArrowHostDeviceTest : public cudf::test::BaseFixture {};

template <typename T>
struct FromArrowHostDeviceTestDurationsTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FromArrowHostDeviceTestDurationsTest, cudf::test::DurationTypes);

TEST_F(FromArrowHostDeviceTest, EmptyTable)
{
  auto [tbl, schema, arr] = get_nanoarrow_host_tables(0);

  auto expected_cudf_table = tbl->view();
  ArrowDeviceArray input;
  memcpy(&input.array, arr.get(), sizeof(ArrowArray));
  input.device_id   = -1;
  input.device_type = ARROW_DEVICE_CPU;

  auto got_cudf_table = cudf::from_arrow_host(schema.get(), &input);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_cudf_table, got_cudf_table->view());
}

TEST_F(FromArrowHostDeviceTest, DateTimeTable)
{
  auto data = std::vector<int64_t>{1, 2, 3, 4, 5, 6};
  auto col  = cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(
    data.begin(), data.end());
  cudf::table_view expected_table_view({col});

  // construct equivalent arrow schema with nanoarrow
  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 1));
  ArrowSchemaInit(input_schema->children[0]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDateTime(
    input_schema->children[0], NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_MILLI, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));

  // equivalent arrow record batch
  nanoarrow::UniqueArray input_array;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
  input_array->length     = 6;
  input_array->null_count = 0;

  auto arr = get_nanoarrow_array<int64_t>(data);
  arr.move(input_array->children[0]);
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr));

  ArrowDeviceArray input;
  memcpy(&input.array, input_array.get(), sizeof(ArrowArray));
  input.device_id   = -1;
  input.device_type = ARROW_DEVICE_CPU;

  // test that we get the same cudf table as we expect by converting the
  // host arrow memory to a cudf table
  auto got_cudf_table = cudf::from_arrow_host(input_schema.get(), &input);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, got_cudf_table->view());

  // test that we get a cudf table with a single struct column that is equivalent
  // if we use from_arrow_host_column
  auto got_cudf_col = cudf::from_arrow_host_column(input_schema.get(), &input);
  EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
  auto got_cudf_col_view = got_cudf_col->view();
  cudf::table_view from_struct{
    std::vector<cudf::column_view>(got_cudf_col_view.child_begin(), got_cudf_col_view.child_end())};
  CUDF_TEST_EXPECT_TABLES_EQUAL(got_cudf_table->view(), from_struct);
}

TYPED_TEST(FromArrowHostDeviceTestDurationsTest, DurationTable)
{
  using T = TypeParam;
  if (cudf::type_to_id<TypeParam>() == cudf::type_id::DURATION_DAYS) { return; }

  auto data = {T{1}, T{2}, T{3}, T{4}, T{5}, T{6}};
  auto col  = cudf::test::fixed_width_column_wrapper<T>(data);

  cudf::table_view expected_table_view({col});
  const ArrowTimeUnit time_unit = [&] {
    switch (cudf::type_to_id<TypeParam>()) {
      case cudf::type_id::DURATION_SECONDS: return NANOARROW_TIME_UNIT_SECOND;
      case cudf::type_id::DURATION_MILLISECONDS: return NANOARROW_TIME_UNIT_MILLI;
      case cudf::type_id::DURATION_MICROSECONDS: return NANOARROW_TIME_UNIT_MICRO;
      case cudf::type_id::DURATION_NANOSECONDS: return NANOARROW_TIME_UNIT_NANO;
      default: CUDF_FAIL("Unsupported duration unit in arrow");
    }
  }();

  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 1));

  ArrowSchemaInit(input_schema->children[0]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDateTime(
    input_schema->children[0], NANOARROW_TYPE_DURATION, time_unit, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));

  nanoarrow::UniqueArray input_array;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
  input_array->length     = expected_table_view.num_rows();
  input_array->null_count = 0;

  auto arr = get_nanoarrow_array<T>(data);
  arr.move(input_array->children[0]);
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr));

  ArrowDeviceArray input;
  memcpy(&input.array, input_array.get(), sizeof(ArrowArray));
  input.device_id   = -1;
  input.device_type = ARROW_DEVICE_CPU;

  // converting arrow host memory to cudf table gives us the expected table
  auto got_cudf_table = cudf::from_arrow_host(input_schema.get(), &input);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, got_cudf_table->view());

  // converting to a cudf table with a single struct column gives us the expected
  // result column
  auto got_cudf_col = cudf::from_arrow_host_column(input_schema.get(), &input);
  EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
  auto got_cudf_col_view = got_cudf_col->view();
  cudf::table_view from_struct{
    std::vector<cudf::column_view>(got_cudf_col_view.child_begin(), got_cudf_col_view.child_end())};
  CUDF_TEST_EXPECT_TABLES_EQUAL(got_cudf_table->view(), from_struct);
}

TEST_F(FromArrowHostDeviceTest, NestedList)
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 3 != 0; });
  auto col = cudf::test::lists_column_wrapper<int64_t>(
    {{{{{1, 2}, valids}, {{3, 4}, valids}, {5}}, {{6}, {{7, 8, 9}, valids}}}, valids});
  cudf::table_view expected_table_view({col});

  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 1));

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(input_schema->children[0], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));
  input_schema->children[0]->flags = ARROW_FLAG_NULLABLE;

  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(input_schema->children[0]->children[0], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0]->children[0], "element"));
  input_schema->children[0]->children[0]->flags = 0;

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(
    input_schema->children[0]->children[0]->children[0], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaSetName(input_schema->children[0]->children[0]->children[0], "element"));
  input_schema->children[0]->children[0]->children[0]->flags = ARROW_FLAG_NULLABLE;

  // create the base arrow list array
  auto list_arr = get_nanoarrow_list_array<int64_t>({6, 7, 8, 9}, {0, 1, 4}, {1, 0, 1, 1});
  std::vector<int32_t> offset{0, 0, 2};

  // populate the bitmask we're going to use for the top level list
  ArrowBitmap mask;
  ArrowBitmapInit(&mask);
  NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(&mask, 2));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(&mask, 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(&mask, 1, 1));

  nanoarrow::UniqueArray input_array;
  EXPECT_EQ(NANOARROW_OK, ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
  input_array->length     = expected_table_view.num_rows();
  input_array->null_count = 0;

  ArrowArraySetValidityBitmap(input_array->children[0], &mask);
  input_array->children[0]->length     = expected_table_view.num_rows();
  input_array->children[0]->null_count = 1;
  auto offset_buf                      = ArrowArrayBuffer(input_array->children[0], 1);
  EXPECT_EQ(
    NANOARROW_OK,
    ArrowBufferAppend(
      offset_buf, reinterpret_cast<const void*>(offset.data()), offset.size() * sizeof(int32_t)));

  // move our base list to be the child of the one we just created
  // so that we now have an equivalent value to what we created for cudf
  list_arr.move(input_array->children[0]->children[0]);
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

  ArrowDeviceArray input;
  memcpy(&input.array, input_array.get(), sizeof(ArrowArray));
  input.device_id   = -1;
  input.device_type = ARROW_DEVICE_CPU;

  // converting from arrow host memory to cudf gives us the expected table
  auto got_cudf_table = cudf::from_arrow_host(input_schema.get(), &input);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, got_cudf_table->view());

  // converting to a single column cudf table gives us the expected struct column
  auto got_cudf_col = cudf::from_arrow_host_column(input_schema.get(), &input);
  EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
  auto got_cudf_col_view = got_cudf_col->view();
  cudf::table_view from_struct{
    std::vector<cudf::column_view>(got_cudf_col_view.child_begin(), got_cudf_col_view.child_end())};
  CUDF_TEST_EXPECT_TABLES_EQUAL(got_cudf_table->view(), from_struct);
}

TEST_F(FromArrowHostDeviceTest, StructColumn)
{
  // Create cudf table
  auto nested_type_field_names =
    std::vector<std::vector<std::string>>{{"string", "integral", "bool", "nested_list", "struct"}};
  auto str_col =
    cudf::test::strings_column_wrapper{
      "Samuel Vimes", "Carrot Ironfoundersson", "Angua von Überwald"}
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
  auto [null_mask, null_count] =
    cudf::bools_to_mask(cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}});
  auto sub_struct_col =
    cudf::make_structs_column(num_rows, std::move(cols2), null_count, std::move(*null_mask));
  vector_of_columns cols;
  cols.push_back(std::move(str_col));
  cols.push_back(std::move(int_col));
  cols.push_back(std::move(bool_col));
  cols.push_back(std::move(list_col));
  cols.push_back(std::move(sub_struct_col));

  auto struct_col = cudf::make_structs_column(num_rows, std::move(cols), 0, {});
  cudf::table_view expected_table_view({struct_col->view()});

  // Create name metadata
  auto sub_metadata          = cudf::column_metadata{"struct"};
  sub_metadata.children_meta = {{"string2"}, {"integral2"}};
  auto metadata              = cudf::column_metadata{"a"};
  metadata.children_meta     = {{"string"}, {"integral"}, {"bool"}, {"nested_list"}, sub_metadata};

  // create the equivalent arrow schema using nanoarrow
  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 1));

  ArrowSchemaInit(input_schema->children[0]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema->children[0], 5));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));
  input_schema->children[0]->flags = 0;

  auto child = input_schema->children[0];
  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(child->children[0], NANOARROW_TYPE_STRING));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[0], "string"));
  child->children[0]->flags = 0;

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(child->children[1], NANOARROW_TYPE_INT32));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[1], "integral"));
  child->children[1]->flags = 0;

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(child->children[2], NANOARROW_TYPE_BOOL));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[2], "bool"));
  child->children[2]->flags = 0;

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(child->children[3], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[3], "nested_list"));
  child->children[3]->flags = 0;
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(child->children[3]->children[0], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[3]->children[0], "element"));
  child->children[3]->children[0]->flags = 0;
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(child->children[3]->children[0]->children[0], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaSetName(child->children[3]->children[0]->children[0], "element"));
  child->children[3]->children[0]->children[0]->flags = 0;

  ArrowSchemaInit(child->children[4]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(child->children[4], 2));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[4], "struct"));

  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(child->children[4]->children[0], NANOARROW_TYPE_STRING));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[4]->children[0], "string2"));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(child->children[4]->children[1], NANOARROW_TYPE_INT32));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child->children[4]->children[1], "integral2"));

  // create nanoarrow table
  // first our underlying arrays
  std::vector<std::string> str{"Samuel Vimes", "Carrot Ironfoundersson", "Angua von Überwald"};
  std::vector<std::string> str2{"CUDF", "ROCKS", "EVERYWHERE"};
  auto str_array  = get_nanoarrow_array<cudf::string_view>(str);
  auto int_array  = get_nanoarrow_array<int32_t>({48, 27, 25});
  auto str2_array = get_nanoarrow_array<cudf::string_view>(str2, {0, 1, 0});
  auto int2_array = get_nanoarrow_array<int32_t, uint8_t>({12, 24, 47}, {1, 0, 1});
  auto bool_array = get_nanoarrow_array<bool>({true, true, false});
  auto list_arr =
    get_nanoarrow_list_array<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 2, 4, 5, 6, 7, 9});
  std::vector<int32_t> offset{0, 3, 4, 6};

  // create the struct array
  nanoarrow::UniqueArray input_array;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));

  input_array->length = expected_table_view.num_rows();

  auto array_a        = input_array->children[0];
  auto view_a         = expected_table_view.column(0);
  array_a->length     = view_a.size();
  array_a->null_count = view_a.null_count();
  // populate the children of our struct by moving them from the original arrays
  str_array.move(array_a->children[0]);
  int_array.move(array_a->children[1]);
  bool_array.move(array_a->children[2]);

  array_a->children[3]->length     = expected_table_view.num_rows();
  array_a->children[3]->null_count = 0;
  auto offset_buf                  = ArrowArrayBuffer(array_a->children[3], 1);
  EXPECT_EQ(
    NANOARROW_OK,
    ArrowBufferAppend(
      offset_buf, reinterpret_cast<const void*>(offset.data()), offset.size() * sizeof(int32_t)));

  list_arr.move(array_a->children[3]->children[0]);

  // set our struct bitmap validity mask
  ArrowBitmap mask;
  ArrowBitmapInit(&mask);
  NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(&mask, 3));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(&mask, 1, 2));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(&mask, 0, 1));

  auto array_struct = array_a->children[4];
  auto view_struct  = view_a.child(4);
  ArrowArraySetValidityBitmap(array_struct, &mask);
  array_struct->null_count = view_struct.null_count();
  array_struct->length     = view_struct.size();

  str2_array.move(array_struct->children[0]);
  int2_array.move(array_struct->children[1]);

  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

  ArrowDeviceArray input;
  memcpy(&input.array, input_array.get(), sizeof(ArrowArray));
  input.device_id   = -1;
  input.device_type = ARROW_DEVICE_CPU;

  // test we get the expected cudf::table from the arrow host memory data
  auto got_cudf_table = cudf::from_arrow_host(input_schema.get(), &input);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, got_cudf_table->view());

  // test we get the expected cudf struct column
  auto got_cudf_col = cudf::from_arrow_host_column(input_schema.get(), &input);
  EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
  auto got_cudf_col_view = got_cudf_col->view();
  cudf::table_view from_struct{
    std::vector<cudf::column_view>(got_cudf_col_view.child_begin(), got_cudf_col_view.child_end())};
  CUDF_TEST_EXPECT_TABLES_EQUAL(got_cudf_table->view(), from_struct);
}

TEST_F(FromArrowHostDeviceTest, DictionaryIndicesType)
{
  // test dictionary arrays with different index types
  // cudf asserts that the index type must be unsigned
  auto array1 =
    get_nanoarrow_dict_array<int64_t, uint8_t>({1, 2, 5, 7}, {0, 1, 2, 1, 3}, {1, 0, 1, 1, 1});
  auto array2 =
    get_nanoarrow_dict_array<int64_t, uint16_t>({1, 2, 5, 7}, {0, 1, 2, 1, 3}, {1, 0, 1, 1, 1});
  auto array3 =
    get_nanoarrow_dict_array<int64_t, uint64_t>({1, 2, 5, 7}, {0, 1, 2, 1, 3}, {1, 0, 1, 1, 1});

  // create equivalent cudf dictionary columns
  auto keys_col = cudf::test::fixed_width_column_wrapper<int64_t>({1, 2, 5, 7});
  auto ind1_col = cudf::test::fixed_width_column_wrapper<uint8_t>({0, 1, 2, 1, 3}, {1, 0, 1, 1, 1});
  auto ind2_col =
    cudf::test::fixed_width_column_wrapper<uint16_t>({0, 1, 2, 1, 3}, {1, 0, 1, 1, 1});
  auto ind3_col =
    cudf::test::fixed_width_column_wrapper<uint64_t>({0, 1, 2, 1, 3}, {1, 0, 1, 1, 1});

  vector_of_columns columns;
  columns.emplace_back(cudf::make_dictionary_column(keys_col, ind1_col));
  columns.emplace_back(cudf::make_dictionary_column(keys_col, ind2_col));
  columns.emplace_back(cudf::make_dictionary_column(keys_col, ind3_col));

  cudf::table expected_table(std::move(columns));

  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 3));

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(input_schema->children[0], NANOARROW_TYPE_UINT8));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));
  NANOARROW_THROW_NOT_OK(ArrowSchemaAllocateDictionary(input_schema->children[0]));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(input_schema->children[0]->dictionary, NANOARROW_TYPE_INT64));

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(input_schema->children[1], NANOARROW_TYPE_UINT16));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[1], "b"));
  NANOARROW_THROW_NOT_OK(ArrowSchemaAllocateDictionary(input_schema->children[1]));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(input_schema->children[1]->dictionary, NANOARROW_TYPE_INT64));

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(input_schema->children[2], NANOARROW_TYPE_UINT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[2], "c"));
  NANOARROW_THROW_NOT_OK(ArrowSchemaAllocateDictionary(input_schema->children[2]));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(input_schema->children[2]->dictionary, NANOARROW_TYPE_INT64));

  nanoarrow::UniqueArray input_array;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
  input_array->length     = expected_table.num_rows();
  input_array->null_count = 0;

  array1.move(input_array->children[0]);
  array2.move(input_array->children[1]);
  array3.move(input_array->children[2]);

  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

  ArrowDeviceArray input;
  memcpy(&input.array, input_array.get(), sizeof(ArrowArray));
  input.device_id   = -1;
  input.device_type = ARROW_DEVICE_CPU;

  // test we get the expected cudf table when we convert from Arrow host memory
  auto got_cudf_table = cudf::from_arrow_host(input_schema.get(), &input);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table.view(), got_cudf_table->view());

  // test we get the expected cudf::column as a struct column
  auto got_cudf_col = cudf::from_arrow_host_column(input_schema.get(), &input);
  EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
  auto got_cudf_col_view = got_cudf_col->view();
  cudf::table_view from_struct{
    std::vector<cudf::column_view>(got_cudf_col_view.child_begin(), got_cudf_col_view.child_end())};
  CUDF_TEST_EXPECT_TABLES_EQUAL(got_cudf_table->view(), from_struct);
}

void slice_host_nanoarrow(ArrowArray* arr, int64_t start, int64_t end)
{
  auto op = [&](ArrowArray* array) {
    // slicing only needs to happen at the top level of an array
    array->offset = start;
    array->length = end - start;
    if (array->null_count != 0) {
      array->null_count =
        array->length -
        ArrowBitCountSet(ArrowArrayValidityBitmap(array)->buffer.data, start, end - start);
    }
  };

  if (arr->n_children == 0) {
    op(arr);
    return;
  }

  // since we want to simulate a sliced table where the children are sliced,
  // we slice each individual child of the record batch
  arr->length = end - start;
  for (int64_t i = 0; i < arr->n_children; ++i) {
    op(arr->children[i]);
  }
}

struct FromArrowHostDeviceTestSlice
  : public FromArrowHostDeviceTest,
    public ::testing::WithParamInterface<std::tuple<cudf::size_type, cudf::size_type>> {};

TEST_P(FromArrowHostDeviceTestSlice, SliceTest)
{
  auto [table, schema, array] = get_nanoarrow_host_tables(10000);
  auto cudf_table_view        = table->view();
  auto const [start, end]     = GetParam();

  auto sliced_cudf_table   = cudf::slice(cudf_table_view, {start, end})[0];
  auto expected_cudf_table = cudf::table{sliced_cudf_table};
  slice_host_nanoarrow(array.get(), start, end);

  ArrowDeviceArray input;
  memcpy(&input.array, array.get(), sizeof(ArrowArray));
  input.device_id   = -1;
  input.device_type = ARROW_DEVICE_CPU;

  auto got_cudf_table = cudf::from_arrow_host(schema.get(), &input);
  if (got_cudf_table->num_rows() == 0 and sliced_cudf_table.num_rows() == 0) {
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_cudf_table.view(), got_cudf_table->view());

    auto got_cudf_col = cudf::from_arrow_host_column(schema.get(), &input);
    EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
    auto got_cudf_col_view = got_cudf_col->view();
    cudf::table_view from_struct{std::vector<cudf::column_view>(got_cudf_col_view.child_begin(),
                                                                got_cudf_col_view.child_end())};
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(got_cudf_table->view(), from_struct);
  } else {
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_cudf_table.view(), got_cudf_table->view());

    auto got_cudf_col = cudf::from_arrow_host_column(schema.get(), &input);
    EXPECT_EQ(got_cudf_col->type(), cudf::data_type{cudf::type_id::STRUCT});
    auto got_cudf_col_view = got_cudf_col->view();
    cudf::table_view from_struct{std::vector<cudf::column_view>(got_cudf_col_view.child_begin(),
                                                                got_cudf_col_view.child_end())};
    CUDF_TEST_EXPECT_TABLES_EQUAL(got_cudf_table->view(), from_struct);
  }
}

INSTANTIATE_TEST_CASE_P(FromArrowHostDeviceTest,
                        FromArrowHostDeviceTestSlice,
                        ::testing::Values(std::make_tuple(0, 10000),
                                          std::make_tuple(2912, 2915),
                                          std::make_tuple(100, 3000),
                                          std::make_tuple(0, 0),
                                          std::make_tuple(0, 3000),
                                          std::make_tuple(10000, 10000)));
