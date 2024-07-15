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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <iostream>

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;

struct BaseToArrowHostFixture : public cudf::test::BaseFixture {
  void compare_arrays(ArrowArrayView const* expected, ArrowArrayView const* actual)
  {
    EXPECT_EQ(expected->length, actual->length);
    EXPECT_EQ(expected->null_count, actual->null_count);
    EXPECT_EQ(expected->offset, actual->offset);
    EXPECT_EQ(expected->n_children, actual->n_children);

    for (int64_t i = 0; i < actual->array->n_buffers; ++i) {
      SCOPED_TRACE("buffer " + std::to_string(i));
      auto expected_buf = expected->buffer_views[i];
      auto actual_buf   = actual->buffer_views[i];

      EXPECT_TRUE(
        0 == std::memcmp(expected_buf.data.data, actual_buf.data.data, expected_buf.size_bytes));
    }

    if (expected->dictionary != nullptr) {
      EXPECT_NE(nullptr, actual->dictionary);
      SCOPED_TRACE("dictionary");
      compare_arrays(expected->dictionary, actual->dictionary);
    } else {
      EXPECT_EQ(nullptr, actual->dictionary);
    }

    if (expected->n_children == 0) {
      EXPECT_EQ(nullptr, actual->children);
    } else {
      for (int64_t i = 0; i < expected->n_children; ++i) {
        SCOPED_TRACE("child " + std::to_string(i));
        compare_arrays(expected->children[i], actual->children[i]);
      }
    }
  }
};

struct ToArrowHostDeviceTest : public BaseToArrowHostFixture {};
template <typename T>
struct ToArrowHostDeviceTestDurationsTest : public BaseToArrowHostFixture {};

TYPED_TEST_SUITE(ToArrowHostDeviceTestDurationsTest, cudf::test::DurationTypes);

TEST_F(ToArrowHostDeviceTest, EmptyTable)
{
  auto [tbl, schema, arr] = get_nanoarrow_host_tables(0);

  auto got_arrow_host = cudf::to_arrow_host(tbl->view());
  EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
  EXPECT_EQ(-1, got_arrow_host->device_id);
  EXPECT_EQ(nullptr, got_arrow_host->sync_event);

  ArrowArrayView expected, actual;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, arr.get(), nullptr));

  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  compare_arrays(&expected, &actual);

  ArrowArrayViewReset(&expected);
  ArrowArrayViewReset(&actual);
}

TEST_F(ToArrowHostDeviceTest, DateTimeTable)
{
  auto data = std::initializer_list<int64_t>{1, 2, 3, 4, 5, 6};
  auto col =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(data);
  cudf::table_view input_view({col});

  nanoarrow::UniqueSchema expected_schema;
  ArrowSchemaInit(expected_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));
  ArrowSchemaInit(expected_schema->children[0]);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDateTime(
    expected_schema->children[0], NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_MILLI, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
  expected_schema->children[0]->flags = 0;

  auto got_arrow_host = cudf::to_arrow_host(input_view);
  EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
  EXPECT_EQ(-1, got_arrow_host->device_id);
  EXPECT_EQ(nullptr, got_arrow_host->sync_event);

  ArrowArrayView expected, actual;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
  expected.length              = data.size();
  expected.children[0]->length = data.size();
  ArrowArrayViewSetLength(expected.children[0], data.size());
  expected.children[0]->buffer_views[0].data.data  = nullptr;
  expected.children[0]->buffer_views[0].size_bytes = 0;
  expected.children[0]->buffer_views[1].data.data  = data.begin();

  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  compare_arrays(&expected, &actual);
  ArrowArrayViewReset(&actual);

  got_arrow_host = cudf::to_arrow_host(input_view.column(0));
  NANOARROW_THROW_NOT_OK(
    ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  BaseToArrowHostFixture::compare_arrays(expected.children[0], &actual);
  ArrowArrayViewReset(&actual);

  ArrowArrayViewReset(&expected);
  ArrowArrayViewReset(&actual);
}

TYPED_TEST(ToArrowHostDeviceTestDurationsTest, DurationTable)
{
  using T = TypeParam;

  if (cudf::type_to_id<TypeParam>() == cudf::type_id::DURATION_DAYS) { return; }

  auto data = {T{1}, T{2}, T{3}, T{4}, T{5}, T{6}};
  auto col  = cudf::test::fixed_width_column_wrapper<T>(data);

  cudf::table_view input_view({col});

  nanoarrow::UniqueSchema expected_schema;
  ArrowSchemaInit(expected_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));

  ArrowSchemaInit(expected_schema->children[0]);
  const ArrowTimeUnit arrow_unit = [&] {
    switch (cudf::type_to_id<TypeParam>()) {
      case cudf::type_id::DURATION_SECONDS: return NANOARROW_TIME_UNIT_SECOND;
      case cudf::type_id::DURATION_MILLISECONDS: return NANOARROW_TIME_UNIT_MILLI;
      case cudf::type_id::DURATION_MICROSECONDS: return NANOARROW_TIME_UNIT_MICRO;
      case cudf::type_id::DURATION_NANOSECONDS: return NANOARROW_TIME_UNIT_NANO;
      default: CUDF_FAIL("Unsupported duration unit in arrow");
    }
  }();
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDateTime(
    expected_schema->children[0], NANOARROW_TYPE_DURATION, arrow_unit, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
  expected_schema->children[0]->flags = 0;

  auto got_arrow_host = cudf::to_arrow_host(input_view);
  EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
  EXPECT_EQ(-1, got_arrow_host->device_id);
  EXPECT_EQ(nullptr, got_arrow_host->sync_event);

  ArrowArrayView expected, actual;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));

  expected.length              = data.size();
  expected.children[0]->length = data.size();
  ArrowArrayViewSetLength(expected.children[0], data.size());
  expected.children[0]->buffer_views[0].data.data  = nullptr;
  expected.children[0]->buffer_views[0].size_bytes = 0;
  expected.children[0]->buffer_views[1].data.data  = data.begin();

  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  BaseToArrowHostFixture::compare_arrays(&expected, &actual);
  ArrowArrayViewReset(&actual);

  got_arrow_host = cudf::to_arrow_host(input_view.column(0));
  NANOARROW_THROW_NOT_OK(
    ArrowArrayViewInitFromSchema(&actual, expected_schema->children[0], nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  BaseToArrowHostFixture::compare_arrays(expected.children[0], &actual);
  ArrowArrayViewReset(&actual);

  ArrowArrayViewReset(&expected);
}

TEST_F(ToArrowHostDeviceTest, NestedList)
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 3 != 0; });
  auto col = cudf::test::lists_column_wrapper<int64_t>(
    {{{{{1, 2}, valids}, {{3, 4}, valids}, {5}}, {{6}, {{7, 8, 9}, valids}}}, valids});
  cudf::table_view input_view({col});

  nanoarrow::UniqueSchema expected_schema;
  ArrowSchemaInit(expected_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(expected_schema.get(), 1));

  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(expected_schema->children[0], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0], "a"));
  expected_schema->children[0]->flags = ARROW_FLAG_NULLABLE;

  NANOARROW_THROW_NOT_OK(
    ArrowSchemaInitFromType(expected_schema->children[0]->children[0], NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(expected_schema->children[0]->children[0], "element"));
  expected_schema->children[0]->children[0]->flags = 0;

  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(
    expected_schema->children[0]->children[0]->children[0], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaSetName(expected_schema->children[0]->children[0]->children[0], "element"));
  expected_schema->children[0]->children[0]->children[0]->flags = ARROW_FLAG_NULLABLE;

  auto got_arrow_host = cudf::to_arrow_host(input_view);
  EXPECT_EQ(ARROW_DEVICE_CPU, got_arrow_host->device_type);
  EXPECT_EQ(-1, got_arrow_host->device_id);
  EXPECT_EQ(nullptr, got_arrow_host->sync_event);

  auto list_arr = get_nanoarrow_list_array<int64_t>({6, 7, 8, 9}, {0, 1, 4}, {1, 0, 1, 1});
  std::vector<int32_t> offset{0, 0, 2};

  ArrowBitmap mask;
  ArrowBitmapInit(&mask);
  NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(&mask, 2));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(&mask, 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(&mask, 1, 1));

  nanoarrow::UniqueArray expected_arr;
  EXPECT_EQ(NANOARROW_OK,
            ArrowArrayInitFromSchema(expected_arr.get(), expected_schema.get(), nullptr));
  expected_arr->length     = input_view.num_rows();
  expected_arr->null_count = 0;

  ArrowArraySetValidityBitmap(expected_arr->children[0], &mask);
  expected_arr->children[0]->length     = input_view.num_rows();
  expected_arr->children[0]->null_count = 1;
  auto offset_buf                       = ArrowArrayBuffer(expected_arr->children[0], 1);
  EXPECT_EQ(
    NANOARROW_OK,
    ArrowBufferAppend(
      offset_buf, reinterpret_cast<void const*>(offset.data()), offset.size() * sizeof(int32_t)));
  list_arr.move(expected_arr->children[0]->children[0]);
  NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(expected_arr.get(), nullptr));

  ArrowArrayView expected, actual;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&expected, expected_schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&expected, expected_arr.get(), nullptr));  

  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&actual, expected_schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&actual, &got_arrow_host->array, nullptr));
  compare_arrays(&expected, &actual);

  ArrowArrayViewReset(&actual);
  ArrowArrayViewReset(&expected);
}
