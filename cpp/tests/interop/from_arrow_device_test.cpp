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

struct FromArrowDeviceTest : public cudf::test::BaseFixture {};

template <typename T>
struct FromArrowDeviceTestDurationsTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FromArrowDeviceTestDurationsTest, cudf::test::DurationTypes);

TEST_F(FromArrowDeviceTest, FailConditions)
{
  // can't pass null for schema or device array
  EXPECT_THROW(cudf::from_arrow_device(nullptr, nullptr), cudf::logic_error);
  // can't pass null for device array
  ArrowSchema schema;
  EXPECT_THROW(cudf::from_arrow_device(&schema, nullptr), cudf::logic_error);
  // device_type must be CUDA/CUDA_HOST/CUDA_MANAGED
  // should fail with ARROW_DEVICE_CPU
  ArrowDeviceArray arr;
  arr.device_type = ARROW_DEVICE_CPU;
  EXPECT_THROW(cudf::from_arrow_device(&schema, &arr), cudf::logic_error);
}

TEST_F(FromArrowDeviceTest, EmptyTable)
{
  const auto [table, schema, arr] = get_nanoarrow_tables(0);

  auto expected_cudf_table = table->view();

  ArrowDeviceArray input;
  memcpy(&input.array, arr.get(), sizeof(ArrowArray));
  input.device_id   = rmm::get_current_cuda_device().value();
  input.device_type = ARROW_DEVICE_CUDA;
  input.sync_event  = nullptr;

  auto got_cudf_table = cudf::from_arrow_device(schema.get(), &input);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_cudf_table, *got_cudf_table);
}

TEST_F(FromArrowDeviceTest, DateTimeTable)
{
  auto data = std::vector<int64_t>{1, 2, 3, 4, 5, 6};
  auto col  = cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(
    data.begin(), data.end());

  cudf::table_view expected_table_view({col});

  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  ArrowSchemaSetTypeStruct(input_schema.get(), 1);
  ArrowSchemaInit(input_schema->children[0]);
  ArrowSchemaSetTypeDateTime(
    input_schema->children[0], NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_MILLI, nullptr);
  ArrowSchemaSetName(input_schema->children[0], "a");

  nanoarrow::UniqueArray input_array;
  ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr);
  input_array->length                  = 6;
  input_array->null_count              = 0;
  input_array->children[0]->length     = 6;
  input_array->children[0]->null_count = 0;
  ArrowBufferSetAllocator(ArrowArrayBuffer(input_array->children[0], 1), noop_alloc);
  ArrowArrayBuffer(input_array->children[0], 1)->data =
    const_cast<uint8_t*>(cudf::column_view(col).data<uint8_t>());
  ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr);

  ArrowDeviceArray input_device_array;
  input_device_array.device_id   = rmm::get_current_cuda_device().value();
  input_device_array.device_type = ARROW_DEVICE_CUDA;
  input_device_array.sync_event  = nullptr;
  memcpy(&input_device_array.array, input_array.get(), sizeof(ArrowArray));

  auto got_cudf_table_view = cudf::from_arrow_device(input_schema.get(), &input_device_array);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, *got_cudf_table_view);
}

TYPED_TEST(FromArrowDeviceTestDurationsTest, DurationTable)
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
  ArrowSchemaSetTypeStruct(input_schema.get(), 1);

  ArrowSchemaInit(input_schema->children[0]);
  ArrowSchemaSetTypeDateTime(
    input_schema->children[0], NANOARROW_TYPE_DURATION, time_unit, nullptr);
  ArrowSchemaSetName(input_schema->children[0], "a");

  auto data_ptr = expected_table_view.column(0).data<uint8_t>();
  nanoarrow::UniqueArray input_array;
  ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr);
  input_array->length                  = expected_table_view.num_rows();
  input_array->null_count              = 0;
  input_array->children[0]->length     = expected_table_view.num_rows();
  input_array->children[0]->null_count = 0;
  ArrowBufferSetAllocator(ArrowArrayBuffer(input_array->children[0], 1), noop_alloc);
  ArrowArrayBuffer(input_array->children[0], 1)->data = const_cast<uint8_t*>(data_ptr);
  ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr);

  ArrowDeviceArray input_device_array;
  input_device_array.device_id   = rmm::get_current_cuda_device().value();
  input_device_array.device_type = ARROW_DEVICE_CUDA;
  input_device_array.sync_event  = nullptr;
  memcpy(&input_device_array.array, input_array.get(), sizeof(ArrowArray));

  auto got_cudf_table_view = cudf::from_arrow_device(input_schema.get(), &input_device_array);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, *got_cudf_table_view);
}

TEST_F(FromArrowDeviceTest, NestedList) 
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 3 != 0; });
  auto col = cudf::test::lists_column_wrapper<int64_t>(
    {{{{{1, 2}, valids}, {{3, 4}, valids}, {5}}, {{6}, {{7, 8, 9}, valids}}}, valids});
  cudf::table_view expected_table_view({col});

  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  ArrowSchemaSetTypeStruct(input_schema.get(), 1);

  ArrowSchemaInitFromType(input_schema->children[0], NANOARROW_TYPE_LIST);
  ArrowSchemaSetName(input_schema->children[0], "a");
  input_schema->children[0]->flags = ARROW_FLAG_NULLABLE;

  ArrowSchemaInitFromType(input_schema->children[0]->children[0], NANOARROW_TYPE_LIST);
  ArrowSchemaSetName(input_schema->children[0]->children[0], "element");
  input_schema->children[0]->children[0]->flags = 0;

  ArrowSchemaInitFromType(input_schema->children[0]->children[0]->children[0],
                          NANOARROW_TYPE_INT64);
  ArrowSchemaSetName(input_schema->children[0]->children[0]->children[0], "element");
  input_schema->children[0]->children[0]->children[0]->flags = ARROW_FLAG_NULLABLE;

  nanoarrow::UniqueArray input_array;
  EXPECT_EQ(NANOARROW_OK,
            ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
  input_array->length = expected_table_view.num_rows();
  auto top_list          = input_array->children[0];
  cudf::lists_column_view lview{expected_table_view.column(0)};
  populate_list_from_col(top_list, lview);
  cudf::lists_column_view nested_view{lview.child()};
  populate_list_from_col(top_list->children[0], nested_view);
  populate_from_col<int64_t>(top_list->children[0]->children[0], nested_view.child());
  ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr);

  ArrowDeviceArray input_device_array;
  input_device_array.device_id   = rmm::get_current_cuda_device().value();
  input_device_array.device_type = ARROW_DEVICE_CUDA;
  input_device_array.sync_event  = nullptr;
  memcpy(&input_device_array.array, input_array.get(), sizeof(ArrowArray));

  auto got_cudf_table_view = cudf::from_arrow_device(input_schema.get(), &input_device_array);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table_view, *got_cudf_table_view);
}