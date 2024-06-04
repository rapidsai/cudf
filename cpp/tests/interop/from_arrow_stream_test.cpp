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
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <thrust/iterator/counting_iterator.h>

// create a cudf::table and equivalent arrow table with host memory
std::tuple<std::unique_ptr<cudf::table>, nanoarrow::UniqueSchema, nanoarrow::UniqueArray>
get_nanoarrow_host_tables_for_stream(cudf::size_type length)
{
  auto [table, schema, test_data] = get_nanoarrow_cudf_table(length);

  auto int64_array = get_nanoarrow_array<int64_t>(test_data.int64_data, test_data.validity);
  auto string_array =
    get_nanoarrow_array<cudf::string_view>(test_data.string_data, test_data.validity);
  cudf::dictionary_column_view view(table->get_column(2).view());
  auto keys       = cudf::test::to_host<int64_t>(view.keys()).first;
  auto indices    = cudf::test::to_host<uint32_t>(view.indices()).first;
  auto dict_array = get_nanoarrow_dict_array(std::vector<int64_t>(keys.begin(), keys.end()),
                                             std::vector<int32_t>(indices.begin(), indices.end()),
                                             test_data.validity);
  auto boolarray  = get_nanoarrow_array<bool>(test_data.bool_data, test_data.bool_validity);
  auto list_array = get_nanoarrow_list_array<int64_t>(test_data.list_int64_data,
                                                      test_data.list_offsets,
                                                      test_data.list_int64_data_validity,
                                                      test_data.bool_data_validity);

  nanoarrow::UniqueArray arrow;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(arrow.get(), schema.get(), nullptr));
  arrow->length = length;

  int64_array.move(arrow->children[0]);
  string_array.move(arrow->children[1]);
  dict_array.move(arrow->children[2]);
  boolarray.move(arrow->children[3]);
  list_array.move(arrow->children[4]);

  int64_array  = get_nanoarrow_array<int64_t>(test_data.int64_data, test_data.validity);
  string_array = get_nanoarrow_array<cudf::string_view>(test_data.string_data, test_data.validity);
  int64_array.move(arrow->children[5]->children[0]);
  string_array.move(arrow->children[5]->children[1]);

  ArrowBitmap struct_validity;
  ArrowBitmapInit(&struct_validity);
  NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(&struct_validity, length));
  ArrowBitmapAppendInt8Unsafe(
    &struct_validity, reinterpret_cast<const int8_t*>(test_data.bool_data_validity.data()), length);
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

  return std::make_tuple(std::move(table), std::move(schema), std::move(arrow));
}

static void null_release_array(ArrowArray* stream) {}

struct VectorOfArrays {
  std::vector<nanoarrow::UniqueArray> arrays;
  nanoarrow::UniqueSchema schema;
  size_t index{0};

  static int get_schema(ArrowArrayStream* stream, ArrowSchema* out_schema)
  {
    auto private_data = static_cast<VectorOfArrays*>(stream->private_data);
    // TODO: Can the deep copy be avoided here? I tried creating a new schema
    // with a shallow copy of the fields and seeing the release function to a
    // no-op, but that resulted in the children being freed somehow in the
    // EmptyTest (I didn't investigate further).
    ArrowSchemaDeepCopy(private_data->schema.get(), out_schema);

    return 0;
  }

  static int get_next(ArrowArrayStream* stream, ArrowArray* out_array)
  {
    auto private_data = static_cast<VectorOfArrays*>(stream->private_data);
    if (private_data->index >= private_data->arrays.size()) {
      out_array->release = nullptr;
      return 0;
    }
    auto ret_array = private_data->arrays[private_data->index++].get();
    // TODO: This shallow copy seems to work, but is it safe, especially with
    // respect to the children? I believe we should be safe from double-freeing
    // because everything will check for a null release pointer before freeing,
    // but that could produce use-after-free bugs especially with the children
    // since that pointer is just copied over. My current tests won't reflect
    // that but creating multiple streams from the same set of arrays would.
    // Is that even a valid use case?
    out_array->length     = ret_array->length;
    out_array->null_count = ret_array->null_count;
    out_array->offset     = ret_array->offset;
    out_array->n_buffers  = ret_array->n_buffers;
    out_array->buffers    = ret_array->buffers;
    out_array->n_children = ret_array->n_children;
    out_array->children   = ret_array->children;
    out_array->dictionary = ret_array->dictionary;
    out_array->release    = null_release_array;

    return 0;
  }

  static const char* get_last_error(ArrowArrayStream* stream) { return nullptr; }

  static void release(ArrowArrayStream* stream)
  {
    delete static_cast<VectorOfArrays*>(stream->private_data);
  }
};

struct FromArrowStreamTest : public cudf::test::BaseFixture {};

void makeStreamFromArrays(std::vector<nanoarrow::UniqueArray> arrays,
                          nanoarrow::UniqueSchema schema,
                          ArrowArrayStream* out)
{
  auto* private_data  = new VectorOfArrays{std::move(arrays), std::move(schema)};
  out->get_schema     = VectorOfArrays::get_schema;
  out->get_next       = VectorOfArrays::get_next;
  out->get_last_error = VectorOfArrays::get_last_error;
  out->release        = VectorOfArrays::release;
  out->private_data   = private_data;
}

TEST_F(FromArrowStreamTest, BasicTest)
{
  constexpr auto num_copies = 3;
  std::vector<std::unique_ptr<cudf::table>> tables;
  // The schema is unique across all tables.
  nanoarrow::UniqueSchema schema;
  std::vector<nanoarrow::UniqueArray> arrays;
  for (auto i = 0; i < num_copies; ++i) {
    auto [tbl, sch, arr] = get_nanoarrow_host_tables_for_stream(0);
    tables.push_back(std::move(tbl));
    arrays.push_back(std::move(arr));
    if (i == 0) { sch.move(schema.get()); }
  }
  std::vector<cudf::table_view> table_views;
  for (auto const& table : tables) {
    table_views.push_back(table->view());
  }
  auto expected = cudf::concatenate(table_views);

  ArrowArrayStream stream;
  makeStreamFromArrays(std::move(arrays), std::move(schema), &stream);
  auto result = cudf::from_arrow_stream(&stream);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

TEST_F(FromArrowStreamTest, EmptyTest)
{
  auto [tbl, sch, arr] = get_nanoarrow_host_tables_for_stream(0);
  std::vector<cudf::table_view> table_views{tbl->view()};
  auto expected = cudf::concatenate(table_views);

  ArrowArrayStream stream;
  makeStreamFromArrays({}, std::move(sch), &stream);
  auto result = cudf::from_arrow_stream(&stream);
  cudf::have_same_types(expected->view(), result->view());
}
