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
#include <cudf_test/table_utilities.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_checks.hpp>

struct VectorOfArrays {
  std::vector<nanoarrow::UniqueArray> arrays;
  nanoarrow::UniqueSchema schema;
  size_t index{0};

  static int get_schema(ArrowArrayStream* stream, ArrowSchema* out_schema)
  {
    auto private_data = static_cast<VectorOfArrays*>(stream->private_data);

    [[maybe_unused]] auto rc = ArrowSchemaDeepCopy(private_data->schema.get(), out_schema);
    return 0;
  }

  static int get_next(ArrowArrayStream* stream, ArrowArray* out_array)
  {
    auto private_data = static_cast<VectorOfArrays*>(stream->private_data);
    if (private_data->index >= private_data->arrays.size()) {
      out_array->release = nullptr;
      return 0;
    }
    ArrowArrayMove(private_data->arrays[private_data->index++].get(), out_array);
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
    auto [tbl, sch, arr] = get_nanoarrow_host_tables(0);
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
  auto [tbl, sch, arr] = get_nanoarrow_host_tables(0);
  std::vector<cudf::table_view> table_views{tbl->view()};
  auto expected = cudf::concatenate(table_views);

  ArrowArrayStream stream;
  makeStreamFromArrays({}, std::move(sch), &stream);
  auto result = cudf::from_arrow_stream(&stream);
  cudf::have_same_types(expected->view(), result->view());
}
