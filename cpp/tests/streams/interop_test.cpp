/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/interop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

struct ArrowTest : public cudf::test::BaseFixture {};

TEST_F(ArrowTest, ToArrow)
{
  int32_t const value{42};
  auto col = cudf::test::fixed_width_column_wrapper<int32_t>{{value}};
  cudf::table_view tbl{{col}};

  std::vector<cudf::column_metadata> metadata{{""}};
  cudf::to_arrow(tbl, metadata, cudf::test::get_default_stream());
}

TEST_F(ArrowTest, FromArrow)
{
  std::vector<int64_t> host_values = {1, 2, 3, 5, 6, 7, 8};
  std::vector<bool> host_validity  = {true, true, true, false, true, true, true};

  arrow::Int64Builder builder;
  auto status      = builder.AppendValues(host_values, host_validity);
  auto maybe_array = builder.Finish();
  auto array       = *maybe_array;

  auto field  = arrow::field("", arrow::int32());
  auto schema = arrow::schema({field});
  auto table  = arrow::Table::Make(schema, {array});
  cudf::from_arrow(*table, cudf::test::get_default_stream());
}

TEST_F(ArrowTest, ToArrowScalar)
{
  int32_t const value{42};
  auto cudf_scalar =
    cudf::make_fixed_width_scalar<int32_t>(value, cudf::test::get_default_stream());

  cudf::column_metadata metadata{""};
  cudf::to_arrow(*cudf_scalar, metadata, cudf::test::get_default_stream());
}

TEST_F(ArrowTest, FromArrowScalar)
{
  int32_t const value{42};
  auto arrow_scalar = arrow::MakeScalar(value);
  cudf::from_arrow(*arrow_scalar, cudf::test::get_default_stream());
}
