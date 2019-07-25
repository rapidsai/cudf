/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <tests/utilities/cudf_test_fixtures.h>
#include <cudf/copying.hpp>
#include <cudf/table.hpp>
#include <tests/utilities/column_wrapper.cuh>

struct CopyErrorTest : GdfTest {};

TEST_F(CopyErrorTest, NullInput) {
  gdf_column input{};
  input.size = 10;
  input.data = 0;

  CUDF_EXPECT_THROW_MESSAGE(cudf::copy(input), "Null input data");
}

template <typename T>
struct CopyTest : GdfTest {};

using test_types =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_CASE(CopyTest, test_types);

TYPED_TEST(CopyTest, BasicCopy) {
  constexpr gdf_size_type source_size{1000};
  cudf::test::column_wrapper<TypeParam> source{
      source_size, [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }};
  gdf_column copy{};
  EXPECT_NO_THROW(copy = cudf::copy(source));
  EXPECT_TRUE(source == copy);
}

TYPED_TEST(CopyTest, NoNullMask) {
  constexpr gdf_size_type source_size{1000};
  // No null mask
  std::vector<TypeParam> data(source_size, TypeParam{0});
  std::iota(data.begin(), data.end(), TypeParam{0});
  cudf::test::column_wrapper<TypeParam> source{data};

  gdf_column copy{};
  EXPECT_NO_THROW(copy = cudf::copy(source));
  EXPECT_TRUE(source == copy);
}

TYPED_TEST(CopyTest, EmptyInput) {
  constexpr gdf_size_type source_size{0};
  cudf::test::column_wrapper<TypeParam> source{
      source_size, [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }};
  gdf_column copy{};
  EXPECT_NO_THROW(copy = cudf::copy(source));
  EXPECT_TRUE(source == copy);
}

TYPED_TEST(CopyTest, EmptyTable) {
  constexpr gdf_size_type source_size{0};

  cudf::test::column_wrapper<TypeParam> source0{
      source_size, [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }};

  cudf::test::column_wrapper<TypeParam> source1{
      source_size, [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }};

  cudf::table table{source0.get(), source1.get()};

  cudf::table copy;
  EXPECT_NO_THROW(copy = cudf::copy(table));
  EXPECT_TRUE(source0 == *copy.get_column(0));
  EXPECT_TRUE(source1 == *copy.get_column(1));
}

TYPED_TEST(CopyTest, TableNoNullMask) {
  constexpr gdf_size_type source_size{1000};
  cudf::test::column_wrapper<TypeParam> source0{
      source_size, [](gdf_index_type row) { return row; }};
  cudf::test::column_wrapper<TypeParam> source1{
      source_size, [](gdf_index_type row) { return row; }};

  cudf::table table{source0.get(), source1.get()};

  cudf::table copy;
  EXPECT_NO_THROW(copy = cudf::copy(table));
  EXPECT_TRUE(source0 == *copy.get_column(0));
  EXPECT_TRUE(source1 == *copy.get_column(1));
}

TYPED_TEST(CopyTest, TableCopy) {
  constexpr gdf_size_type source_size{1000};

  cudf::test::column_wrapper<TypeParam> source0{
      source_size, [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }};

  cudf::test::column_wrapper<TypeParam> source1{
      source_size, [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }};

  cudf::table table{source0.get(), source1.get()};

  cudf::table copy;
  EXPECT_NO_THROW(copy = cudf::copy(table));
  EXPECT_TRUE(source0 == *copy.get_column(0));
  EXPECT_TRUE(source1 == *copy.get_column(1));
}
