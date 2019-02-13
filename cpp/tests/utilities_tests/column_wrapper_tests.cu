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

#include "tests/utilities/column_wrapper.cuh"
#include "tests/utilities/cudf_test_fixtures.h"
#include "utilities/type_dispatcher.hpp"
#include "utilities/wrapper_types.hpp"

#include <cstdint>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

template <typename T>
struct ColumnWrapperTest : public GdfTest {};

using TestingTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float,
                                      double, cudf::date32, cudf::date64,
                                      cudf::timestamp, cudf::category>;

TYPED_TEST_CASE(ColumnWrapperTest, TestingTypes);

TYPED_TEST(ColumnWrapperTest, SizeConstructor) {
  gdf_size_type const size{1000};
  cudf::test::column_wrapper<TypeParam> const col(size);

  gdf_column const* underlying_column = col.get();
  ASSERT_NE(nullptr, underlying_column);
  EXPECT_NE(nullptr, underlying_column->data);
  EXPECT_EQ(nullptr, underlying_column->valid);
  EXPECT_EQ(size, underlying_column->size);
  gdf_dtype expected = cudf::type_to_gdf_dtype<TypeParam>::value;
  EXPECT_EQ(expected, underlying_column->dtype);

  std::vector<TypeParam> col_data;
  std::vector<gdf_valid_type> col_bitmask;
  std::tie(col_data, col_bitmask) = col.to_host();
  EXPECT_EQ(static_cast<size_t>(size), col_data.size());
  EXPECT_EQ(0u, col_bitmask.size());

  bool const all_default_values{
      std::all_of(col_data.begin(), col_data.end(),
                  [](TypeParam element) { return (TypeParam{} == element); })};
  EXPECT_TRUE(all_default_values);
}