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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <bitset>
#include <cstdint>

template <typename T>
struct ColumnWrapperTest : public GdfTest {};

using TestingTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float,
                                      double, cudf::date32, cudf::date64,
                                      cudf::timestamp, cudf::category>;

TYPED_TEST_CASE(ColumnWrapperTest, TestingTypes);

template <typename T>
void test_constructor(cudf::test::column_wrapper<T> const& col,
                      std::vector<T> const& expected_values,
                      std::vector<gdf_valid_type> const& expected_bitmask =
                          std::vector<gdf_valid_type>{}) {
  gdf_column const* underlying_column = col.get();
  ASSERT_NE(nullptr, underlying_column);
  EXPECT_EQ(expected_values.size(),
            static_cast<size_t>(underlying_column->size));
  gdf_dtype expected_dtype = cudf::type_to_gdf_dtype<T>::value;
  EXPECT_EQ(expected_dtype, underlying_column->dtype);

  std::vector<T> actual_values;
  std::vector<gdf_valid_type> actual_bitmask;

  std::tie(actual_values, actual_bitmask) = col.to_host();
  EXPECT_EQ(expected_values.size(), actual_values.size());
  EXPECT_EQ(expected_bitmask.size(), actual_bitmask.size());

  // Check the actual values matchs expected
  if (expected_values.size() > 0) {
    EXPECT_NE(nullptr, underlying_column->data);
    EXPECT_TRUE(std::equal(expected_values.begin(), expected_values.end(),
                           actual_values.begin()));
  } else {
    EXPECT_EQ(nullptr, underlying_column->data);
  }

  // Check that actual bitmask matchs expected
  if (expected_bitmask.size() > 0) {
    EXPECT_NE(nullptr, underlying_column->valid);
    // The last element in the bitmask has to be handled as a special case
    EXPECT_TRUE(std::equal(expected_bitmask.begin(), expected_bitmask.end() - 1,
                           actual_bitmask.begin()));
    std::bitset<GDF_VALID_BITSIZE> expected_last_mask{expected_bitmask.back()};
    std::bitset<GDF_VALID_BITSIZE> actual_last_mask{actual_bitmask.back()};

    gdf_size_type valid_bits_last_mask =
        expected_values.size() % GDF_VALID_BITSIZE;
    if (0 == valid_bits_last_mask) {
      valid_bits_last_mask = GDF_VALID_BITSIZE;
    }

    for (gdf_size_type i = 0; i < valid_bits_last_mask; ++i) {
      EXPECT_EQ(expected_last_mask[i], actual_last_mask[i]);
    }
  } else {
    EXPECT_EQ(nullptr, underlying_column->valid);
    EXPECT_EQ(0, underlying_column->null_count);
  }
}

TYPED_TEST(ColumnWrapperTest, SizeConstructor) {
  gdf_size_type const size{1000};
  cudf::test::column_wrapper<TypeParam> const col(size);
  std::vector<TypeParam> expected_values(size);
  test_constructor(col, expected_values);
}

TYPED_TEST(ColumnWrapperTest, ValueBitInitConstructor) {
  gdf_size_type const size{1000};
  cudf::test::column_wrapper<TypeParam> col(
      size, [](auto row) { return static_cast<TypeParam>(row); },
      [](auto row) { return true; });

  gdf_column const* underlying_column = col.get();
  ASSERT_NE(nullptr, underlying_column);
  EXPECT_NE(nullptr, underlying_column->data);
  EXPECT_NE(nullptr, underlying_column->valid);
  EXPECT_EQ(size, underlying_column->size);
  gdf_dtype expected = cudf::type_to_gdf_dtype<TypeParam>::value;
  EXPECT_EQ(expected, underlying_column->dtype);

  std::vector<TypeParam> col_data;
  std::vector<gdf_valid_type> col_bitmask;
  std::tie(col_data, col_bitmask) = col.to_host();
  EXPECT_EQ(static_cast<size_t>(size), col_data.size());
  EXPECT_EQ(static_cast<size_t>(gdf_get_num_chars_bitmask(size)),
            col_bitmask.size());
}
