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

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <utilities/type_dispatcher.hpp>
#include <utilities/wrapper_types.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <bitset>
#include <cstdint>
#include <random>

template <typename T>
struct ColumnWrapperTest : public GdfTest {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution{1000, 10000};
  int random_size() { return distribution(generator); }
};

using TestingTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float,
                                      double, cudf::date32, cudf::date64,
                                      cudf::timestamp, cudf::category>;

TYPED_TEST_CASE(ColumnWrapperTest, TestingTypes);

template <typename T>
void test_column(cudf::test::column_wrapper<T> const& col,
                 std::vector<T> const& expected_values,
                 std::vector<gdf_valid_type> const& expected_bitmask =
                     std::vector<gdf_valid_type>{}) {
  gdf_column const* underlying_column = col.get();
  ASSERT_NE(nullptr, underlying_column);
  EXPECT_EQ(expected_values.size(),
            static_cast<size_t>(underlying_column->size));
  gdf_dtype expected_dtype = cudf::gdf_dtype_of<T>();
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

    // Ensure data on device matches expected
    rmm::device_vector<T> expected_device_data(expected_values);
    T const* actual_device_data =
        static_cast<T const*>(underlying_column->data);
    EXPECT_TRUE(thrust::equal(rmm::exec_policy()->on(0),
                              expected_device_data.begin(),
                              expected_device_data.end(), actual_device_data));

  } else {
    EXPECT_EQ(nullptr, underlying_column->data);
  }

  // Check that actual bitmask matchs expected
  if (expected_bitmask.size() > 0) {
    EXPECT_NE(nullptr, underlying_column->valid);

    gdf_size_type const expected_null_count =
        expected_values.size() -
        count_valid_bits_host(expected_bitmask, underlying_column->size);

    EXPECT_EQ(expected_null_count, col.null_count());

    // Ensure data on device matches expected
    rmm::device_vector<gdf_valid_type> expected_device_bitmask(
        expected_bitmask);
    EXPECT_TRUE(thrust::equal(
        rmm::exec_policy()->on(0), expected_device_bitmask.begin(),
        expected_device_bitmask.begin() +
            gdf_num_bitmask_elements(expected_values.size()),
        underlying_column->valid));

    // The last element in the bitmask has to be handled as a special case
    EXPECT_TRUE(std::equal(expected_bitmask.begin(),
                           expected_bitmask.begin() +
                               gdf_num_bitmask_elements(expected_values.size()),
                           actual_bitmask.begin()));

    // Only check the bits in the last mask that correspond to rows
    std::bitset<GDF_VALID_BITSIZE> expected_last_mask =
        expected_bitmask[gdf_num_bitmask_elements(expected_values.size()) - 1];
    std::bitset<GDF_VALID_BITSIZE> actual_last_mask =
        actual_bitmask[gdf_num_bitmask_elements(expected_values.size()) - 1];
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

  // Ensure operator== for column_wrapper to it's underlying gdf_column
  // returns true
  EXPECT_TRUE(col == *col.get());
}

TYPED_TEST(ColumnWrapperTest, SizeConstructor) {
  gdf_size_type const size{this->random_size()};
  cudf::test::column_wrapper<TypeParam> const col(size);
  std::vector<TypeParam> expected_values(size);
  test_column(col, expected_values);
}

TYPED_TEST(ColumnWrapperTest, SizeConstructorWithBitmask) {
  gdf_size_type const size{this->random_size()};
  cudf::test::column_wrapper<TypeParam> const col(size, true);
  std::vector<TypeParam> expected_values(size);
  std::vector<gdf_valid_type> expected_bitmask(gdf_valid_allocation_size(size));
  test_column(col, expected_values, expected_bitmask);
}

TYPED_TEST(ColumnWrapperTest, ValueBitmaskVectorConstructor) {
  gdf_size_type const size{this->random_size()};

  std::vector<TypeParam> expected_values(size);

  std::generate(expected_values.begin(), expected_values.end(),
                [this]() { return TypeParam(this->generator()); });

  std::vector<gdf_valid_type> expected_bitmask(gdf_valid_allocation_size(size),
                                               0xFF);

  cudf::test::column_wrapper<TypeParam> const col(expected_values,
                                                  expected_bitmask);

  test_column(col, expected_values, expected_bitmask);
}

TYPED_TEST(ColumnWrapperTest, ValueVectorConstructor) {
  gdf_size_type const size{this->random_size()};

  std::vector<TypeParam> expected_values(size);

  std::generate(expected_values.begin(), expected_values.end(),
                [this]() { return TypeParam(this->generator()); });

  cudf::test::column_wrapper<TypeParam> const col(expected_values);

  test_column(col, expected_values);
}

TYPED_TEST(ColumnWrapperTest, ValueBitInitConstructor) {
  gdf_size_type const size{this->random_size()};

  cudf::test::column_wrapper<TypeParam> col(
      size, [](auto row) { return static_cast<TypeParam>(row); },
      [](auto row) { return true; });

  std::vector<TypeParam> expected_values(size);
  std::generate(expected_values.begin(), expected_values.end(), [](){
    static cudf::detail::unwrapped_type_t<TypeParam> ut{0}; 
    return TypeParam{ut++};
  });

  std::vector<gdf_valid_type> expected_bitmask(gdf_valid_allocation_size(size),
                                               0xff);
  test_column(col, expected_values, expected_bitmask);
}

TYPED_TEST(ColumnWrapperTest, ValueVectorBitmaskInitConstructor) {
  gdf_size_type const size{this->random_size()};

  std::vector<TypeParam> expected_values(size);

  std::generate(expected_values.begin(), expected_values.end(),
                [this]() { return TypeParam(this->generator()); });

  // Every even bit is null
  std::vector<gdf_valid_type> expected_bitmask(gdf_valid_allocation_size(size),
                                               0xAA);

  auto even_bits_null = [](auto row) { return (row % 2); };

  cudf::test::column_wrapper<TypeParam> const col(expected_values,
                                                  even_bits_null);

  test_column(col, expected_values, expected_bitmask);
}

TYPED_TEST(ColumnWrapperTest, CopyConstructor) {
  gdf_size_type const size{this->random_size()};

  cudf::test::column_wrapper<TypeParam> const source(
      size, [](auto row) { return static_cast<TypeParam>(row); },
      [](auto row) { return true; });

  cudf::test::column_wrapper<TypeParam> copy(source);

  // Ensure the underlying columns are equal except for their data and valid
  // pointers
  gdf_column const* source_column = source.get();
  gdf_column const* copy_column = copy.get();
  ASSERT_NE(nullptr, source_column);
  ASSERT_NE(nullptr, copy_column);
  EXPECT_NE(source_column, copy_column);
  EXPECT_NE(source_column->data, copy_column->data);
  EXPECT_NE(source_column->valid, copy_column->valid);
  EXPECT_EQ(source_column->size, copy_column->size);
  EXPECT_EQ(source_column->dtype, copy_column->dtype);
  EXPECT_EQ(source_column->null_count, copy_column->null_count);
  EXPECT_EQ(source_column->dtype_info.time_unit,
            copy_column->dtype_info.time_unit);

  // Ensure device data and bitmasks are equal
  TypeParam* source_device_data = static_cast<TypeParam*>(source_column->data);
  TypeParam* copy_device_data = static_cast<TypeParam*>(copy_column->data);
  EXPECT_TRUE(thrust::equal(rmm::exec_policy()->on(0), source_device_data,
                            source_device_data + size, copy_device_data));

  EXPECT_TRUE(thrust::equal(rmm::exec_policy()->on(0), source_column->valid,
                            source_column->valid + gdf_num_bitmask_elements(size),
                            copy_column->valid));

  // Ensure to_host data is equal
  std::vector<TypeParam> source_data;
  std::vector<gdf_valid_type> source_bitmask;
  std::tie(source_data, source_bitmask) = source.to_host();

  std::vector<TypeParam> copy_data;
  std::vector<gdf_valid_type> copy_bitmask;
  std::tie(copy_data, copy_bitmask) = copy.to_host();

  EXPECT_TRUE(
      std::equal(source_data.begin(), source_data.end(), copy_data.begin()));

  EXPECT_TRUE(std::equal(source_bitmask.begin(), source_bitmask.end(),
                         copy_bitmask.begin()));

  // Ensure operator== works
  EXPECT_TRUE(source == copy);
}

TYPED_TEST(ColumnWrapperTest, EqualColumnsNoNulls) {
  gdf_size_type const size{this->random_size()};
  std::vector<TypeParam> values(size);
  std::generate(values.begin(), values.end(),
                [this]() { return TypeParam(this->generator()); });

  cudf::test::column_wrapper<TypeParam> const col1(values);
  cudf::test::column_wrapper<TypeParam> const col2(values);

  EXPECT_TRUE(col1 == col2);
}

TYPED_TEST(ColumnWrapperTest, EqualColumnsWithNulls) {
  gdf_size_type const size{this->random_size()};
  std::vector<TypeParam> values(size);
  std::generate(values.begin(), values.end(),
                [this]() { return TypeParam(this->generator()); });

  auto even_bits_null = [](auto row) { return row % 2; };

  cudf::test::column_wrapper<TypeParam> const col1(values, even_bits_null);
  cudf::test::column_wrapper<TypeParam> const col2(values, even_bits_null);

  EXPECT_TRUE(col1 == col2);
}

TYPED_TEST(ColumnWrapperTest, UnEqualColumnsNoNulls) {
  gdf_size_type const size{this->random_size()};
  std::vector<TypeParam> values1(size);
  std::generate(values1.begin(), values1.end(),
                [this]() { return TypeParam(this->generator()); });

  std::vector<TypeParam> values2(values1);
  std::shuffle(values2.begin(), values2.end(), this->generator);

  cudf::test::column_wrapper<TypeParam> const col1(values1);
  cudf::test::column_wrapper<TypeParam> const col2(values2);

  EXPECT_FALSE(col1 == col2);
}

TYPED_TEST(ColumnWrapperTest, UnEqualColumnsWithNulls) {
  gdf_size_type const size{this->random_size()};
  std::vector<TypeParam> values(size);
  std::generate(values.begin(), values.end(),
                [this]() { return TypeParam(this->generator()); });

  auto even_bits_null = [](auto row) { return row % 2; };
  auto odd_bits_null = [](auto row) { return (row + 1) % 2; };

  cudf::test::column_wrapper<TypeParam> const col1(values, even_bits_null);
  cudf::test::column_wrapper<TypeParam> const col2(values, odd_bits_null);

  EXPECT_FALSE(col1 == col2);
}

TYPED_TEST(ColumnWrapperTest, AllInvalid) {
  gdf_size_type const size{this->random_size()};
  std::vector<TypeParam> values(size);
  std::generate(values.begin(), values.end(),
                [this]() { return TypeParam(this->generator()); });

  auto all_null = [](auto row) { return false; };

  cudf::test::column_wrapper<TypeParam> const col(values, all_null);

  std::vector<TypeParam> data;
  std::vector<gdf_valid_type> bitmask;
  std::tie(data, bitmask) = col.to_host();

  for (gdf_size_type i = 0; i < size; ++i) {
    EXPECT_FALSE(gdf_is_valid(bitmask.data(), i));
  }
}

TYPED_TEST(ColumnWrapperTest, AllValid) {
  gdf_size_type const size{this->random_size()};
  std::vector<TypeParam> values(size);
  std::generate(values.begin(), values.end(),
                [this]() { return TypeParam(this->generator()); });

  auto all_valid = [](auto row) { return true; };

  cudf::test::column_wrapper<TypeParam> const col(values, all_valid);

  std::vector<TypeParam> data;
  std::vector<gdf_valid_type> bitmask;
  std::tie(data, bitmask) = col.to_host();

  for (gdf_size_type i = 0; i < size; ++i) {
    EXPECT_TRUE(gdf_is_valid(bitmask.data(), i));
  }
}
