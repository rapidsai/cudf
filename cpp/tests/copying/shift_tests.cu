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

#include <tests/copying/copying_test_helper.hpp>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/scalar_wrapper.cuh>
#include <cudf/copying.hpp>
 
using cudf::test::column_wrapper;
using cudf::test::scalar_wrapper;

namespace
{

template <typename ColumnType>
column_wrapper<ColumnType> make_column_wrapper(
  std::vector<ColumnType> data,
  std::vector<gdf_valid_type> mask
)
{
  return column_wrapper<ColumnType>(
    data,
    [mask](gdf_size_type row){ return mask[row] != 0; }
  );
}

template <typename ColumnType>
column_wrapper<ColumnType> shift(
  gdf_index_type periods,
  column_wrapper<ColumnType> source,
  scalar_wrapper<ColumnType> fill_value
)
{
  gdf_column actual = cudf::shift(source, periods, fill_value);
  return column_wrapper<ColumnType>(actual);
}

template <typename ColumnType>
column_wrapper<ColumnType> shift_cpu(
  int periods,
  column_wrapper<ColumnType> source,
  scalar_wrapper<ColumnType> fill_value
){
  auto source_column = source.get();
  auto host = source.to_host();
  auto data_host = std::get<0>(host);
  std::vector<gdf_valid_type> valid_host = std::get<1>(host);

  auto size = source.size();

  return column_wrapper<ColumnType>(
    source.size(),
    [size, periods, fill_value, data_host](gdf_size_type row){
      return row < periods || row >= size + periods
        ? fill_value.value()
        : data_host[row - periods];
    },
    [size, periods, fill_value, valid_host](gdf_size_type row){
      return row < periods || row >= size + periods
        ? fill_value.is_valid()
        : cudf::util::bit_is_set(&valid_host.front(), row - periods);
    }
  );
}

template <typename ColumnType>
struct ShiftTest : public testing::Test {
  void test_shift(const int size, const int periods) {
    auto fill_value = scalar_wrapper<ColumnType>(0, true);
    auto source_column = create_random_column<ColumnType>(size);
    auto expect_column = shift_cpu(periods, source_column, fill_value);
    auto actual_column = shift(periods, source_column, fill_value);
  
    ASSERT_EQ(expect_column, actual_column);
  }
};

using TestTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_CASE(ShiftTest, TestTypes);

TYPED_TEST(ShiftTest, Positive)
{
  const int SIZE = 2^10;
  this->test_shift(SIZE, SIZE / 2);
}

TYPED_TEST(ShiftTest, Negative)
{
  const int SIZE = 2^10;
  this->test_shift(SIZE, SIZE / -2);
}

TYPED_TEST(ShiftTest, LowerBounds)
{
  const int SIZE = 2^10;
  this->test_shift(SIZE, -SIZE);
  this->test_shift(SIZE, -SIZE - 1);
}

TYPED_TEST(ShiftTest, UpperBounds)
{
  const int SIZE = 2^10;
  this->test_shift(SIZE, SIZE);
  this->test_shift(SIZE, SIZE + 1);
}

TYPED_TEST(ShiftTest, Zero)
{
  const int SIZE = 2^10;
  this->test_shift(SIZE, 0);
}

}