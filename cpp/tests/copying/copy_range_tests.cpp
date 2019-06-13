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

#include <cudf/copying.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/scalar_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

template <typename T>
struct CopyRangeTest : GdfTest 
{
  static constexpr gdf_size_type column_size{1000};

  void test(column_wrapper<T> &dest,
            column_wrapper<T> const &source,
            column_wrapper<T> const &expected,
            gdf_index_type out_begin, gdf_index_type out_end,
            gdf_index_type in_begin)
  {
    EXPECT_NO_THROW(cudf::copy_range(dest.get(), *source.get(),
                    out_begin, out_end, in_begin));

    EXPECT_TRUE(expected == dest);

    if (!(expected == dest)) {
      std::cout << "expected\n";
      expected.print();
      std::cout << expected.get()->null_count << "\n";
      std::cout << "dest\n";
      dest.print();
      std::cout << dest.get()->null_count << "\n";
      std::cout << "source\n";
      source.print();
      std::cout << source.get()->null_count << "\n";
    }
  }
};

using test_types =
  ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double,
                   cudf::bool8>;
TYPED_TEST_CASE(CopyRangeTest, test_types);

template <typename T>
struct original_value {
  T operator()(gdf_index_type row) { return static_cast<T>(0); }
};
template <typename T>
struct replacement_value {
  T operator()(gdf_index_type row) { return static_cast<T>(row); }
};
auto valid = [](gdf_index_type row) { return true; };
auto depends = [](gdf_index_type row) { return (row % 2 == 0); };

TYPED_TEST(CopyRangeTest, CopyWithNulls)
{
  using T = TypeParam;

  gdf_size_type size = this->column_size;
  gdf_index_type out_begin = 30;
  gdf_index_type out_end = size - 20;
  gdf_index_type in_begin = 9;
  gdf_index_type row_diff = in_begin - out_begin;

  column_wrapper<T> dest(size, original_value<T>{}, valid);
  
  this->test(
    dest,
    column_wrapper<T>(size, replacement_value<T>{}, depends),
    column_wrapper<T>(size, 
      [&](gdf_index_type row) { 
        return ((row >= out_begin) && (row < out_end)) ? 
          replacement_value<T>{}(row + row_diff) : original_value<T>{}(row);
      },
      [&](gdf_index_type row) { 
        return ((row >= out_begin) && (row < out_end)) ? 
          depends(row + row_diff) : true;
      }),
    out_begin, out_end, in_begin);
}

TYPED_TEST(CopyRangeTest, CopyNoNulls)
{
  using T = TypeParam;

  gdf_size_type size = this->column_size;
  gdf_index_type out_begin = 30;
  gdf_index_type out_end = size - 20;
  gdf_index_type in_begin = 9;
  gdf_index_type row_diff = in_begin - out_begin;

  column_wrapper<T> dest(size, original_value<T>{});

  // First set it as valid
  this->test(
    dest,
    column_wrapper<T>(size, replacement_value<T>{}),
    column_wrapper<T>(size, 
      [&](gdf_index_type row) { 
        return ((row >= out_begin) && (row < out_end)) ? 
          replacement_value<T>{}(row + row_diff) : original_value<T>{}(row);
      }),
    out_begin, out_end, in_begin);
}

struct CopyRangeErrorTest : GdfTest {};

TEST_F(CopyRangeErrorTest, InvalidColumn)
{
  column_wrapper<int32_t> source(100, replacement_value<int32_t>{});
  CUDF_EXPECT_THROW_MESSAGE(cudf::copy_range(nullptr, *source.get(), 0, 10, 0),
                            "Null gdf_column pointer");

  gdf_column bad_input;
  gdf_column_view(&bad_input, 0, 0, 0, GDF_INT32);
  // empty range == no-op, even on invalid output column...
  EXPECT_NO_THROW(cudf::copy_range(&bad_input, *source.get(), 0, 0, 0));

  // for zero-size column, non-empty range is out of bounds
  CUDF_EXPECT_THROW_MESSAGE(cudf::copy_range(&bad_input, *source.get(), 0, 10, 0),
                            "Range is out of bounds");

  // invalid data pointer
  bad_input.size = 20;
  CUDF_EXPECT_THROW_MESSAGE(cudf::copy_range(&bad_input, *source.get(), 0, 10, 0),
                            "Null column data with non-zero size");
}

TEST_F(CopyRangeErrorTest, InvalidRange)
{
  column_wrapper<int32_t> dest(100, original_value<int32_t>{});
  column_wrapper<int32_t> source(100, replacement_value<int32_t>{});
  CUDF_EXPECT_THROW_MESSAGE(cudf::copy_range(dest.get(), *source.get(), 0, 10, 95),
                            "Range is out of bounds");
  CUDF_EXPECT_THROW_MESSAGE(cudf::copy_range(dest.get(), *source.get(), 0, 110, 0),
                            "Range is out of bounds");
  CUDF_EXPECT_THROW_MESSAGE(cudf::copy_range(dest.get(), *source.get(), -10, 0, 0),
                            "Range is out of bounds");
  CUDF_EXPECT_THROW_MESSAGE(cudf::copy_range(dest.get(), *source.get(), 0, 10, -10),
                            "Range is out of bounds");
  CUDF_EXPECT_THROW_MESSAGE(cudf::copy_range(dest.get(), *source.get(), 10, 0, 0),
                            "Range is empty or reversed");
}

TEST_F(CopyRangeErrorTest, DTypeMismatch)
{
  column_wrapper<int32_t> dest(100, original_value<int32_t>{});
  column_wrapper<float> source(100, replacement_value<float>{});
  CUDF_EXPECT_THROW_MESSAGE(cudf::copy_range(dest.get(), *source.get(), 0, 10, 0),
                            "Data type mismatch");
}
