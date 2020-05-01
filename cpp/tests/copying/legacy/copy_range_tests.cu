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

#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <cudf/legacy/copying.hpp>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/column_wrapper_factory.hpp>
#include <tests/utilities/legacy/nvcategory_utils.cuh>
#include <tests/utilities/legacy/scalar_wrapper.cuh>

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

template <typename T>
struct CopyRangeTest : GdfTest {
  static constexpr cudf::size_type column_size{1000};

  void test(column_wrapper<T> const &dest,
            column_wrapper<T> const &source,
            column_wrapper<T> const &expected,
            cudf::size_type out_begin,
            cudf::size_type out_end,
            cudf::size_type in_begin)
  {
    gdf_column dest_cpy = cudf::copy(*dest.get());

    EXPECT_NO_THROW(cudf::copy_range(&dest_cpy, *source.get(), out_begin, out_end, in_begin));

    EXPECT_EQ(expected, dest_cpy);

    if (!(expected == dest_cpy)) {
      std::cout << "expected\n";
      expected.print();
      std::cout << expected.get()->null_count << "\n";
      std::cout << "dest\n";
      print_gdf_column(&dest_cpy);
      std::cout << dest_cpy.null_count << "\n";
      std::cout << "source\n";
      source.print();
      std::cout << source.get()->null_count << "\n";
    }
  }
};

using test_types = ::testing::
  Types<int8_t, int16_t, int32_t, int64_t, float, double, cudf::bool8, cudf::nvstring_category>;
TYPED_TEST_CASE(CopyRangeTest, test_types);

struct row_value {
  int scale;
  cudf::size_type operator()(cudf::size_type row) { return row * scale; }
};

auto valid   = [](cudf::size_type row) { return true; };
auto depends = [](cudf::size_type row) { return (row % 2 == 0); };

TYPED_TEST(CopyRangeTest, CopyWithNulls)
{
  using T = TypeParam;

  cudf::size_type size      = this->column_size;
  cudf::size_type out_begin = 30;
  cudf::size_type out_end   = size - 20;
  cudf::size_type in_begin  = 9;
  cudf::size_type row_diff  = in_begin - out_begin;

  cudf::test::column_wrapper_factory<T> factory;

  column_wrapper<T> dest = factory.make(size, row_value{1}, valid);

  this->test(dest,
             factory.make(size, row_value{2}, depends),
             factory.make(
               size,
               [&](cudf::size_type row) {
                 return ((row >= out_begin) && (row < out_end)) ? row_value{2}(row + row_diff)
                                                                : row_value{1}(row);
               },
               [&](cudf::size_type row) {
                 return ((row >= out_begin) && (row < out_end)) ? depends(row + row_diff) : true;
               }),
             out_begin,
             out_end,
             in_begin);
}

TYPED_TEST(CopyRangeTest, CopyNoNulls)
{
  using T = TypeParam;

  cudf::size_type size      = this->column_size;
  cudf::size_type out_begin = 30;
  cudf::size_type out_end   = size - 20;
  cudf::size_type in_begin  = 9;
  cudf::size_type row_diff  = in_begin - out_begin;

  cudf::test::column_wrapper_factory<T> factory;

  column_wrapper<T> dest = factory.make(size, row_value{1});

  // First set it as valid
  this->test(dest,
             factory.make(size, row_value{2}),
             factory.make(size,
                          [&](cudf::size_type row) {
                            return ((row >= out_begin) && (row < out_end))
                                     ? row_value{2}(row + row_diff)
                                     : row_value{1}(row);
                          }),
             out_begin,
             out_end,
             in_begin);
}

struct CopyRangeErrorTest : GdfTest {
};

TEST_F(CopyRangeErrorTest, InvalidColumn)
{
  column_wrapper<int32_t> source(100, row_value{1});
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
  column_wrapper<int32_t> dest(100, row_value{1});
  column_wrapper<int32_t> source(100, row_value{2});
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
  column_wrapper<int32_t> dest(100, row_value{1});
  column_wrapper<float> source(100, row_value{2});
  CUDF_EXPECT_THROW_MESSAGE(cudf::copy_range(dest.get(), *source.get(), 0, 10, 0),
                            "Data type mismatch");
}
