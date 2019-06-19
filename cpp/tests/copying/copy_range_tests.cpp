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
#include <tests/utilities/nvcategory_utils.cuh>

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

template <typename T>
struct CopyRangeTest : GdfTest 
{
  static constexpr gdf_size_type column_size{1000};

  void test(column_wrapper<T> const &dest,
            column_wrapper<T> const &source,
            column_wrapper<T> const &expected,
            gdf_index_type out_begin, gdf_index_type out_end,
            gdf_index_type in_begin)
  {
    gdf_column dest_cpy = cudf::copy(*dest.get());

    EXPECT_NO_THROW(cudf::copy_range(&dest_cpy, *source.get(),
                    out_begin, out_end, in_begin));

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

using test_types =
  ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double,
                   cudf::bool8, cudf::nvstring_category>;
TYPED_TEST_CASE(CopyRangeTest, test_types);

template <typename T>
T convert(gdf_index_type row) {
  return static_cast<T>(row);
}

template <>
const char* convert<const char*>(gdf_index_type row) {
  std::ostringstream convert;
  convert << row;
  char *s = new char[convert.str().size()+1];
  std::strcpy(s, convert.str().c_str());
  return s; 
}

template <typename T>
struct make_input
{
  template<typename DataInitializer>
  column_wrapper<T>
  operator()(gdf_size_type size,
             DataInitializer data_init) {
    return column_wrapper<T>(size,
      [&](gdf_index_type row) { return convert<T>(data_init(row)); });
  }

  template<typename DataInitializer, typename BitInitializer>
  column_wrapper<T>
  operator()(gdf_size_type size,
             DataInitializer data_init,
             BitInitializer bit_init) {
    return column_wrapper<T>(size,
      [&](gdf_index_type row) { return convert<T>(data_init(row)); },
      bit_init);
  }
};

template <>
struct make_input<cudf::nvstring_category>
{
  int scale;
  template<typename DataInitializer>
  column_wrapper<cudf::nvstring_category>
  operator()(gdf_size_type size,
             DataInitializer data_init) {
    std::vector<const char*> strings(size);
    std::generate(strings.begin(), strings.end(), [data_init, row=0]() mutable {
      return convert<const char*>(data_init(row++));
    });
    
    auto c =  column_wrapper<cudf::nvstring_category>{size, strings.data()};
    
    std::for_each(strings.begin(), strings.end(), [](const char* x) { 
      delete [] x; 
    });

    return c;
  }

  template<typename DataInitializer, typename BitInitializer>
  column_wrapper<cudf::nvstring_category>
  operator()(gdf_size_type size,
             DataInitializer data_init,
             BitInitializer bit_init) {
    std::vector<const char*> strings(size);
    std::generate(strings.begin(), strings.end(), [data_init, row=0]() mutable {
      return convert<const char*>(data_init(row++));
    });

    auto c =  column_wrapper<cudf::nvstring_category>{size,
                                                      strings.data(),
                                                      bit_init};
    
    std::for_each(strings.begin(), strings.end(), [](const char* x) { 
      delete [] x; 
    });

    return c;
  }
};

struct row_value {
  int scale;
  gdf_index_type operator()(gdf_index_type row) { return row * scale; }
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

  make_input<T> maker{};

  column_wrapper<T> dest = maker(size, row_value{1}, valid);
  
  this->test(
    dest,
    maker(size, row_value{2}, depends),
    maker(size,
      [&](gdf_index_type row) {
        return ((row >= out_begin) && (row < out_end)) ?
          row_value{2}(row + row_diff) : row_value{1}(row);
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

  make_input<T> maker{};

  column_wrapper<T> dest = maker(size, row_value{1});

  // First set it as valid
  this->test(
    dest,
    maker(size, row_value{2}),
    maker(size,
      [&](gdf_index_type row) {
        return ((row >= out_begin) && (row < out_end)) ?
          row_value{2}(row + row_diff) : row_value{1}(row);
      }),
    out_begin, out_end, in_begin);
}

struct CopyRangeErrorTest : GdfTest {};

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
