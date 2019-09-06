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

#include <cudf/filling.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/column_wrapper_factory.hpp>
#include <tests/utilities/scalar_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

#include <algorithm>

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

template <typename T>
using scalar_wrapper = cudf::test::scalar_wrapper<T>;

template <typename T>
struct FillingTest : GdfTest {};

using test_types =
  ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double,
                   cudf::bool8>;
TYPED_TEST_CASE(FillingTest, test_types);

constexpr gdf_size_type column_size{1000};

auto all_valid = [](gdf_index_type row) { return true; };

template <typename T, typename BitInitializerType = decltype(all_valid)>
void FillTest(gdf_index_type begin, gdf_index_type end,
              T value, bool value_is_valid = true, 
              BitInitializerType source_validity = all_valid)
{
  column_wrapper<T> source(column_size, 
    [](gdf_index_type row) { return static_cast<T>(row); },
    [&](gdf_index_type row) { return source_validity(row); });

  scalar_wrapper<T> val(value, value_is_valid);

  column_wrapper<T> expected(column_size,
    [&](gdf_index_type row) { 
      return (row >= begin && row < end) ? 
        value : static_cast<T>(row);
    },
    [&](gdf_index_type row) { 
      return (row >= begin && row < end) ? 
        value_is_valid : source_validity(row); 
    });

  EXPECT_NO_THROW(cudf::fill(source.get(), *val.get(), begin, end));

  EXPECT_TRUE(expected == source);

  if (!(expected == source)) {
    std::cout << "expected\n";
    expected.print();
    std::cout << expected.get()->null_count << "\n";
    std::cout << "source\n";
    source.print();
    std::cout << source.get()->null_count << "\n";
  }
}

TYPED_TEST(FillingTest, SetSingle)
{
  gdf_index_type index = 9;
  TypeParam val = TypeParam{1};
  
  // First set it as valid
  FillTest(index, index+1, val, true);
  // next set it as invalid
  FillTest(index, index+1, val, false);
}

TYPED_TEST(FillingTest, SetAll)
{
  TypeParam val = TypeParam{1};

  // First set it as valid
  FillTest(0, column_size, val, true);
  // next set it as invalid
  FillTest(0, column_size, val, false);
}

TYPED_TEST(FillingTest, SetRange)
{
  gdf_index_type begin = 99;
  gdf_index_type end   = 299;
  TypeParam val = TypeParam{1};

  // First set it as valid
  FillTest(begin, end, val, true);
  // Next set it as invalid
  FillTest(begin, end, val, false);
}

TYPED_TEST(FillingTest, SetRangeNullCount)
{
  gdf_index_type begin = 10;
  gdf_index_type end = 50;
  TypeParam val = TypeParam{1};

  auto some_valid = [](gdf_index_type row) { 
    return row % 2 != 0;
  };

  auto all_invalid = [](gdf_index_type row) { 
    return false;
  };

  // First set it as valid value
  FillTest(begin, end, val, true, some_valid);

  // Next set it as invalid
  FillTest(begin, end, val, false, some_valid);

  // All invalid column should have some valid
  FillTest(begin, end, val, true, all_invalid);

  // All should be invalid
  FillTest(begin, end, val, false, all_invalid);

  // All should be valid
  FillTest(0, column_size, val, true, some_valid);
}

struct FillingErrorTest : GdfTest {};

TEST_F(FillingErrorTest, InvalidColumn)
{
  scalar_wrapper<int32_t> val(5, true);
  CUDF_EXPECT_THROW_MESSAGE(cudf::fill(nullptr, *val.get(), 0, 10),
                            "Null gdf_column pointer");

  gdf_column bad_input;
  gdf_column_view(&bad_input, 0, 0, 0, GDF_INT32);
  // empty range == no-op, even on invalid output column...
  EXPECT_NO_THROW(cudf::fill(&bad_input, *val.get(), 0, 0));

  // for zero-size column, non-empty range is out of bounds
  CUDF_EXPECT_THROW_MESSAGE(cudf::fill(&bad_input, *val.get(), 0, 10),
                            "Range is out of bounds");

  // invalid data pointer
  bad_input.size = 20;
  CUDF_EXPECT_THROW_MESSAGE(cudf::fill(&bad_input, *val.get(), 0, 10),
                            "Null column data with non-zero size");
}

TEST_F(FillingErrorTest, InvalidRange)
{
  scalar_wrapper<int32_t> val(5, true);
  column_wrapper<int32_t> dest(100, 
    [](gdf_index_type row) { return static_cast<int32_t>(row); },
    [](gdf_index_type row) { return true; });
  
  CUDF_EXPECT_THROW_MESSAGE(cudf::fill(dest.get(), *val.get(), 0, 110),
                            "Range is out of bounds");
  CUDF_EXPECT_THROW_MESSAGE(cudf::fill(dest.get(), *val.get(), -10, 0),
                            "Range is out of bounds");
  CUDF_EXPECT_THROW_MESSAGE(cudf::fill(dest.get(), *val.get(), 10, 0),
                            "Range is empty or reversed");
}

TEST_F(FillingErrorTest, DTypeMismatch)
{
  scalar_wrapper<int32_t> val(5, true);
  column_wrapper<float> dest(100, 
    [](gdf_index_type row) { return static_cast<float>(row); },
    [](gdf_index_type row) { return true; });
  CUDF_EXPECT_THROW_MESSAGE(cudf::fill(dest.get(), *val.get(), 0, 10),
                            "Data type mismatch");
}

TEST_F(FillingErrorTest, StringCategoryNotSupported)
{
  scalar_wrapper<int32_t> val(5, true);
  std::vector<const char*> strings{"foo"};
  column_wrapper<cudf::nvstring_category> dest(1, strings.data());
  CUDF_EXPECT_THROW_MESSAGE(cudf::fill(dest.get(), *val.get(), 0, 1),
    "cudf::fill() does not support GDF_STRING_CATEGORY columns");
}

template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

template <typename T>
struct MultiFillingTest : GdfTest {};

using multi_fill_test_types =
  ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double,
                   cudf::bool8, cudf::nvstring_category>;
TYPED_TEST_CASE(MultiFillingTest, multi_fill_test_types);

TYPED_TEST(MultiFillingTest, SetRanges)
{
  using T = TypeParam;

  // Making the ranges that will be filled
  gdf_size_type num_ranges = 9;
  gdf_size_type max_range_size = column_size/(num_ranges+1);
  gdf_size_type begin_offset = random_int(0, max_range_size);

  std::vector<gdf_size_type> range_starts(num_ranges);
  std::iota(range_starts.begin(), range_starts.end(), 0);
  std::transform(range_starts.begin(), range_starts.end(), range_starts.begin(),
    [&](gdf_size_type i) { return i * max_range_size + begin_offset; });

  std::vector<gdf_size_type> range_sizes(num_ranges);
  std::transform(range_sizes.begin(), range_sizes.end(), range_sizes.begin(),
    [&](gdf_size_type i) { return random_int(0, max_range_size); });

  std::vector<gdf_size_type> range_ends(num_ranges);
  std::iota(range_ends.begin(), range_ends.end(), 0);
  std::transform(range_ends.begin(), range_ends.end(), range_ends.begin(),
    [&](gdf_size_type i) { return range_starts[i] + range_sizes[i]; });

  std::vector<int> values_data(num_ranges);
  std::iota(values_data.begin(), values_data.end(), 0);
  std::transform(values_data.begin(), values_data.end(), values_data.begin(),
    [](gdf_size_type i) { return i * 2; });

  // set your expectations
  std::vector<int> expect_vals(column_size);
  std::iota(expect_vals.begin(), expect_vals.end(), 0);
  for (gdf_size_type i = 0; i < num_ranges; ++i) {
    auto expect_it = expect_vals.begin();
    std::advance(expect_it, range_starts[i]);
    std::fill_n(expect_it, range_sizes[i], values_data[i]);
  }

  cudf::test::column_wrapper_factory<T> factory;

  column_wrapper<T> source = factory.make(column_size,
    [](gdf_index_type row) { return row; });

  column_wrapper<T> values = factory.make(num_ranges,
    [&](gdf_size_type i) { return values_data[i]; });

  column_wrapper<T> expected = factory.make(column_size,
    [&](gdf_index_type row) { return expect_vals[row]; });

  cudf::fill(source.get(), *values.get(), range_starts, range_ends);

  EXPECT_TRUE(expected == source);

  if (!(expected == source)) {
    std::cout << "expected\n";
    expected.print();
    std::cout << expected.get()->null_count << "\n";
    std::cout << "source\n";
    source.print();
    std::cout << source.get()->null_count << "\n";
  }
}
