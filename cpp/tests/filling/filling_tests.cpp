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

#include <filling.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/scalar_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

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

template <typename T>
void FillTest(gdf_index_type begin, gdf_index_type end,
              T value, bool value_is_valid = true)
{
  column_wrapper<T> source(column_size, 
    [](gdf_index_type row) { return static_cast<T>(row); },
    [](gdf_index_type row) { return true; });

  scalar_wrapper<T> val(value, value_is_valid);

  column_wrapper<T> expected(column_size,
    [&](gdf_index_type row) { 
      return (row >= begin && row < end) ? 
        value : static_cast<T>(row);
    },
    [&](gdf_index_type row) { 
      return (row >= begin && row < end) ? 
        value_is_valid : true; 
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
