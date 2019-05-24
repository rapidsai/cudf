/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
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
#include <replace.hpp>

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/column_wrapper.cuh>

#include <cudf.h>
#include <bitmask/bit_mask.cuh>

#include <thrust/device_vector.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <vector>

struct ReplaceErrorTest : public GdfTest{};

// Error: Testing with Empty Replace
TEST_F(ReplaceErrorTest, EmptyReplace)
{

  cudf::test::column_wrapper<int32_t> gdf_replace_column{ {7, 5, 6, 3, 1, 2, 8, 4}};
  cudf::test::column_wrapper<int32_t> gdf_old_values_column{ {}};
  cudf::test::column_wrapper<int32_t> gdf_new_values_column{ {}};

  CUDF_EXPECT_THROW_MESSAGE(cudf::gdf_find_and_replace_all(gdf_replace_column,
                                                           gdf_old_values_column,
                                                           gdf_new_values_column),
                            "Null replace data.");

}

// Error: Testing with Empty Data
TEST_F(ReplaceErrorTest, EmptyData)
{

  cudf::test::column_wrapper<int32_t> gdf_replace_column{ {}};
  cudf::test::column_wrapper<int32_t> gdf_old_values_column{ {10, 11, 12}};
  cudf::test::column_wrapper<int32_t> gdf_new_values_column{ {15, 16, 17}};

  CUDF_EXPECT_THROW_MESSAGE(cudf::gdf_find_and_replace_all(gdf_replace_column,
                                                           gdf_old_values_column,
                                                           gdf_new_values_column),
                            "Null input data.");
}

// Error: old-values and new-values size mismatch
TEST_F(ReplaceErrorTest, SizeMismatch)
{

  cudf::test::column_wrapper<int32_t> gdf_replace_column{ {7, 5, 6, 3, 1, 2, 8, 4}};
  cudf::test::column_wrapper<int32_t> gdf_old_values_column{ {10, 11, 12, 13}};
  cudf::test::column_wrapper<int32_t> gdf_new_values_column{ {15, 16, 17}};

  CUDF_EXPECT_THROW_MESSAGE(cudf::gdf_find_and_replace_all(gdf_replace_column,
                                                           gdf_old_values_column,
                                                           gdf_new_values_column),
                            "old_values and new_values size mismatch.");
}

// Error: column type mismatch
TEST_F(ReplaceErrorTest, TypeMismatch)
{

  cudf::test::column_wrapper<int32_t> gdf_replace_column{ {7, 5, 6, 3, 1, 2, 8, 4}};
  cudf::test::column_wrapper<float> gdf_old_values_column{ {10, 11, 12}};
  cudf::test::column_wrapper<int32_t> gdf_new_values_column{ {15, 16, 17}};

  CUDF_EXPECT_THROW_MESSAGE(cudf::gdf_find_and_replace_all(gdf_replace_column,
                                                           gdf_old_values_column,
                                                           gdf_new_values_column),
                            "Columns type mismatch.");
}

// Error: nulls in old-values
TEST_F(ReplaceErrorTest, NullInOldValues)
{
  std::vector<gdf_valid_type> old_valid(gdf_valid_allocation_size(4), 0xA);
  cudf::test::column_wrapper<int32_t> gdf_replace_column{ {7, 5, 6, 3, 1, 2, 8, 4}};
  cudf::test::column_wrapper<int32_t> gdf_old_values_column{ {10, 11, 12, 13}, old_valid};
  cudf::test::column_wrapper<int32_t> gdf_new_values_column{ {15, 16, 17, 18}};

  CUDF_EXPECT_THROW_MESSAGE(cudf::gdf_find_and_replace_all(gdf_replace_column,
                                                           gdf_old_values_column,
                                                           gdf_new_values_column),
                            "Nulls can not be replaced.");
}


// This is the main test feature
template <class T>
struct ReplaceTest : public GdfTest
{

  ReplaceTest()
  {
    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    static size_t number_of_instantiations{0};
    std::srand(number_of_instantiations++);
  }

  ~ReplaceTest()
  {
  }
};

/* --------------------------------------------------------------------------*
   * @brief Main method for testing.
   * Initializes the input columns with the given values. Then compute the actual
   * resultant column by invoking `cudf::gdf_find_and_replace_all()` and then
   * compute the expected column.
   *
   * @param replace_column The original values
   * @param old_values_column The values that will be replaced
   * @param new_values_column The new values
   * @param replace_column_valid The mask for replace column
   * @param new_column_valid The mask for new values
   * @param print Optionally print the set of columns for debug
   * -------------------------------------------------------------------------*/
template <typename T>
void test_replace(std::vector<T> const &replace_column,
                  std::vector<T> const &old_values_column,
                  std::vector<T> const &new_values_column,
                  std::vector<gdf_valid_type> const& replace_column_valid =
                     std::vector<gdf_valid_type>{},
                  std::vector<gdf_valid_type> const& new_column_valid =
                     std::vector<gdf_valid_type>{},
                  bool print = false) {

    cudf::test::column_wrapper<T> gdf_replace_column{ replace_column, replace_column_valid};
    cudf::test::column_wrapper<T> gdf_old_values_column{ old_values_column};
    cudf::test::column_wrapper<T> gdf_new_values_column{new_values_column, new_column_valid};

    if(print)
    {
      std::cout << "replace column: \n";
      gdf_replace_column.print();
      std::cout << "old_values column: \n";
      gdf_old_values_column.print();
      std::cout << "new_values column: \n";
      gdf_new_values_column.print();
      std::cout << "\n";
    }
    /* getting the actual result*/
    gdf_column actual_result;
    EXPECT_NO_THROW( actual_result = cudf::gdf_find_and_replace_all(gdf_replace_column,
                                                                   gdf_old_values_column,
                                                                   gdf_new_values_column));
    if(print)
    {
      std::cout<<"printing result:\n";
      print_gdf_column(&actual_result);
    }
    /* computing the expected result */
    std::vector<T> reference_result(replace_column);
    std::vector<bool> isReplaced(reference_result.size(), false);
    std::vector<gdf_valid_type> expected_valid(replace_column_valid);

    if (new_column_valid.size() > 0 && 0==replace_column_valid.size()){
        expected_valid.assign(gdf_valid_allocation_size(replace_column.size()),
                                                   0xFF);
    }

    bit_mask::bit_mask_t *typed_expected_valid = reinterpret_cast<bit_mask::bit_mask_t*>(expected_valid.data());
    const bit_mask::bit_mask_t *typed_new_valid = reinterpret_cast<const bit_mask::bit_mask_t*>(new_column_valid.data());

    const bool col_is_nullable = (typed_expected_valid != nullptr);
    const bool new_is_nullable = (typed_new_valid != nullptr);

    for(size_t i = 0; i < old_values_column.size(); i++)
    {
      size_t k = 0;
      auto pred = [& ](T element) {
        bool toBeReplaced = false;
        if(!isReplaced[k])
        {
        if((!col_is_nullable || bit_mask::is_valid(typed_expected_valid, k)) && (element == old_values_column[i])) {
          toBeReplaced = true;
          isReplaced[k] = toBeReplaced;
          if(new_is_nullable && !bit_mask::is_valid(typed_new_valid, i)){
              if(print)std::cout << "clearing bit at: "<<k<<"\n";
              bit_mask::clear_bit_unsafe(typed_expected_valid, (int)k);
            }
          }
        }

        ++k;
        return toBeReplaced;
      };
      std::replace_if(reference_result.begin(), reference_result.end(), pred, new_values_column[i]);
    }

    cudf::test::column_wrapper<T> expected{reference_result, expected_valid};

    if(print)
    {
      std::cout << "Expected result: \n";
      expected.print();
      std::cout << "\n";
    }

    EXPECT_TRUE(expected == actual_result);
    gdf_column_free(&actual_result);
}


using Types = testing::Types<int8_t,
                             int16_t,
                             int, 
                             int64_t,
                             float,
                             double>;

TYPED_TEST_CASE(ReplaceTest, Types);

// This test is used for debugging purposes and is disabled by default.
// The input sizes are small and has a large amount of debug printing enabled.
TYPED_TEST(ReplaceTest, DISABLED_DebugTest)
{

  std::vector<TypeParam> replace_column{7, 5, 6, 3, 1, 2, 8, 4};
  std::vector<gdf_valid_type> replace_column_valid(gdf_valid_allocation_size(replace_column.size()),
                                                                             0xFE);
  std::vector<TypeParam> old_values_column{2, 6, 4, 8};
  std::vector<TypeParam> new_values_column{0, 4, 2, 6};
  std::vector<gdf_valid_type> new_column_valid(gdf_valid_allocation_size(new_values_column.size()),
                                                                             0xA);

  test_replace<TypeParam>(replace_column,
                          old_values_column,
                          new_values_column,
                          replace_column_valid,
                          new_column_valid,
                          true);
}


// Simple test, replacing all even gdf_new_values_column
TYPED_TEST(ReplaceTest, ReplaceEvenPosition)
{

  std::vector<TypeParam> replace_column{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<TypeParam> old_values_column{2, 6, 4, 8};
  std::vector<TypeParam> new_values_column{0, 4, 2, 6};

  test_replace<TypeParam>(replace_column,
                          old_values_column,
                          new_values_column);
}

// Similar test as ReplaceEvenPosition, but with unordered data
TYPED_TEST(ReplaceTest, Unordered)
{

  std::vector<TypeParam> replace_column{7, 5, 6, 3, 1, 2, 8, 4};
  std::vector<TypeParam> old_values_column{2, 6, 4, 8};
  std::vector<TypeParam> new_values_column{0, 4, 2, 6};

  test_replace<TypeParam>(replace_column,
                          old_values_column,
                          new_values_column);
}

// Testing with Nothing To Replace
TYPED_TEST(ReplaceTest, NothingToReplace)
{

  std::vector<TypeParam> replace_column{7, 5, 6, 3, 1, 2, 8, 4};
  std::vector<TypeParam> old_values_column{10, 11, 12};
  std::vector<TypeParam> new_values_column{15, 16, 17};

  test_replace<TypeParam>(replace_column,
                          old_values_column,
                          new_values_column);
}


// Testing with input column containing nulls
TYPED_TEST(ReplaceTest, NullsInData)
{
  std::vector<TypeParam> replace_column{7, 5, 6, 3, 1, 2, 8, 4};
  std::vector<gdf_valid_type> replace_column_valid(gdf_valid_allocation_size(replace_column.size()),
                                                                             0xFE);
  std::vector<TypeParam> old_values_column{2, 6, 4, 8};
  std::vector<TypeParam> new_values_column{0, 4, 2, 6};

  test_replace<TypeParam>(replace_column,
                          old_values_column,
                          new_values_column,
                          replace_column_valid);
}

// Testing with replacement column containing nulls
TYPED_TEST(ReplaceTest, NullsInNewValues)
{
  std::vector<TypeParam> replace_column{7, 5, 6, 3, 1, 2, 8, 4};
  std::vector<TypeParam> old_values_column{2, 6, 4, 8};
  std::vector<TypeParam> new_values_column{0, 4, 2, 6};
  std::vector<gdf_valid_type> new_column_valid(gdf_valid_allocation_size(new_values_column.size()),
                                                                             0xA);

  test_replace<TypeParam>(replace_column,
                          old_values_column,
                          new_values_column,
                          {},
                          new_column_valid);
}


// Testing with both replacement and input column containing nulls
TYPED_TEST(ReplaceTest, NullsInBoth)
{
  std::vector<TypeParam> replace_column{7, 5, 6, 3, 1, 2, 8, 4};
  std::vector<gdf_valid_type> replace_column_valid(gdf_valid_allocation_size(replace_column.size()),
                                                                             0xFE);
  std::vector<TypeParam> old_values_column{2, 6, 4, 8};
  std::vector<TypeParam> new_values_column{0, 4, 2, 6};
  std::vector<gdf_valid_type> new_column_valid(gdf_valid_allocation_size(new_values_column.size()),
                                                                             0xA);

  test_replace<TypeParam>(replace_column,
                          old_values_column,
                          new_values_column,
                          replace_column_valid,
                          new_column_valid);
}

// Test with much larger data sets
TYPED_TEST(ReplaceTest, LargeScaleReplaceTest)
{
  const size_t DATA_SIZE    = 1000000;
  const size_t REPLACE_SIZE = 10000;

  std::vector<TypeParam> replace_column(DATA_SIZE);
  for (size_t i = 0; i < DATA_SIZE; i++) {
      replace_column[i] = std::rand() % (REPLACE_SIZE);
  }

  std::vector<TypeParam> old_values_column(REPLACE_SIZE);
  std::vector<TypeParam> new_values_column(REPLACE_SIZE);
  size_t count = 0;
  for (size_t i = 0; i < 7; i++) {
    for (size_t j = 0; j < REPLACE_SIZE; j += 7) {
      if (i + j < REPLACE_SIZE) {
        old_values_column[i + j] = count;
        count++;
        new_values_column[i + j] = count;
      }
    }
  }
  cudf::test::column_wrapper<TypeParam> gdf_replace_column{ replace_column};
  cudf::test::column_wrapper<TypeParam> gdf_old_values_column{ old_values_column};
  cudf::test::column_wrapper<TypeParam> gdf_new_values_column{new_values_column};

  gdf_column actual_result;
  EXPECT_NO_THROW( actual_result = cudf::gdf_find_and_replace_all(gdf_replace_column,
                                                                   gdf_old_values_column,
                                                                   gdf_new_values_column));

  std::for_each(replace_column.begin(), replace_column.end(), [](TypeParam& d) { d+=1;});
  cudf::test::column_wrapper<TypeParam> expected{replace_column };
  EXPECT_TRUE(expected == actual_result);
}