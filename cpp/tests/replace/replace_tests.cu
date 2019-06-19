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
#include <cudf/replace.hpp>

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/column_wrapper.cuh>

#include <cudf/cudf.h>
#include <bitmask/bit_mask.cuh>

#include <thrust/device_vector.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <vector>

struct ReplaceErrorTest : public GdfTest{};


// Error: old-values and new-values size mismatch
TEST_F(ReplaceErrorTest, SizeMismatch)
{

  cudf::test::column_wrapper<int32_t> gdf_input_column{ {7, 5, 6, 3, 1, 2, 8, 4}};
  cudf::test::column_wrapper<int32_t> gdf_values_to_replace_column{ {10, 11, 12, 13}};
  cudf::test::column_wrapper<int32_t> gdf_replacement_values_column{ {15, 16, 17}};

  CUDF_EXPECT_THROW_MESSAGE(cudf::find_and_replace_all(gdf_input_column,
                                                           gdf_values_to_replace_column,
                                                           gdf_replacement_values_column),
                            "values_to_replace and replacement_values size mismatch.");
}

// Error: column type mismatch
TEST_F(ReplaceErrorTest, TypeMismatch)
{

  cudf::test::column_wrapper<int32_t> gdf_input_column{ {7, 5, 6, 3, 1, 2, 8, 4}};
  cudf::test::column_wrapper<float> gdf_values_to_replace_column{ {10, 11, 12}};
  cudf::test::column_wrapper<int32_t> gdf_replacement_values_column{ {15, 16, 17}};

  CUDF_EXPECT_THROW_MESSAGE(cudf::find_and_replace_all(gdf_input_column,
                                                           gdf_values_to_replace_column,
                                                           gdf_replacement_values_column),
                            "Columns type mismatch.");
}

// Error: nulls in old-values
TEST_F(ReplaceErrorTest, NullInOldValues)
{
  std::vector<gdf_valid_type> old_valid(gdf_valid_allocation_size(4), 0xA);
  cudf::test::column_wrapper<int32_t> gdf_input_column{ {7, 5, 6, 3, 1, 2, 8, 4}};
  cudf::test::column_wrapper<int32_t> gdf_values_to_replace_column{ {10, 11, 12, 13}, old_valid};
  cudf::test::column_wrapper<int32_t> gdf_replacement_values_column{ {15, 16, 17, 18}};

  CUDF_EXPECT_THROW_MESSAGE(cudf::find_and_replace_all(gdf_input_column,
                                                           gdf_values_to_replace_column,
                                                           gdf_replacement_values_column),
                            "Nulls are in values_to_replace column.");
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
   * resultant column by invoking `cudf::find_and_replace_all()` and then
   * compute the expected column.
   *
   * @param input_column The original values
   * @param values_to_replace_column The values that will be replaced
   * @param replacement_values_column The new values
   * @param input_column_valid The mask for replace column
   * @param replacement_values_valid The mask for new values
   * @param print Optionally print the set of columns for debug
   * -------------------------------------------------------------------------*/
template <typename T>
void test_replace(std::vector<T> const &input_column,
                  std::vector<T> const &values_to_replace_column,
                  std::vector<T> const &replacement_values_column,
                  std::vector<gdf_valid_type> const& input_column_valid =
                     std::vector<gdf_valid_type>{},
                  std::vector<gdf_valid_type> const& replacement_values_valid =
                     std::vector<gdf_valid_type>{},
                  bool print = false) {

    cudf::test::column_wrapper<T> gdf_input_column{ input_column, input_column_valid};
    cudf::test::column_wrapper<T> gdf_values_to_replace_column{ values_to_replace_column};
    cudf::test::column_wrapper<T> gdf_replacement_values_column{replacement_values_column,
                                                                replacement_values_valid};

    if(print)
    {
      std::cout << "replace column: \n";
      gdf_input_column.print();
      std::cout << "values_to_replace column: \n";
      gdf_values_to_replace_column.print();
      std::cout << "replacement_values column: \n";
      gdf_replacement_values_column.print();
      std::cout << "\n";
    }
    /* getting the actual result*/
    gdf_column actual_result;
    EXPECT_NO_THROW( actual_result = cudf::find_and_replace_all(gdf_input_column,
                                                                gdf_values_to_replace_column,
                                                                gdf_replacement_values_column));
    if(print)
    {
      std::cout<<"printing result:\n";
      print_gdf_column(&actual_result);
    }
    /* computing the expected result */
    std::vector<T> reference_result(input_column);
    std::vector<bool> isReplaced(reference_result.size(), false);
    std::vector<gdf_valid_type> expected_valid(input_column_valid);

    if (replacement_values_valid.size() > 0 && 0==input_column_valid.size()){
        expected_valid.assign(gdf_valid_allocation_size(input_column.size()),
                                                   0xFF);
    }

    bit_mask::bit_mask_t *typed_expected_valid =
                    reinterpret_cast<bit_mask::bit_mask_t*>(expected_valid.data());
    const bit_mask::bit_mask_t *typed_new_valid =
                    reinterpret_cast<const bit_mask::bit_mask_t*>(replacement_values_valid.data());

    const bool input_has_nulls = (typed_expected_valid != nullptr);
    const bool replacement_has_nulls = (typed_new_valid != nullptr);

    for(size_t i = 0; i < values_to_replace_column.size(); i++)
    {
      size_t k = 0;
      auto pred = [=, &k, &typed_expected_valid, &isReplaced](T element) {
        bool toBeReplaced = false;
        if(!isReplaced[k])
        {
        if(!input_has_nulls || bit_mask::is_valid(typed_expected_valid, k)){
          if(element == values_to_replace_column[i]) {
          toBeReplaced = true;
          isReplaced[k] = toBeReplaced;
            if(replacement_has_nulls && !bit_mask::is_valid(typed_new_valid, i)){
              if(print)std::cout << "clearing bit at: "<<k<<"\n";
              bit_mask::clear_bit_unsafe(typed_expected_valid, (int)k);
            }
          }
         }
        }

        ++k;
        return toBeReplaced;
      };
      std::replace_if(reference_result.begin(), reference_result.end(),
                      pred, replacement_values_column[i]);
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

  std::vector<TypeParam> input_column{7, 5, 6, 3, 1, 2, 8, 4};
  std::vector<gdf_valid_type> input_column_valid(gdf_valid_allocation_size(input_column.size()),
                                                                             0xFE);
  std::vector<TypeParam> values_to_replace_column{2, 6, 4, 8};
  std::vector<TypeParam> replacement_values_column{0, 4, 2, 6};
  std::vector<gdf_valid_type> replacement_values_valid(
                                    gdf_valid_allocation_size(replacement_values_column.size()),
                                    0xA);

  test_replace<TypeParam>(input_column,
                          values_to_replace_column,
                          replacement_values_column,
                          input_column_valid,
                          replacement_values_valid,
                          true);
}


// Simple test, replacing all even gdf_replacement_values_column
TYPED_TEST(ReplaceTest, ReplaceEvenPosition)
{

  std::vector<TypeParam> input_column{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<TypeParam> values_to_replace_column{2, 6, 4, 8};
  std::vector<TypeParam> replacement_values_column{0, 4, 2, 6};

  test_replace<TypeParam>(input_column,
                          values_to_replace_column,
                          replacement_values_column);
}

// Similar test as ReplaceEvenPosition, but with unordered data
TYPED_TEST(ReplaceTest, Unordered)
{

  std::vector<TypeParam> input_column{7, 5, 6, 3, 1, 2, 8, 4};
  std::vector<TypeParam> values_to_replace_column{2, 6, 4, 8};
  std::vector<TypeParam> replacement_values_column{0, 4, 2, 6};

  test_replace<TypeParam>(input_column,
                          values_to_replace_column,
                          replacement_values_column);
}

// Testing with Nothing To Replace
TYPED_TEST(ReplaceTest, NothingToReplace)
{

  std::vector<TypeParam> input_column{7, 5, 6, 3, 1, 2, 8, 4};
  std::vector<TypeParam> values_to_replace_column{10, 11, 12};
  std::vector<TypeParam> replacement_values_column{15, 16, 17};

  test_replace<TypeParam>(input_column,
                          values_to_replace_column,
                          replacement_values_column);
}

// Testing with empty Data
TYPED_TEST(ReplaceTest, EmptyData)
{

  std::vector<TypeParam> input_column{ {}};
  std::vector<TypeParam> values_to_replace_column{10, 11, 12};
  std::vector<TypeParam> replacement_values_column{15, 16, 17};

  test_replace<TypeParam>(input_column,
                          values_to_replace_column,
                          replacement_values_column);
}

// Testing with empty Replace
TYPED_TEST(ReplaceTest, EmptyReplace)
{

  std::vector<TypeParam> input_column{7, 5, 6, 3, 1, 2, 8, 4};
  std::vector<TypeParam> values_to_replace_column{};
  std::vector<TypeParam> replacement_values_column{};

  test_replace<TypeParam>(input_column,
                          values_to_replace_column,
                          replacement_values_column);
}

// Testing with input column containing nulls
TYPED_TEST(ReplaceTest, NullsInData)
{
  std::vector<TypeParam> input_column{7, 5, 6, 3, 1, 2, 8, 4};
  std::vector<gdf_valid_type> input_column_valid(gdf_valid_allocation_size(input_column.size()),
                                                                             0xFE);
  std::vector<TypeParam> values_to_replace_column{2, 6, 4, 8};
  std::vector<TypeParam> replacement_values_column{0, 4, 2, 6};

  test_replace<TypeParam>(input_column,
                          values_to_replace_column,
                          replacement_values_column,
                          input_column_valid);
}

// Testing with replacement column containing nulls
TYPED_TEST(ReplaceTest, NullsInNewValues)
{
  std::vector<TypeParam> input_column{7, 5, 6, 3, 1, 2, 8, 4};
  std::vector<TypeParam> values_to_replace_column{2, 6, 4, 8};
  std::vector<TypeParam> replacement_values_column{0, 4, 2, 6};
  std::vector<gdf_valid_type> replacement_values_valid(
                                    gdf_valid_allocation_size(replacement_values_column.size()),
                                    0xA);

  test_replace<TypeParam>(input_column,
                          values_to_replace_column,
                          replacement_values_column,
                          {},
                          replacement_values_valid);
}


// Testing with both replacement and input column containing nulls
TYPED_TEST(ReplaceTest, NullsInBoth)
{
  std::vector<TypeParam> input_column{7, 5, 6, 3, 1, 2, 8, 4};
  std::vector<gdf_valid_type> input_column_valid(gdf_valid_allocation_size(input_column.size()),
                                                                             0xFE);
  std::vector<TypeParam> values_to_replace_column{2, 6, 4, 8};
  std::vector<TypeParam> replacement_values_column{0, 4, 2, 6};
  std::vector<gdf_valid_type> replacement_values_valid(
                                    gdf_valid_allocation_size(replacement_values_column.size()),
                                    0xA);

  test_replace<TypeParam>(input_column,
                          values_to_replace_column,
                          replacement_values_column,
                          input_column_valid,
                          replacement_values_valid);
}

// Test with much larger data sets
TYPED_TEST(ReplaceTest, LargeScaleReplaceTest)
{
  const size_t DATA_SIZE    = 1000000;
  const size_t REPLACE_SIZE = 10000;

  std::vector<TypeParam> input_column(DATA_SIZE);
  for (size_t i = 0; i < DATA_SIZE; i++) {
      input_column[i] = std::rand() % (REPLACE_SIZE);
  }

  std::vector<TypeParam> values_to_replace_column(REPLACE_SIZE);
  std::vector<TypeParam> replacement_values_column(REPLACE_SIZE);
  size_t count = 0;
  for (size_t i = 0; i < 7; i++) {
    for (size_t j = 0; j < REPLACE_SIZE; j += 7) {
      if (i + j < REPLACE_SIZE) {
        values_to_replace_column[i + j] = count;
        count++;
        replacement_values_column[i + j] = count;
      }
    }
  }
  cudf::test::column_wrapper<TypeParam> gdf_input_column{ input_column};
  cudf::test::column_wrapper<TypeParam> gdf_values_to_replace_column{ values_to_replace_column};
  cudf::test::column_wrapper<TypeParam> gdf_replacement_values_column{replacement_values_column};

  gdf_column actual_result;
  EXPECT_NO_THROW( actual_result = cudf::find_and_replace_all(gdf_input_column,
                                                                   gdf_values_to_replace_column,
                                                                   gdf_replacement_values_column));

  std::for_each(input_column.begin(), input_column.end(), [](TypeParam& d) { d+=1;});
  cudf::test::column_wrapper<TypeParam> expected{input_column };
  EXPECT_TRUE(expected == actual_result);
}
