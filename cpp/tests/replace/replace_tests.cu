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

#include <cstdlib>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include <cudf.h>
#include <cudf/functions.h>

#include <thrust/device_vector.h>

#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/utilities/cudf_test_utils.cuh"

// This is the main test feature
template <class T>
struct ReplaceTest : public GdfTest
{
  std::vector<T> replace_column;
  std::vector<T> old_values_column;
  std::vector<T> new_values_column;

  gdf_col_pointer gdf_replace_column;
  gdf_col_pointer gdf_old_values_column;
  gdf_col_pointer gdf_new_values_column;

  gdf_column* gdf_raw_replace_column;
  gdf_column* gdf_raw_old_values_column;
  gdf_column* gdf_raw_new_values_column;

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

  /* --------------------------------------------------------------------------*
   * @brief Initializes the input columns with the given values.
   *
   * @Param replace_column_list The original values
   * @Param old_values_column_list The values that will be replaced
   * @Param new_values_column_list The new values
   * @Param print Optionally print the set of columns for debug
   * -------------------------------------------------------------------------*/
  void create_input(const std::initializer_list<T> &replace_column_list,
                    const std::initializer_list<T> &old_values_column_list,
                    const std::initializer_list<T> &new_values_column_list,
                    bool print = false)
  {
    replace_column    = replace_column_list;
    old_values_column = old_values_column_list;
    new_values_column = new_values_column_list;

    gdf_replace_column    = create_gdf_column(replace_column);
    gdf_old_values_column = create_gdf_column(old_values_column);
    gdf_new_values_column = create_gdf_column(new_values_column);

    gdf_raw_replace_column = gdf_replace_column.get();
    gdf_raw_old_values_column = gdf_old_values_column.get();
    gdf_raw_new_values_column = gdf_new_values_column.get();

    if(print)
    {
      std::cout << "replace column(s) created. Size: " << replace_column.size() << std::endl;
      print_vector(replace_column);
      std::cout << "\n";
    }
  }

  /* --------------------------------------------------------------------------*/
  /**
   * @brief Computes a reference solution
   *
   * @Param print Option to print the solution for debug
   *
   * @Returns A vector of 'T' with the old values replaced  
   */
  /* ----------------------------------------------------------------------------*/
  std::vector<T> compute_reference_solution(bool print = false)
  {
    std::vector<T> reference_result(replace_column);
    std::vector<bool> isReplaced(reference_result.size(), false);

    for(size_t i = 0; i < old_values_column.size(); i++)
    {
      size_t k = 0;
      auto pred = [&, this] (T element) {
        bool toBeReplaced = false;
        if(!isReplaced[k])
        {
          toBeReplaced = (element == this->old_values_column[i]);
          isReplaced[k] = toBeReplaced;
        }    

        ++k;
        return toBeReplaced;
      };
      std::replace_if(reference_result.begin(), reference_result.end(), pred, new_values_column[i]);
    }

    if(print)
    {
      std::cout << "Reference result size: " << reference_result.size() << std::endl;
      print_vector(reference_result);
      std::cout << "\n";
    }

    return reference_result;
  }

  /* --------------------------------------------------------------------------*/
  /**
   * @brief Replaces the values in a column given a map of old values to be replaced
   * and new values with the libgdf functions
   *
   * @Param print Option to print the result computed by the libgdf function
   * 
   * @Returns A vector of 'T' with the old values replaced  
   */
  /* ----------------------------------------------------------------------------*/
  std::vector<T> compute_gdf_result(bool print = false, gdf_error expected_result = GDF_SUCCESS)
  {
    gdf_error result_error{GDF_SUCCESS};

    gdf_error status = gdf_find_and_replace_all(gdf_raw_replace_column, gdf_raw_old_values_column, gdf_raw_new_values_column);

    EXPECT_EQ(expected_result, result_error) << "The gdf order by function did not complete successfully";

    // If the expected result was not GDF_SUCCESS, then this test was testing for a
    // specific error condition, in which case we return imediately and do not do
    // any further work on the output
    if(GDF_SUCCESS != expected_result){
      return std::vector<T>();
    }

    size_t output_size = gdf_raw_replace_column->size;
    std::vector<T> host_result(output_size);

    EXPECT_EQ(cudaMemcpy(host_result.data(),
               gdf_raw_replace_column->data, output_size * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);

    if(print){
      std::cout << "GDF result size: " << host_result.size() << std::endl;
      print_vector(host_result);
      std::cout << "\n";
    }

    return host_result;
  }
};

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
  this->create_input({7, 5, 6, 3, 1, 2, 8, 4}, {2, 6, 4, 8}, {0, 4, 2, 6}, true);

  auto reference_result = this->compute_reference_solution(true);
  auto gdf_result = this->compute_gdf_result(true);
  
  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";
   // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}


// Simple test, replacing all even gdf_new_values_column
TYPED_TEST(ReplaceTest, ReplaceEvenPosition)
{
  this->create_input({1, 2, 3, 4, 5, 6, 7, 8}, {2, 4, 6, 8}, {0, 2, 4, 6});
  
  auto reference_result = this->compute_reference_solution();
  auto gdf_result = this->compute_gdf_result();
  
  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";
   // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

// Similar test as ReplaceEvenPosition, but with unordered data
TYPED_TEST(ReplaceTest, Unordered)
{
  this->create_input({7, 5, 6, 3, 1, 2, 8, 4}, {2, 6, 4, 8}, {0, 4, 2, 6});
  
  auto reference_result = this->compute_reference_solution();
  auto gdf_result = this->compute_gdf_result();
  
  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";
   // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

// Testing with Empty Replace
TYPED_TEST(ReplaceTest, EmptyReplace)
{
  this->create_input({7, 5, 6, 3, 1, 2, 8, 4}, {}, {});
  
  auto reference_result = this->compute_reference_solution();
  auto gdf_result = this->compute_gdf_result();
  
  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";
   // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

// Testing with Nothing To Replace
TYPED_TEST(ReplaceTest, NothingToReplace)
{
  this->create_input({7, 5, 6, 3, 1, 2, 8, 4}, {10, 11, 12}, {15, 16, 17});
  
  auto reference_result = this->compute_reference_solution();
  auto gdf_result = this->compute_gdf_result();
  
  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";
   // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

// Testing with Empty Data
TYPED_TEST(ReplaceTest, EmptyData)
{
  this->create_input({}, {10, 11, 12}, {15, 16, 17});
  
  auto reference_result = this->compute_reference_solution();
  auto gdf_result = this->compute_gdf_result();
  
  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";
   // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

// Test with much larger data sets
TYPED_TEST(ReplaceTest, LargeScaleReplaceTest)
{
  const size_t DATA_SIZE    = 1000000;
  const size_t REPLACE_SIZE = 10000;

  this->replace_column.resize(DATA_SIZE);
  for (size_t i = 0; i < DATA_SIZE; i++) {
      this->replace_column[i] = std::rand() % (2 * REPLACE_SIZE);
  }

  this->old_values_column.resize(REPLACE_SIZE);
  this->new_values_column.resize(REPLACE_SIZE);
  size_t count = 0;
  for (size_t i = 0; i < 7; i++) {
    for (size_t j = 0; j < REPLACE_SIZE; j += 7) {
      if (i + j < REPLACE_SIZE) {
        this->old_values_column[i + j] = count;
        count++;
        this->new_values_column[i + j] = count;
      }
    }
  }

  this->gdf_replace_column    = create_gdf_column(this->replace_column);
  this->gdf_old_values_column = create_gdf_column(this->old_values_column);
  this->gdf_new_values_column = create_gdf_column(this->new_values_column);

  this->gdf_raw_replace_column = this->gdf_replace_column.get();
  this->gdf_raw_old_values_column = this->gdf_old_values_column.get();
  this->gdf_raw_new_values_column = this->gdf_new_values_column.get();

  auto gdf_result = this->compute_gdf_result();

  for (size_t i = 0; i < DATA_SIZE; i++) {
    if ((size_t)(this->replace_column[i]) < REPLACE_SIZE) {
      EXPECT_EQ((TypeParam)(this->replace_column[i] + 1), gdf_result[i]);
    }
  }
}
