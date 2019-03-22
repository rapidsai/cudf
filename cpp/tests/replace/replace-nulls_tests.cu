/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <cristhian@blazingdb.com>
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

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>

#include <cudf.h>

#include <thrust/device_vector.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <vector>

// This is the main test feature
template <class T>
struct ReplaceNullTest : public GdfTest
{
  std::vector<T>              replace_column;
  std::vector<gdf_valid_type> replace_valid_column;
  std::vector<T>              new_values_column;

  gdf_col_pointer gdf_replace_column;
  gdf_col_pointer gdf_new_values_column;

  gdf_column* gdf_raw_replace_column;
  gdf_column* gdf_raw_new_values_column;

  ReplaceNullTest()
  {
    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    static size_t number_of_instantiations{0};
    std::srand(number_of_instantiations++);
  }

  ~ReplaceNullTest()
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
                    const std::vector<gdf_valid_type> & replace_column_valid_list,
                    const std::initializer_list<T> &new_values_column_list,
                    bool print = false)
  {
    replace_column    = replace_column_list;
    replace_valid_column = replace_column_valid_list;
    new_values_column = new_values_column_list;

    gdf_replace_column    = create_gdf_column(replace_column, replace_column_valid_list);
    gdf_new_values_column = create_gdf_column(new_values_column);

    gdf_raw_replace_column = gdf_replace_column.get();
    gdf_raw_new_values_column = gdf_new_values_column.get();

    if(print)
    {
      std::cout << "replace column(s) created. Size: " << replace_column.size() << std::endl;
      print_gdf_column(gdf_raw_replace_column);
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
    if (new_values_column.size() == 1) {
        int k = 0;
        auto pred = [&, this] (T element) {
          bool toBeReplaced = false;
          if( !gdf_is_valid(this->replace_valid_column.data(), k)) {
            toBeReplaced = true;
          }
          ++k;
          return toBeReplaced;
        };
        std::replace_if(reference_result.begin(), reference_result.end(), pred, new_values_column[0]);  
    } else {
        auto pred = [&, this] (size_t k) {
          bool toBeReplaced = false;
          if( !gdf_is_valid(this->replace_valid_column.data(), k)) {
            toBeReplaced = true;
          }
          return toBeReplaced;
        };

        for (size_t index=0; index < new_values_column.size(); index++) {
          if ( pred(index) ) {
            reference_result[index] = new_values_column[index];
          }
        }
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

    gdf_error status = gdf_replace_nulls(gdf_raw_replace_column, gdf_raw_new_values_column);

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

using Types = testing::Types<int32_t>;

TYPED_TEST_CASE(ReplaceNullTest, Types);

// This test is used for debugging purposes and is disabled by default.
// The input sizes are small and has a large amount of debug printing enabled.
TYPED_TEST(ReplaceNullTest, case1)
{
  this->create_input({7, 5, 6, 3, 1, 2, 8, 4, 1, 2, 8, 4, 1, 2, 8, 4}, {0xF0, 0x0F}, {-1}, true);

  auto reference_result = this->compute_reference_solution(true);
  auto gdf_result = this->compute_gdf_result(true);
  
  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";
   // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

 TYPED_TEST(ReplaceNullTest, case2)
{
  this->create_input({7, 5, 6, 3, 1, 2, 8, 4, 1, 2, 8, 4, 1, 2, 8, 4}, {0xF0, 0x0F}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, true);

  auto reference_result = this->compute_reference_solution(true);
  auto gdf_result = this->compute_gdf_result(true);
  
  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";
   // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}
