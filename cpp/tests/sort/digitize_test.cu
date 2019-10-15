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


#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>
#include <cudf/cudf.h>


#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <thrust/device_vector.h>

#include <gtest/gtest.h>

template <class ColumnType>
struct DigitizeTest : public GdfTest {
  using gdf_col_pointer =
    typename std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;

  std::vector<ColumnType> col_in_data;
  std::vector<ColumnType> bins_data;
  std::vector<gdf_index_type> out_data;

  gdf_col_pointer col_in;
  gdf_col_pointer bins;

  DigitizeTest(){
    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    static size_t number_of_instantiations{0};
    std::srand(number_of_instantiations++);
  }

  ~DigitizeTest(){}

  void initialize_data(size_t column_length, size_t column_range,
                       size_t bins_length, size_t bins_range)
  {
    initialize_vector(col_in_data, column_length, column_range, false);
    col_in = create_gdf_column(col_in_data);

    initialize_vector(bins_data, bins_length, bins_range, true);
    bins = create_gdf_column(bins_data);

  }

  gdf_error digitize(bool right) {

    rmm::device_vector<gdf_index_type> out_indices_dev(col_in->size);
    gdf_error result = gdf_digitize(col_in.get(), bins.get(), right, out_indices_dev.data().get());
    out_data.resize(out_indices_dev.size());
    cudaMemcpy(out_data.data(),
               out_indices_dev.data().get(),
               out_indices_dev.size() * sizeof(gdf_index_type),
               cudaMemcpyDeviceToHost);
    return result;
  }
};

typedef ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double> ValidGdfTypes;
TYPED_TEST_CASE(DigitizeTest, ValidGdfTypes);

TYPED_TEST(DigitizeTest, UpperBound)
{
  this->initialize_data(1000, 56, 4, 100);
  gdf_error result = this->digitize(true);
  EXPECT_EQ(result, GDF_SUCCESS);
}

TYPED_TEST(DigitizeTest, LowerBound)
{
  this->initialize_data(10000, 60, 10, 100);
  gdf_error result = this->digitize(false);
  EXPECT_EQ(result, GDF_SUCCESS);
}

void digitize_detail(bool right, const std::vector<int32_t>& expected) {
  std::vector<double> bins_data{0, 2, 5, 7, 8};
  gdf_col_pointer bins = create_gdf_column(bins_data);

  std::vector<double> col_in_data{-10, 0, 1, 2, 3, 8, 9};
  gdf_col_pointer col_in = create_gdf_column(col_in_data);

  rmm::device_vector<gdf_index_type> out_indices_dev(col_in_data.size());
  gdf_error result = gdf_digitize(col_in.get(), bins.get(), right, out_indices_dev.data().get());

  std::vector<gdf_index_type> out_indices(out_indices_dev.size());
  cudaMemcpy(out_indices.data(),
             out_indices_dev.data().get(),
             out_indices_dev.size() * sizeof(gdf_index_type),
             cudaMemcpyDeviceToHost);
  EXPECT_EQ(result, GDF_SUCCESS);

  const size_t num_rows = col_in_data.size();
  for (unsigned int i = 0; i < num_rows; ++i) {
    EXPECT_EQ(expected[i], out_indices[i]);
  }
}

TYPED_TEST(DigitizeTest, UpperBoundDetail) {
  std::vector<int32_t> expected{0, 0, 1, 1, 2, 4, 5};
  digitize_detail(true, expected);
}

TYPED_TEST(DigitizeTest, LowerBoundDetail) {
  std::vector<int32_t> expected{0, 1, 1, 2, 2, 5, 5};
  digitize_detail(false, expected);
}
