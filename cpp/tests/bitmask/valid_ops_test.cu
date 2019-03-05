/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <cudf.h>
#include <cudf/functions.h>

#include "cuda_profiler_api.h"

#include "tests/utilities/cudf_test_utils.cuh"
#include "tests/utilities/cudf_test_fixtures.h"

#include <chrono>

struct ValidsTest : public GdfTest {};

TEST_F(ValidsTest, NoValids)
{
  const int num_rows = 100;
  std::vector<int> data(num_rows);
  const int num_masks = std::ceil(num_rows/static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks,0x00);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(0, count);
}

TEST_F(ValidsTest, NullValids)
{
  int count{-1};
  const gdf_size_type size{100};
  gdf_error error_code = gdf_count_nonzero_mask(nullptr, size, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";
  EXPECT_EQ(size, count);
}

TEST_F(ValidsTest, NullCount)
{
  std::vector<int> data(0);
  std::vector<gdf_valid_type> valid{0x0};
  auto input_gdf_col = create_gdf_column(data, valid);
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, 1, nullptr);

  ASSERT_EQ(GDF_DATASET_EMPTY,error_code) << "Expected failure for null input.";
}

TEST_F(ValidsTest, FirstRowValid)
{
  std::vector<int> data(4);
  std::vector<gdf_valid_type> valid{0x1};

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, 1, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(1, count);
}

TEST_F(ValidsTest, EightRowsValid)
{
  std::vector<int> data(8);
  std::vector<gdf_valid_type> valid{0xFF};

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, 8, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(8, count);
}

TEST_F(ValidsTest, EveryOtherBit)
{
  std::vector<int> data(8);
  std::vector<gdf_valid_type> valid{0xAA};

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, 8, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(4, count);
}

TEST_F(ValidsTest, OtherEveryOtherBit)
{
  std::vector<int> data(8);
  std::vector<gdf_valid_type> valid{0x55};

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, 8, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(4, count);
}

TEST_F(ValidsTest, 15rows)
{
  const int num_rows = 15;
  std::vector<int> data(num_rows);
  const int num_masks = std::ceil(num_rows/static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks,0x01);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(2, count);
}

TEST_F(ValidsTest, 5rows)
{
  const int num_rows = 5;
  std::vector<int> data(num_rows);
  const int num_masks = std::ceil(num_rows/static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks,0x01);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(1, count);
}

TEST_F(ValidsTest, 10ValidRows)
{
  const int num_rows = 10;
  std::vector<float> data(num_rows);
  const int num_masks = std::ceil(num_rows/static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks,0xFF);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(10, count);
}

TEST_F(ValidsTest, MultipleOfEight)
{
  const int num_rows = 1024;
  std::vector<int> data(num_rows);

  const int num_masks = std::ceil(num_rows/static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks,0x01);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(128, count);
}

TEST_F(ValidsTest, NotMultipleOfEight)
{
  const int num_rows = 1023;
  std::vector<int> data(num_rows);

  const int num_masks = std::ceil(num_rows/static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks, 0x80);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(127, count);
}

TEST_F(ValidsTest, TenThousandRows)
{
  const int num_rows = 10000;
  std::vector<int> data(num_rows);

  const int num_masks = std::ceil(num_rows/static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks, 0xFF);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(10000, count);
}

TEST_F(ValidsTest, PerformanceTest)
{
  const int num_rows = 100000000;
  std::vector<int> data(num_rows);

  const int num_masks = std::ceil(num_rows/8);
  std::vector<gdf_valid_type> valid(num_masks, 0x55);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  
  auto start = std::chrono::system_clock::now();
  cudaProfilerStart();
  for(int i = 0; i < 1000; ++i)
    gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);
  cudaProfilerStop();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "Elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;
}



