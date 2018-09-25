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
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include "../test_utils/gdf_test_utils.cuh"


TEST(ValidsTest, FirstRowValid)
{
  std::vector<int> data(4);
  std::vector<gdf_valid_type> valid{0x1};

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, 1, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(1, count);
}

TEST(ValidsTest, EightRowsValid)
{
  std::vector<int> data(8);
  std::vector<gdf_valid_type> valid{0xFF};

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, 8, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(8, count);
}

TEST(ValidsTest, EveryOtherBit)
{
  std::vector<int> data(8);
  std::vector<gdf_valid_type> valid{0xAA};

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, 8, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(4, count);
}

TEST(ValidsTest, OtherEveryOtherBit)
{
  std::vector<int> data(8);
  std::vector<gdf_valid_type> valid{0x55};

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, 8, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(4, count);
}

TEST(ValidsTest, MultipleOfEight)
{
  const int num_rows = 1024;
  std::vector<int> data(num_rows);

  const int num_masks = std::ceil(num_rows/8);
  std::vector<gdf_valid_type> valid(num_masks,0x01);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(128, count);
}

TEST(ValidsTest, NotMultipleOfEight)
{
  const int num_rows = 1023;
  std::vector<int> data(num_rows);

  const int num_masks = std::ceil(num_rows/8);
  std::vector<gdf_valid_type> valid(num_masks, 0x01);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code = gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS,error_code) << "GDF Operation did not complete successfully.";

  EXPECT_EQ(127, count);
}



