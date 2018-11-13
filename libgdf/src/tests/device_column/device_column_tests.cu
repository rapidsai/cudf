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
#include <chrono>

#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include "../test_utils/gdf_test_utils.cuh"
#include "gdf_test_fixtures.h"
#include "../../device_column.cuh"

struct DeviceColumnTest  : public GdfTest 
{

};

TEST_F(DeviceColumnTest, FirstTest)
{
  int const num_rows{10};
  std::vector<int> data(num_rows,0);
  std::vector<gdf_valid_type> valids(gdf_get_num_chars_bitmask(num_rows),0xFF);

  auto gdf_col = create_gdf_column(data,valids);

  auto device_col = DeviceColumn::make_device_column(*gdf_col);
}


