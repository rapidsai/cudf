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

#include <gtest/gtest.h>

struct KafkaDatasourceTest : public ::testing::Test {
};

TEST_F(KafkaDatasourceTest, UserImplementedSource)
{
  //   constexpr auto num_rows = 10;
  //   auto int8_values        = random_values<int8_t>(num_rows);
  //   auto int16_values       = random_values<int16_t>(num_rows);
  //   auto int32_values       = random_values<int32_t>(num_rows);

  //   std::ostringstream csv_data;
  //   for (int i = 0; i < num_rows; ++i) {
  //     csv_data << std::to_string(int8_values[i]) << "," << int16_values[i] << "," <<
  //     int32_values[i]
  //              << "\n";
  //   }
  //   TestSource source{csv_data.str()};
  //   cudf_io::read_csv_args in_args{cudf_io::source_info{&source}};
  //   in_args.dtype  = {"int8", "int16", "int32"};
  //   in_args.header = -1;
  //   auto result    = cudf_io::read_csv(in_args);

  //   auto const view = result.tbl->view();
  //   expect_column_data_equal(int8_values, view.column(0));
  //   expect_column_data_equal(int16_values, view.column(1));
  //   expect_column_data_equal(int32_values, view.column(2));
}