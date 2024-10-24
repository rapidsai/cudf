/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/testing_main.hpp>

#include <src/io/utilities/row_selection.hpp>

#include <limits>

using cudf::io::detail::skip_rows_num_rows_from_options;

// Base test fixture for tests
struct FromOptsTest : public cudf::test::BaseFixture {};

TEST_F(FromOptsTest, PassThrough)
{
  // select all rows
  {
    auto [out_skip, out_num] = skip_rows_num_rows_from_options(0, 100, 100);
    EXPECT_EQ(out_skip, 0);
    EXPECT_EQ(out_num, 100);
  }

  // select all except first skip_rows
  {
    auto [out_skip, out_num] = skip_rows_num_rows_from_options(10, 90, 100);
    EXPECT_EQ(out_skip, 10);
    EXPECT_EQ(out_num, 90);
  }

  // select first num_rows
  {
    auto [out_skip, out_num] = skip_rows_num_rows_from_options(0, 60, 100);
    EXPECT_EQ(out_skip, 0);
    EXPECT_EQ(out_num, 60);
  }
}

TEST_F(FromOptsTest, DefaultNumRows)
{
  // no skip_rows
  {
    auto [out_skip, out_num] = skip_rows_num_rows_from_options(0, std::nullopt, 100);
    EXPECT_EQ(out_skip, 0);
    EXPECT_EQ(out_num, 100);
  }

  // with skip_rows
  {
    auto [out_skip, out_num] = skip_rows_num_rows_from_options(20, std::nullopt, 100);
    EXPECT_EQ(out_skip, 20);
    EXPECT_EQ(out_num, 80);
  }
}

TEST_F(FromOptsTest, InputSize32BitOverflow)
{
  // Input number of rows too large to fit into cudf::size_type
  // Test that we can still select rows from such input
  auto const too_large_for_32bit = std::numeric_limits<int64_t>::max();

  // no num_rows
  {
    auto [out_skip, out_num] =
      skip_rows_num_rows_from_options(too_large_for_32bit - 10, std::nullopt, too_large_for_32bit);
    EXPECT_EQ(out_skip, too_large_for_32bit - 10);
    EXPECT_EQ(out_num, 10);
  }

  // with num_rows
  {
    auto [out_skip, out_num] =
      skip_rows_num_rows_from_options(too_large_for_32bit - 100, 30, too_large_for_32bit);
    EXPECT_EQ(out_skip, too_large_for_32bit - 100);
    EXPECT_EQ(out_num, 30);
  }
}

TEST_F(FromOptsTest, LimitOptionsToFileRows)
{
  // limit skip_rows without num_rows
  {
    auto [out_skip, out_num] = skip_rows_num_rows_from_options(1000, std::nullopt, 100);
    EXPECT_EQ(out_skip, 100);
    EXPECT_EQ(out_num, 0);
  }

  // limit skip_rows with num_rows
  {
    auto [out_skip, out_num] = skip_rows_num_rows_from_options(1000, 2, 100);
    EXPECT_EQ(out_skip, 100);
    EXPECT_EQ(out_num, 0);
  }

  // limit num_rows without skip_rows
  {
    auto [out_skip, out_num] = skip_rows_num_rows_from_options(0, 1000, 100);
    EXPECT_EQ(out_skip, 0);
    EXPECT_EQ(out_num, 100);
  }

  // limit num_rows with skip_rows
  {
    auto [out_skip, out_num] = skip_rows_num_rows_from_options(10, 1000, 100);
    EXPECT_EQ(out_skip, 10);
    EXPECT_EQ(out_num, 90);
  }
}

CUDF_TEST_PROGRAM_MAIN()
