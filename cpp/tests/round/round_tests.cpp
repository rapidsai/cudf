/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

// #include <cudf/column/column.hpp>
// #include <cudf/column/column_view.hpp>
#include <cudf/round.hpp>
// #include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
// #include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
// #include <cudf_test/type_lists.hpp>

struct RoundTests : public cudf::test::BaseFixture {
};

TEST_F(RoundTests, SimpleFixedPointTest)
{
  using namespace numeric;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<int32_t>;

  auto const input    = fp_wrapper{{1140, 1150, 1160}, scale_type{-3}};
  auto const expected = fp_wrapper{{11, 12, 12}, scale_type{-1}};
  // auto const result   = cudf::round(col, 1, cudf::round_option::HALF_UP);

  EXPECT_THROW(cudf::round(input, 1, cudf::round_option::HALF_UP), cudf::logic_error);

  // CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, SimpleFloatingPointTest0)
{
  using namespace numeric;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<float>;

  auto const input    = fw_wrapper{1.4, 1.5, 1.6};
  auto const expected = fw_wrapper{1, 2, 2};
  auto const result   = cudf::round(input, 0, cudf::round_option::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, SimpleFloatingPointTest1)
{
  using namespace numeric;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<float>;

  auto const input    = fw_wrapper{1.24, 1.25, 1.26};
  auto const expected = fw_wrapper{1.2, 1.3, 1.3};
  auto const result   = cudf::round(input, 1, cudf::round_option::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, SimpleFloatingPointTestNeg1)
{
  using namespace numeric;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<float>;

  auto const input    = fw_wrapper{12, 135, 1454, 1455, 1456};
  auto const expected = fw_wrapper{10, 140, 1450, 1460, 1460};
  auto const result   = cudf::round(input, -1, cudf::round_option::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, SimpleFloatingPointTestNeg2)
{
  using namespace numeric;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<float>;

  auto const input    = fw_wrapper{12, 135, 1454, 1455, 1500};
  auto const expected = fw_wrapper{0, 100, 1500, 1500, 1500};
  auto const result   = cudf::round(input, -2, cudf::round_option::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

CUDF_TEST_PROGRAM_MAIN()
