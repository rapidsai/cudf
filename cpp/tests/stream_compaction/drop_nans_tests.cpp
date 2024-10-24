/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cmath>

struct DropNANsTest : public cudf::test::BaseFixture {};

TEST_F(DropNANsTest, MixedNANsAndNull)
{
  using F = float;
  using D = double;
  cudf::test::fixed_width_column_wrapper<float> col1{
    {F(1.0), F(2.0), F(NAN), F(NAN), F(5.0), F(6.0)}, {true, true, false, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10},
                                                       {true, true, false, true, true, false}};
  cudf::test::fixed_width_column_wrapper<double> col3{{D(NAN), 40.0, 70.0, 5.0, 2.0, 10.0},
                                                      {true, true, false, true, true, false}};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 2};
  cudf::test::fixed_width_column_wrapper<float> col1_expected{{2.0, 3.0, 5.0, 6.0},
                                                              {true, false, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{40, 70, 2, 10},
                                                                {true, false, true, false}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{40.0, 70.0, 2.0, 10.0},
                                                               {true, false, true, false}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::drop_nans(input, keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(DropNANsTest, NoNANs)
{
  cudf::test::fixed_width_column_wrapper<float> col1{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                                                     {true, true, false, true, true, true}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10},
                                                       {true, true, true, true, false, true}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10},
                                                      {true, true, false, true, true, true}};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 2};

  auto got = cudf::drop_nans(input, keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(DropNANsTest, MixedWithThreshold)
{
  using F = float;
  using D = double;
  cudf::test::fixed_width_column_wrapper<float> col1{
    {F(1.0), F(2.0), F(NAN), F(NAN), F(5.0), F(6.0)}, {true, true, false, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10},
                                                       {true, true, false, true, true, false}};
  cudf::test::fixed_width_column_wrapper<double> col3{{D(NAN), 40.0, 70.0, D(NAN), 2.0, 10.0},
                                                      {true, true, false, true, true, false}};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 2};
  cudf::test::fixed_width_column_wrapper<float> col1_expected{{1.0, 2.0, 3.0, 5.0, 6.0},
                                                              {true, true, false, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{10, 40, 70, 2, 10},
                                                                {true, true, false, true, false}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{D(NAN), 40.0, 70.0, 2.0, 10.0},
                                                               {true, true, false, true, false}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::drop_nans(input, keys, 1);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(DropNANsTest, EmptyTable)
{
  cudf::table_view input{std::vector<cudf::column_view>()};
  cudf::table_view expected{std::vector<cudf::column_view>()};
  std::vector<cudf::size_type> keys{};

  auto got = cudf::drop_nans(input, keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(DropNANsTest, EmptyColumns)
{
  cudf::test::fixed_width_column_wrapper<float> col1{};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{};
  cudf::test::fixed_width_column_wrapper<double> col3{};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 2};
  cudf::test::fixed_width_column_wrapper<float> col1_expected{};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::drop_nans(input, keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(DropNANsTest, EmptyKeys)
{
  using F = float;
  cudf::test::fixed_width_column_wrapper<float> col1{
    {F(1.0), F(2.0), F(NAN), F(NAN), F(5.0), F(6.0)}, {true, true, false, true, true, false}};
  cudf::table_view input{{col1}};
  std::vector<cudf::size_type> keys{};

  auto got = cudf::drop_nans(input, keys);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(DropNANsTest, NonFloatingKey)
{
  cudf::test::fixed_width_column_wrapper<float> col1{{1.0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{2};
  cudf::test::fixed_width_column_wrapper<double> col3{{3.0}};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 1};
  EXPECT_THROW(cudf::drop_nans(input, keys), cudf::logic_error);
}
