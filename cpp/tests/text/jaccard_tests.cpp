/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/jaccard.hpp>

struct JaccardTest : public cudf::test::BaseFixture {};

TEST_F(JaccardTest, Basic)
{
  auto input1 =
    cudf::test::strings_column_wrapper({"the quick brown fox", "jumped over the lazy dog."});
  auto input2 =
    cudf::test::strings_column_wrapper({"the slowest brown cat", "crawled under the jumping fox"});

  auto view1 = cudf::strings_column_view(input1);
  auto view2 = cudf::strings_column_view(input2);

  auto results = nvtext::jaccard_index(view1, view2, 5);

  auto expected = cudf::test::fixed_width_column_wrapper<float>({0.103448279f, 0.0697674453f});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<float>({1.0f, 1.0f});
  results  = nvtext::jaccard_index(view1, view1, 5);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  results = nvtext::jaccard_index(view2, view2, 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(JaccardTest, WithNulls)
{
  auto input1 =
    cudf::test::strings_column_wrapper({"brown fox", "jumps over dog", "", ""}, {1, 1, 0, 1});
  auto input2 =
    cudf::test::strings_column_wrapper({"brown cat", "jumps on fox", "", ""}, {1, 1, 1, 0});

  auto view1 = cudf::strings_column_view(input1);
  auto view2 = cudf::strings_column_view(input2);

  auto results = nvtext::jaccard_index(view1, view2, 5);

  auto expected =
    cudf::test::fixed_width_column_wrapper<float>({0.25f, 0.200000003f, 0.f, 0.f}, {1, 1, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<float>({1.0f, 1.0f, 0.f, 0.f}, {1, 1, 0, 1});
  results  = nvtext::jaccard_index(view1, view1, 7);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(JaccardTest, Errors)
{
  auto input = cudf::test::strings_column_wrapper({"1", "2", "3"});
  auto view  = cudf::strings_column_view(input);
  // invalid parameter value
  EXPECT_THROW(nvtext::jaccard_index(view, view, 1), std::invalid_argument);
  // invalid size
  auto input2 = cudf::test::strings_column_wrapper({"1", "2"});
  auto view2  = cudf::strings_column_view(input2);
  EXPECT_THROW(nvtext::jaccard_index(view, view2, 5), std::invalid_argument);
}
