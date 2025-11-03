/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
  auto input1 = cudf::test::strings_column_wrapper({"brown fox", "jumps over dog", "", ""},
                                                   {true, true, false, true});
  auto input2 = cudf::test::strings_column_wrapper({"brown cat", "jumps on fox", "", ""},
                                                   {true, true, true, false});

  auto view1 = cudf::strings_column_view(input1);
  auto view2 = cudf::strings_column_view(input2);

  auto results = nvtext::jaccard_index(view1, view2, 5);

  auto expected = cudf::test::fixed_width_column_wrapper<float>({0.25f, 0.200000003f, 0.f, 0.f},
                                                                {true, true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<float>({1.0f, 1.0f, 0.f, 0.f},
                                                           {true, true, false, true});
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
