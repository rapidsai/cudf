/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/edit_distance.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct TextEditDistanceTest : public cudf::test::BaseFixture {};

TEST_F(TextEditDistanceTest, EditDistance)
{
  auto const input = cudf::test::strings_column_wrapper(
    {"dog", "", "cat", "mouse", "pup", "", "puppy", "thé"}, {1, 0, 1, 1, 1, 1, 1, 1});
  auto sv = cudf::strings_column_view(input);

  {
    auto const targets = cudf::test::strings_column_wrapper(
      {"hog", "not", "cake", "house", "fox", "", "puppy", "the"}, {1, 1, 1, 1, 1, 0, 1, 1});
    auto tv = cudf::strings_column_view(targets);

    auto results = nvtext::edit_distance(sv, tv);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 3, 2, 1, 3, 0, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper single({"pup"});
    auto tv      = cudf::strings_column_view(single);
    auto results = nvtext::edit_distance(sv, tv);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({3, 3, 3, 4, 0, 3, 2, 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper single({"pup"}, {1});
    auto tv = cudf::strings_column_view(single);
    std::vector<char const*> h_input(516, "cup");
    auto input    = cudf::test::strings_column_wrapper(h_input.begin(), h_input.end());
    auto sv       = cudf::strings_column_view(input);
    auto results  = nvtext::edit_distance(sv, tv);
    auto begin    = thrust::constant_iterator<int32_t>(1);
    auto expected = cudf::test::fixed_width_column_wrapper<int32_t>(begin, begin + h_input.size());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(TextEditDistanceTest, EditDistanceLong)
{
  auto const input1 = cudf::test::strings_column_wrapper(
    {"the lady brown fox jumps down the wall of the castle with wide windows",
     "the lady brown fox jumps down the wall of thé castlé with wide windows",
     "thé lady brown fox jumps down the wall of the castle with wide windows",
     "the lazy brown dog jumps upon the hill of the castle with long windows",  // exact one
     "why the lazy brown dog jumps upon the hill of the castle with long windows",
     "the lazy brown dog jumps upon the hill of the castle",
     "lazy brown dog jumps upon hill"});
  auto const input2 = cudf::test::strings_column_wrapper(
    {"the lazy brown dog jumps upon the hill of the castle with long windows"});
  auto sv1 = cudf::strings_column_view(input1);
  auto sv2 = cudf::strings_column_view(input2);

  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>({12, 14, 13, 0, 4, 18, 40});
  auto results  = nvtext::edit_distance(sv1, sv2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextEditDistanceTest, EditDistanceMatrix)
{
  std::vector<char const*> h_strings{"dog", nullptr, "hog", "frog", "cat", "", "hat", "clog"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  {
    auto results = nvtext::edit_distance_matrix(cudf::strings_column_view(strings));

    using LCW = cudf::test::lists_column_wrapper<int32_t>;
    LCW expected({LCW{0, 3, 1, 2, 3, 3, 3, 2},
                  LCW{3, 0, 3, 4, 3, 0, 3, 4},
                  LCW{1, 3, 0, 2, 3, 3, 2, 2},
                  LCW{2, 4, 2, 0, 4, 4, 4, 2},
                  LCW{3, 3, 3, 4, 0, 3, 1, 3},
                  LCW{3, 0, 3, 4, 3, 0, 3, 4},
                  LCW{3, 3, 2, 4, 1, 3, 0, 4},
                  LCW{2, 4, 2, 2, 3, 4, 4, 0}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(TextEditDistanceTest, EmptyTest)
{
  auto strings = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  cudf::strings_column_view strings_view(strings->view());
  auto results = nvtext::edit_distance(strings_view, strings_view);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::edit_distance_matrix(strings_view);
  EXPECT_EQ(results->size(), 0);
}

TEST_F(TextEditDistanceTest, ErrorsTest)
{
  auto input   = cudf::test::strings_column_wrapper({"pup"});
  auto targets = cudf::test::strings_column_wrapper({"pup", ""});
  auto svi     = cudf::strings_column_view(input);
  auto tvi     = cudf::strings_column_view(targets);
  EXPECT_THROW(nvtext::edit_distance(svi, tvi), std::invalid_argument);
  EXPECT_THROW(nvtext::edit_distance_matrix(svi), std::invalid_argument);

  auto single = cudf::test::strings_column_wrapper({"pup"}, {0});
  auto sv1    = cudf::strings_column_view(single);
  EXPECT_THROW(nvtext::edit_distance(svi, sv1), std::invalid_argument);
}
