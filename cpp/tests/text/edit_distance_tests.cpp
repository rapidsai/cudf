/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
#include <cudf_test/debug_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/edit_distance.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct TextEditDistanceTest : public cudf::test::BaseFixture {};

TEST_F(TextEditDistanceTest, EditDistance)
{
  std::vector<char const*> h_strings{"dog", nullptr, "cat", "mouse", "pup", "", "puppy", "thé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto sv = cudf::strings_column_view(strings);

  {
    std::vector<char const*> h_targets{
      "hog", "not", "cake", "house", "fox", nullptr, "puppy", "the"};
    cudf::test::strings_column_wrapper targets(
      h_targets.begin(),
      h_targets.end(),
      thrust::make_transform_iterator(h_targets.begin(), [](auto str) { return str != nullptr; }));
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
  auto input1 = cudf::test::strings_column_wrapper(
    {"the lady brown fox jumps down the wall of the castle with wide windows",
     "the lady brown fox jumps down the wall of thé castlé with wide windows",
     "thé lady brown fox jumps down the wall of the castle with wide windows",
     "the lazy brown dog jumps upon the hill of the castle with long windows",  // exact one
     "why the lazy brown dog jumps upon the hill of the castle with long windows",
     "the lazy brown dog jumps upon the hill of the castle",
     "lazy brown dog jumps upon hill"});
  auto input2 = cudf::test::strings_column_wrapper(
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
