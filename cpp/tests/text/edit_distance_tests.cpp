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
  std::vector<char const*> h_strings{"dog", nullptr, "cat", "mouse", "pup", "", "puppy", "thé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<char const*> h_targets{"hog", "not", "cake", "house", "fox", nullptr, "puppy", "the"};
  cudf::test::strings_column_wrapper targets(
    h_targets.begin(),
    h_targets.end(),
    thrust::make_transform_iterator(h_targets.begin(), [](auto str) { return str != nullptr; }));
  {
    auto results =
      nvtext::edit_distance(cudf::strings_column_view(strings), cudf::strings_column_view(targets));
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 3, 2, 1, 3, 0, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper single({"pup"});
    auto results =
      nvtext::edit_distance(cudf::strings_column_view(strings), cudf::strings_column_view(single));
    cudf::test::fixed_width_column_wrapper<int32_t> expected({3, 3, 3, 4, 0, 3, 2, 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
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
  cudf::test::strings_column_wrapper strings({"pup"});
  cudf::test::strings_column_wrapper targets({"pup", ""});
  EXPECT_THROW(
    nvtext::edit_distance(cudf::strings_column_view(strings), cudf::strings_column_view(targets)),
    cudf::logic_error);
  EXPECT_THROW(nvtext::edit_distance_matrix(cudf::strings_column_view(strings)), cudf::logic_error);
}
