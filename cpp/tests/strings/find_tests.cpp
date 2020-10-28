/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <vector>

struct StringsFindTest : public cudf::test::BaseFixture {
};

TEST_F(StringsFindTest, Find)
{
  cudf::test::strings_column_wrapper strings({"Héllo", "thesé", "", "lease", "tést strings", ""},
                                             {1, 1, 0, 1, 1, 1});
  auto strings_view = cudf::strings_column_view(strings);

  {
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 4, -1, -1, 1, -1},
                                                             {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::find(strings_view, cudf::string_scalar("é"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<int32_t> expected({3, -1, -1, 0, -1, -1},
                                                             {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::rfind(strings_view, cudf::string_scalar("l"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 0, 0, 0, 0},
                                                             {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::find(strings_view, cudf::string_scalar(""));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<int32_t> expected({5, 5, 0, 5, 12, 0},
                                                             {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::rfind(strings_view, cudf::string_scalar(""));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsFindTest, Contains)
{
  cudf::test::strings_column_wrapper strings({"Héllo", "thesé", "", "lease", "tést strings", ""},
                                             {1, 1, 0, 1, 1, 1});
  auto strings_view = cudf::strings_column_view(strings);
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 1, 0, 0}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::contains(strings_view, cudf::string_scalar("e"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper targets({"Hello", "é", "e", "x", "", ""},
                                               {1, 1, 1, 1, 1, 0});
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 0}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::contains(strings_view, cudf::strings_column_view(targets));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsFindTest, StartsWith)
{
  cudf::test::strings_column_wrapper strings({"Héllo", "thesé", "", "lease", "tést strings", ""},
                                             {1, 1, 0, 1, 1, 1});
  auto strings_view = cudf::strings_column_view(strings);
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 0}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::starts_with(strings_view, cudf::string_scalar("t"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<const char*> h_targets{"éa", "th", "e", "ll", "tést strings", ""};
    cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());

    auto targets_view = cudf::strings_column_view(targets);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 1}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::starts_with(strings_view, targets_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 0, 0}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::starts_with(strings_view, cudf::string_scalar("thesé"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<const char*> h_targets{"éa", "th", "e", "ll", nullptr, ""};
    cudf::test::strings_column_wrapper targets(
      h_targets.begin(),
      h_targets.end(),
      thrust::make_transform_iterator(h_targets.begin(), [](auto str) { return str != nullptr; }));

    auto targets_view = cudf::strings_column_view(targets);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 0, 1}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::starts_with(strings_view, targets_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsFindTest, EndsWith)
{
  cudf::test::strings_column_wrapper strings({"Héllo", "thesé", "", "lease", "tést strings", ""},
                                             {1, 1, 0, 1, 1, 1});
  auto strings_view = cudf::strings_column_view(strings);
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 0, 0, 1, 0, 0}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::ends_with(strings_view, cudf::string_scalar("se"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<const char*> h_targets{"éa", "sé", "th", "ll", "tést strings", ""};
    cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());

    auto targets_view = cudf::strings_column_view(targets);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 1}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::ends_with(strings_view, targets_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 0, 0}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::ends_with(strings_view, cudf::string_scalar("thesé"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<const char*> h_targets{"éa", "sé", "th", nullptr, "tést strings", ""};
    cudf::test::strings_column_wrapper targets(
      h_targets.begin(),
      h_targets.end(),
      thrust::make_transform_iterator(h_targets.begin(), [](auto str) { return str != nullptr; }));

    auto targets_view = cudf::strings_column_view(targets);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 1}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::ends_with(strings_view, targets_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsFindTest, ZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  auto results      = cudf::strings::find(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::rfind(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::contains(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::starts_with(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::ends_with(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::starts_with(strings_view, strings_view);
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::ends_with(strings_view, strings_view);
  EXPECT_EQ(results->size(), 0);
}

TEST_F(StringsFindTest, EmptyTarget)
{
  cudf::test::strings_column_wrapper strings({"Héllo", "thesé", "", "lease", "tést strings", ""},
                                             {1, 1, 0, 1, 1, 1});
  auto strings_view = cudf::strings_column_view(strings);

  cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 1, 1, 1, 1}, {1, 1, 0, 1, 1, 1});
  auto results = cudf::strings::contains(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::starts_with(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::ends_with(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsFindTest, AllEmpty)
{
  std::vector<std::string> h_strings{"", "", "", "", ""};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());

  std::vector<int32_t> h_expected32(h_strings.size(), -1);
  cudf::test::fixed_width_column_wrapper<int32_t> expected32(h_expected32.begin(),
                                                             h_expected32.end());

  std::vector<bool> h_expected8(h_strings.size(), 0);
  cudf::test::fixed_width_column_wrapper<bool> expected8(h_expected8.begin(), h_expected8.end());

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::find(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected32);
  results = cudf::strings::rfind(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected32);
  results = cudf::strings::contains(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
  results = cudf::strings::starts_with(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
  results = cudf::strings::ends_with(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
  std::vector<std::string> h_targets{"abc", "e", "fdg", "g", "p"};
  cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());
  auto targets_view = cudf::strings_column_view(targets);
  results           = cudf::strings::starts_with(strings_view, targets_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
  results = cudf::strings::ends_with(strings_view, targets_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
}

TEST_F(StringsFindTest, AllNull)
{
  std::vector<const char*> h_strings{nullptr, nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<int32_t> h_expected32(h_strings.size(), -1);
  cudf::test::fixed_width_column_wrapper<int32_t> expected32(
    h_expected32.begin(),
    h_expected32.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<bool> h_expected8(h_strings.size(), -1);
  cudf::test::fixed_width_column_wrapper<bool> expected8(
    h_expected8.begin(),
    h_expected8.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::find(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected32);
  results = cudf::strings::rfind(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected32);
  results = cudf::strings::contains(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
  results = cudf::strings::starts_with(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
  results = cudf::strings::ends_with(strings_view, cudf::string_scalar("e"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
  std::vector<std::string> h_targets{"abc", "e", "fdg", "p"};
  cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());
  auto targets_view = cudf::strings_column_view(targets);
  results           = cudf::strings::starts_with(strings_view, targets_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
  results = cudf::strings::ends_with(strings_view, targets_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);
}

TEST_F(StringsFindTest, ErrorCheck)
{
  cudf::test::strings_column_wrapper strings({"1", "2", "3", "4", "5", "6"});
  auto strings_view = cudf::strings_column_view(strings);
  cudf::test::strings_column_wrapper targets({"1", "2", "3", "4", "5"});
  auto targets_view = cudf::strings_column_view(targets);

  EXPECT_THROW(cudf::strings::contains(strings_view, targets_view), cudf::logic_error);
  EXPECT_THROW(cudf::strings::starts_with(strings_view, targets_view), cudf::logic_error);
  EXPECT_THROW(cudf::strings::ends_with(strings_view, targets_view), cudf::logic_error);

  EXPECT_THROW(cudf::strings::find(strings_view, cudf::string_scalar(""), 2, 1), cudf::logic_error);
  EXPECT_THROW(cudf::strings::rfind(strings_view, cudf::string_scalar(""), 2, 1),
               cudf::logic_error);
}

class FindParmsTest : public StringsFindTest, public testing::WithParamInterface<int32_t> {
};

TEST_P(FindParmsTest, Find)
{
  std::vector<std::string> h_strings{"hello", "", "these", "are stl", "safe"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  cudf::size_type position = GetParam();

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto results = cudf::strings::find(strings_view, cudf::string_scalar("e"), position);
    std::vector<int32_t> h_expected;
    for (auto itr = h_strings.begin(); itr != h_strings.end(); ++itr)
      h_expected.push_back((int32_t)(*itr).find("e", position));
    cudf::test::fixed_width_column_wrapper<int32_t> expected(h_expected.begin(), h_expected.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::rfind(strings_view, cudf::string_scalar("e"), 0, position + 1);
    std::vector<int32_t> h_expected;
    for (auto itr = h_strings.begin(); itr != h_strings.end(); ++itr)
      h_expected.push_back((int32_t)(*itr).rfind("e", position));
    cudf::test::fixed_width_column_wrapper<int32_t> expected(h_expected.begin(), h_expected.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto begin   = position;
    auto results = cudf::strings::find(strings_view, cudf::string_scalar(""), begin);
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {begin, (begin > 0 ? -1 : 0), begin, begin, begin});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto end = position + 1;
    results  = cudf::strings::rfind(strings_view, cudf::string_scalar(""), 0, end);
    cudf::test::fixed_width_column_wrapper<int32_t> rexpected({end, 0, end, end, end});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, rexpected);
  }
}

INSTANTIATE_TEST_CASE_P(StringsFindTest,
                        FindParmsTest,
                        testing::ValuesIn(std::array<int32_t, 4>{0, 1, 2, 3}));
