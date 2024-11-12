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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <vector>

struct StringsFindTest : public cudf::test::BaseFixture {};

TEST_F(StringsFindTest, Find)
{
  cudf::test::strings_column_wrapper strings({"Héllo", "thesé", "", "lest", "tést strings", ""},
                                             {true, true, false, true, true, true});
  auto strings_view = cudf::strings_column_view(strings);

  {
    auto const target = cudf::string_scalar("é");
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {1, 4, -1, -1, 1, -1}, {true, true, false, true, true, true});
    auto results = cudf::strings::find(strings_view, target);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    results = cudf::strings::rfind(strings_view, target);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {3, -1, -1, 0, -1, -1}, {true, true, false, true, true, true});
    auto results = cudf::strings::rfind(strings_view, cudf::string_scalar("l"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const target = cudf::string_scalar("es");
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {-1, 2, -1, 1, -1, -1}, {true, true, false, true, true, true});
    auto results = cudf::strings::find(strings_view, target);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    results = cudf::strings::rfind(strings_view, target);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {0, 0, 0, 0, 0, 0}, {true, true, false, true, true, true});
    auto results = cudf::strings::find(strings_view, cudf::string_scalar(""));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {5, 5, 0, 4, 12, 0}, {true, true, false, true, true, true});
    auto results = cudf::strings::rfind(strings_view, cudf::string_scalar(""));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const targets = cudf::test::strings_column_wrapper({"l", "t", "", "x", "é", "o"});
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {2, 0, 0, -1, 1, -1}, {true, true, false, true, true, true});
    auto results = cudf::strings::find(strings_view, cudf::strings_column_view(targets));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {0, 0, 0, 0, 0, 0}, {true, true, false, true, true, true});
    auto results = cudf::strings::find(strings_view, strings_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsFindTest, FindWithNullTargets)
{
  cudf::test::strings_column_wrapper input({"hello hello", "thesé help", "", "helicopter", "", "x"},
                                           {true, true, false, true, true, true});
  auto strings_view = cudf::strings_column_view(input);

  auto const targets = cudf::test::strings_column_wrapper(
    {"lo he", "", "hhh", "cop", "help", "xyz"}, {true, false, true, true, true, true});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
    {3, -1, -1, 4, -1, -1}, {true, false, false, true, true, true});
  auto results = cudf::strings::find(strings_view, cudf::strings_column_view(targets));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsFindTest, FindLongStrings)
{
  cudf::test::strings_column_wrapper input(
    {"Héllo, there world and goodbye",
     "quick brown fox jumped over the lazy brown dog; the fat cats jump in place without moving",
     "the following code snippet demonstrates how to use search for values in an ordered range",
     "it returns the last position where value could be inserted without violating the ordering",
     "algorithms execution is parallelized as determined by an execution policy. t",
     "he this is a continuation of previous row to make sure string boundaries are honored",
     ""});
  auto view    = cudf::strings_column_view(input);
  auto results = cudf::strings::find(view, cudf::string_scalar("the"));
  auto expected =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({7, 28, 0, 11, -1, -1, -1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  auto targets =
    cudf::test::strings_column_wrapper({"the", "the", "the", "the", "the", "the", "the"});
  results = cudf::strings::find(view, cudf::strings_column_view(targets));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results  = cudf::strings::rfind(view, cudf::string_scalar("the"));
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({7, 48, 0, 77, -1, -1, -1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  targets  = cudf::test::strings_column_wrapper({"there", "cat", "the", "", "the", "are", "dog"});
  results  = cudf::strings::find(view, cudf::strings_column_view(targets));
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({7, 56, 0, 0, -1, 73, -1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results  = cudf::strings::find(view, cudf::string_scalar("ing"));
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({-1, 86, 10, 73, -1, 58, -1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results  = cudf::strings::rfind(view, cudf::string_scalar("ing"));
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({-1, 86, 10, 86, -1, 58, -1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsFindTest, Contains)
{
  cudf::test::strings_column_wrapper strings(
    {"Héllo", "thesé", "", "lease", "tést strings", "", "eé", "éte"},
    {true, true, false, true, true, true, true, true});
  auto strings_view = cudf::strings_column_view(strings);
  {
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {0, 1, 0, 1, 0, 0, 1, 1}, {true, true, false, true, true, true, true, true});
    auto results = cudf::strings::contains(strings_view, cudf::string_scalar("e"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {1, 1, 0, 0, 1, 0, 1, 1}, {true, true, false, true, true, true, true, true});
    auto results = cudf::strings::contains(strings_view, cudf::string_scalar("é"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper targets({"Hello", "é", "e", "x", "", "", "n", "t"},
                                               {true, true, true, true, true, false, true, true});
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {0, 1, 0, 0, 1, 0, 0, 1}, {true, true, false, true, true, true, true, true});
    auto results = cudf::strings::contains(strings_view, cudf::strings_column_view(targets));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsFindTest, ContainsLongStrings)
{
  cudf::test::strings_column_wrapper strings(
    {"Héllo, there world and goodbye",
     "quick brown fox jumped over the lazy brown dog; the fat cats jump in place without moving",
     "the following code snippet demonstrates how to use search for values in an ordered range",
     "it returns the last position where value could be inserted without violating the ordering",
     "algorithms execution is parallelized as determined by an execution policy. t",
     "he this is a continuation of previous row to make sure string boundaries are honored",
     "abcdefghijklmnopqrstuvwxyz 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ !@#$%^&*()~",
     ""});
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::contains(strings_view, cudf::string_scalar("e"));
  auto expected     = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results  = cudf::strings::contains(strings_view, cudf::string_scalar(" the "));
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 1, 0, 1, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results  = cudf::strings::contains(strings_view, cudf::string_scalar("a"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results  = cudf::strings::contains(strings_view, cudf::string_scalar("~"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsFindTest, StartsWith)
{
  cudf::test::strings_column_wrapper strings({"Héllo", "thesé", "", "lease", "tést strings", ""},
                                             {true, true, false, true, true, true});
  auto strings_view = cudf::strings_column_view(strings);
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 0},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::starts_with(strings_view, cudf::string_scalar("t"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<char const*> h_targets{"éa", "th", "e", "ll", "tést strings", ""};
    cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());

    auto targets_view = cudf::strings_column_view(targets);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 1},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::starts_with(strings_view, targets_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 0, 0},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::starts_with(strings_view, cudf::string_scalar("thesé"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<char const*> h_targets{"éa", "th", "e", "ll", nullptr, ""};
    cudf::test::strings_column_wrapper targets(
      h_targets.begin(),
      h_targets.end(),
      thrust::make_transform_iterator(h_targets.begin(), [](auto str) { return str != nullptr; }));

    auto targets_view = cudf::strings_column_view(targets);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 0, 1},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::starts_with(strings_view, targets_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsFindTest, EndsWith)
{
  cudf::test::strings_column_wrapper strings({"Héllo", "thesé", "", "lease", "tést strings", ""},
                                             {true, true, false, true, true, true});
  auto strings_view = cudf::strings_column_view(strings);
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 0, 0, 1, 0, 0},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::ends_with(strings_view, cudf::string_scalar("se"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<char const*> h_targets{"éa", "sé", "th", "ll", "tést strings", ""};
    cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());

    auto targets_view = cudf::strings_column_view(targets);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 1},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::ends_with(strings_view, targets_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 0, 0},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::ends_with(strings_view, cudf::string_scalar("thesé"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<char const*> h_targets{"éa", "sé", "th", nullptr, "tést strings", ""};
    cudf::test::strings_column_wrapper targets(
      h_targets.begin(),
      h_targets.end(),
      thrust::make_transform_iterator(h_targets.begin(), [](auto str) { return str != nullptr; }));

    auto targets_view = cudf::strings_column_view(targets);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 1},
                                                          {true, true, false, true, true, true});
    auto results = cudf::strings::ends_with(strings_view, targets_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsFindTest, ZeroSizeStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto strings_view                   = cudf::strings_column_view(zero_size_strings_column);
  auto results = cudf::strings::find(strings_view, cudf::string_scalar("é"));
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
                                             {true, true, false, true, true, true});
  auto strings_view = cudf::strings_column_view(strings);

  cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 1, 1, 1, 1},
                                                        {true, true, false, true, true, true});
  auto results = cudf::strings::contains(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::starts_with(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::ends_with(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_find(
    {0, 0, 0, 0, 0, 0}, {true, true, false, true, true, true});
  results = cudf::strings::find(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_find);
  auto expected_rfind = cudf::strings::count_characters(strings_view);
  results             = cudf::strings::rfind(strings_view, cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, *expected_rfind);
}

TEST_F(StringsFindTest, AllEmpty)
{
  std::vector<std::string> h_strings{"", "", "", "", ""};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());

  std::vector<cudf::size_type> h_expected32(h_strings.size(), -1);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected32(h_expected32.begin(),
                                                                     h_expected32.end());

  std::vector<bool> h_expected8(h_strings.size(), false);
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
  std::vector<char const*> h_strings{nullptr, nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<cudf::size_type> h_expected32(h_strings.size(), -1);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected32(
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
  EXPECT_THROW(cudf::strings::find(strings_view, targets_view), cudf::logic_error);
  EXPECT_THROW(cudf::strings::find(strings_view, strings_view, -1), cudf::logic_error);
}

class FindParmsTest : public StringsFindTest,
                      public testing::WithParamInterface<cudf::size_type> {};

TEST_P(FindParmsTest, Find)
{
  std::vector<std::string> h_strings{"hello", "", "these", "are stl", "safe"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  cudf::size_type position = GetParam();

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto results = cudf::strings::find(strings_view, cudf::string_scalar("e"), position);
    std::vector<cudf::size_type> h_expected;
    for (auto& h_string : h_strings)
      h_expected.push_back(static_cast<cudf::size_type>(h_string.find("e", position)));
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(h_expected.begin(),
                                                                     h_expected.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::rfind(strings_view, cudf::string_scalar("e"), 0, position + 1);
    std::vector<cudf::size_type> h_expected;
    for (auto& h_string : h_strings)
      h_expected.push_back(static_cast<cudf::size_type>(h_string.rfind("e", position)));
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(h_expected.begin(),
                                                                     h_expected.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto begin   = static_cast<cudf::size_type>(position);
    auto results = cudf::strings::find(strings_view, cudf::string_scalar(""), begin);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      {begin, (begin > 0 ? -1 : 0), begin, begin, begin});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto end = static_cast<cudf::size_type>(position + 1);
    results  = cudf::strings::rfind(strings_view, cudf::string_scalar(""), 0, end);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> rexpected({end, 0, end, end, end});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, rexpected);
  }
  {
    std::vector<std::string> h_targets({"l", "", "", "l", "s"});
    std::vector<cudf::size_type> h_expected;
    for (std::size_t i = 0; i < h_strings.size(); ++i)
      h_expected.push_back(static_cast<cudf::size_type>(h_strings[i].find(h_targets[i], position)));
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(h_expected.begin(),
                                                                     h_expected.end());
    cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());
    auto results = cudf::strings::find(strings_view, cudf::strings_column_view(targets), position);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

INSTANTIATE_TEST_CASE_P(StringsFindTest,
                        FindParmsTest,
                        testing::ValuesIn(std::array<cudf::size_type, 4>{0, 1, 2, 3}));
