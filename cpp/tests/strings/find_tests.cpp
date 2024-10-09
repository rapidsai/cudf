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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/iterator/transform_iterator.h>

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

TEST_F(StringsFindTest, FindSuperLongStrings)
{
  cudf::test::strings_column_wrapper input(
    {"Lorem ipsum odor amet, consectetuer adipiscing elit. Tortor at egestas tellus nec turpis blandit dapibus tellus. Quam proin ut in ac mi lorem dictum. Penatibus sapien aptent, vel finibus ut commodo. Semper at semper vivamus pellentesque dignissim placerat metus. Nostra non nascetur hac; ligula quis quisque."
    "Neque tortor quam est iaculis facilisi facilisi pulvinar porttitor euismod. Conubia varius dis convallis pretium sit! Commodo accumsan tincidunt curae torquent nibh conubia. Volutpat augue elit id tortor libero ut posuere. Mattis pretium congue velit hendrerit metus. Diam nisl potenti varius imperdiet cursus dapibus. Augue turpis ex erat suspendisse per ullamcorper. Montes maximus massa volutpat eu vel tincidunt. Mauris suscipit augue habitasse hac quisque cursus mi. Porta cubilia cubilia gravida fusce blandit lacinia ipsum duis imperdiet.",
    "Aenean habitasse in quis ipsum id tellus. Consectetur justo senectus tortor nullam volutpat in. Pretium elementum class posuere malesuada ac scelerisque. Ut porta sociosqu elit posuere; egestas tincidunt urna. Viverra aptent nisi cras eros nam cursus proin. Porta ultrices feugiat non ullamcorper imperdiet cras. Magnis lorem id quis; consectetur viverra quisque."
    "Scelerisque elementum euismod libero tortor lectus; dictum eleifend aenean. Maecenas semper primis aliquet auctor nec suscipit nulla aptent. Ligula est donec sagittis risus eu facilisi ornare. Faucibus nunc lorem porttitor condimentum auctor et mauris vulputate. Imperdiet arcu imperdiet proin mattis rutrum ante at. Ultrices in dui suspendisse primis adipiscing. Malesuada fringilla tempor tempor sagittis volutpat enim. Turpis fringilla morbi justo posuere pretium magna risus. Vivamus potenti dolor ridiculus orci tortor tempus. Purus lacus nunc turpis, phasellus amet porttitor eleifend.",
    "Aliquet nec etiam; ligula montes platea tortor elit morbi pellentesque! Lacinia interdum et tincidunt cras accumsan aenean velit. Massa vitae adipiscing neque aptent pharetra nisl sociosqu porttitor. Viverra eu aenean torquent rhoncus quam placerat; himenaeos inceptos. Inceptos placerat montes malesuada dis ultrices a netus. Mattis eros mi duis ante class erat diam. Massa class egestas tempor lobortis ante. Hendrerit mauris blandit facilisi habitasse conubia duis vehicula quam. Vehicula orci massa urna blandit per. Sociosqu placerat nunc nostra felis, lacus felis habitant."
    "Phasellus eleifend penatibus ullamcorper tempus euismod himenaeos turpis porta. Amet per ridiculus vehicula tellus habitant. Tempor massa molestie venenatis sodales volutpat a phasellus massa. Ullamcorper placerat ultrices sociosqu ultricies habitant feugiat ipsum. Convallis tempus vehicula congue mollis varius auctor magna justo. Praesent curabitur vehicula torquent venenatis mauris metus.",
    "Est rhoncus egestas mollis inceptos nunc et. Aliquet turpis a nisl at senectus varius quis. Nec taciti semper blandit laoreet turpis praesent aptent at. Ut porta potenti nec vitae consectetur? Vel etiam arcu eleifend accumsan posuere est. Parturient ad porta ad leo, interdum maecenas nascetur lectus. Mauris lacus dolor ultrices semper; a ullamcorper eleifend tellus. Lobortis elit quis conubia sollicitudin ut integer convallis. Nisi eleifend parturient conubia nec eget vel lobortis ligula."
    "Hendrerit sed amet quis praesent parturient sollicitudin. Nisi neque praesent ultrices scelerisque lacus class et malesuada. Proin nam facilisi habitasse lectus lobortis erat dui. Lacinia dapibus molestie consectetur nostra a aliquet odio. Et ad in mauris penatibus tempus venenatis porta. Dui id non morbi auctor augue hendrerit vehicula. Arcu condimentum ultrices fermentum, posuere ipsum morbi hendrerit auctor tortor.",
    "Curabitur eget posuere sapien. Nullam suscipit elit urna, in volutpat ex pharetra a. Vivamus convallis scelerisque felis, ut commodo enim scelerisque nec. In et leo commodo, ullamcorper orci eu, scelerisque quam. Integer imperdiet fringilla magna a convallis. Phasellus pharetra lorem elit, tristique fermentum sem interdum at. Ut nec laoreet neque. Nullam vulputate metus nec tristique vestibulum. Morbi ullamcorper varius mi, et tincidunt elit vulputate nec. Etiam interdum accumsan nibh nec scelerisque. Fusce sagittis interdum placerat. Ut neque diam, venenatis aliquet tempor a, egestas blandit ex. Etiam condimentum dolor scelerisque finibus tempor. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Nunc vel risus ullamcorper, interdum erat eget, luctus ante. Aenean nec mauris facilisis, sodales tortor sit amet, pulvinar neque."
  });

  auto strings_view = cudf::strings_column_view(input);
  {
    // Check the find result
    auto results = cudf::strings::find(strings_view, cudf::string_scalar("ipsum"));
    auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({6, 25, 838, 878, 672});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }

  {
    // Check the contains result
    auto results = cudf::strings::contains(strings_view, cudf::string_scalar("adipiscing"));
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
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
