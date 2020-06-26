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
#include <cudf/strings/padding.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/wrap.hpp>
#include <cudf/utilities/error.hpp>

#include <tests/strings/utilities.h>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <vector>

struct StringsPadTest : public cudf::test::BaseFixture {
};

TEST_F(StringsPadTest, Padding)
{
  std::vector<const char*> h_strings{"eee ddd", "bb cc", nullptr, "", "aa", "bbb", "ééé", "o"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::size_type width = 6;
  std::string phil      = "+";
  auto strings_view     = cudf::strings_column_view(strings);

  {
    auto results = cudf::strings::pad(strings_view, width, cudf::strings::pad_side::RIGHT, phil);

    std::vector<const char*> h_expected{
      "eee ddd", "bb cc+", nullptr, "++++++", "aa++++", "bbb+++", "ééé+++", "o+++++"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    auto results = cudf::strings::pad(strings_view, width, cudf::strings::pad_side::LEFT, phil);

    std::vector<const char*> h_expected{
      "eee ddd", "+bb cc", nullptr, "++++++", "++++aa", "+++bbb", "+++ééé", "+++++o"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    auto results = cudf::strings::pad(strings_view, width, cudf::strings::pad_side::BOTH, phil);

    std::vector<const char*> h_expected{
      "eee ddd", "bb cc+", nullptr, "++++++", "++aa++", "+bbb++", "+ééé++", "++o+++"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    cudf::test::expect_columns_equal(*results, expected);
  }
}

TEST_F(StringsPadTest, PaddingBoth)
{
  cudf::test::strings_column_wrapper strings({"koala", "foxx", "fox", "chameleon"});
  std::string phil  = "+";
  auto strings_view = cudf::strings_column_view(strings);

  {  // even width left justify
    auto results = cudf::strings::pad(strings_view, 6, cudf::strings::pad_side::BOTH, phil);
    cudf::test::strings_column_wrapper expected({"koala+", "+foxx+", "+fox++", "chameleon"});
    cudf::test::expect_columns_equal(*results, expected);
  }
  {  // odd width right justify
    auto results = cudf::strings::pad(strings_view, 7, cudf::strings::pad_side::BOTH, phil);
    cudf::test::strings_column_wrapper expected({"+koala+", "++foxx+", "++fox++", "chameleon"});
    cudf::test::expect_columns_equal(*results, expected);
  }
}

TEST_F(StringsPadTest, ZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  auto results      = cudf::strings::pad(strings_view, 5);
  cudf::test::expect_strings_empty(results->view());
}

class StringsPadParmsTest : public StringsPadTest,
                            public testing::WithParamInterface<cudf::size_type> {
};

TEST_P(StringsPadParmsTest, Padding)
{
  std::vector<std::string> h_strings{"eee ddd", "bb cc", "aa", "bbb", "fff", "", "o"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  cudf::size_type width = GetParam();
  auto strings_view     = cudf::strings_column_view(strings);
  auto results          = cudf::strings::pad(strings_view, width, cudf::strings::pad_side::RIGHT);

  std::vector<std::string> h_expected;
  for (auto itr = h_strings.begin(); itr != h_strings.end(); ++itr) {
    std::string str      = *itr;
    cudf::size_type size = str.size();
    if (size < width) str.insert(size, width - size, ' ');
    h_expected.push_back(str);
  }
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end());

  cudf::test::expect_columns_equal(*results, expected);
}

INSTANTIATE_TEST_CASE_P(StringsPadParmWidthTest,
                        StringsPadParmsTest,
                        testing::ValuesIn(std::array<cudf::size_type, 3>{5, 6, 7}));

TEST_F(StringsPadTest, ZFill)
{
  std::vector<const char*> h_strings{
    "654321", "-12345", nullptr, "", "-5", "0987", "4", "+8.5", "éé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::size_type width = 6;
  std::string phil      = "+";
  auto strings_view     = cudf::strings_column_view(strings);

  auto results = cudf::strings::zfill(strings_view, width);

  std::vector<const char*> h_expected{
    "654321", "-12345", nullptr, "000000", "0000-5", "000987", "000004", "00+8.5", "0000éé"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsPadTest, Wrap1)
{
  std::vector<const char*> h_strings{"12345", "thesé", nullptr, "ARE THE", "tést strings", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::size_type width = 3;

  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::wrap(strings_view, width);

  std::vector<const char*> h_expected{"12345", "thesé", nullptr, "ARE\nTHE", "tést\nstrings", ""};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsPadTest, Wrap2)
{
  std::vector<const char*> h_strings{"the quick brown fox jumped over the lazy brown dog",
                                     "hello, world"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::size_type width = 12;

  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::wrap(strings_view, width);

  std::vector<const char*> h_expected{"the quick\nbrown fox\njumped over\nthe lazy\nbrown dog",
                                      "hello, world"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsPadTest, WrapExpectFailure)
{
  std::vector<const char*> h_strings{"12345", "thesé", nullptr, "ARE THE", "tést strings", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::size_type width = 0;  // this should trigger failure

  auto strings_view = cudf::strings_column_view(strings);

  EXPECT_THROW(cudf::strings::wrap(strings_view, width), cudf::logic_error);
}
