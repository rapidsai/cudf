/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/fill.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct StringsFillTest : public cudf::test::BaseFixture {
};

TEST_F(StringsFillTest, Fill)
{
  std::vector<const char*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view view(strings);
  {
    auto results = cudf::strings::detail::fill(view, 1, 5, cudf::string_scalar("zz"));

    std::vector<const char*> h_expected{"eee", "zz", "zz", "zz", "zz", "bbb", "ééé"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results = cudf::strings::detail::fill(view, 2, 4, cudf::string_scalar("", false));

    std::vector<const char*> h_expected{"eee", "bb", nullptr, nullptr, "aa", "bbb", "ééé"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results = cudf::strings::detail::fill(view, 5, 5, cudf::string_scalar("zz"));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, view.parent());
  }
  {
    auto results = cudf::strings::detail::fill(view, 0, 7, cudf::string_scalar(""));
    cudf::test::strings_column_wrapper expected({"", "", "", "", "", "", ""},
                                                {1, 1, 1, 1, 1, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results = cudf::strings::detail::fill(view, 0, 7, cudf::string_scalar("", false));
    cudf::test::strings_column_wrapper expected({"", "", "", "", "", "", ""},
                                                {0, 0, 0, 0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
}

TEST_F(StringsFillTest, ZeroSizeStringsColumns)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto results = cudf::strings::detail::fill(
    cudf::strings_column_view(zero_size_strings_column), 0, 1, cudf::string_scalar(""));
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsFillTest, FillRangeError)
{
  std::vector<const char*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view view(strings);

  EXPECT_THROW(cudf::strings::detail::fill(view, 5, 1, cudf::string_scalar("")), cudf::logic_error);
  EXPECT_THROW(cudf::strings::detail::fill(view, 5, 9, cudf::string_scalar("")), cudf::logic_error);
}
