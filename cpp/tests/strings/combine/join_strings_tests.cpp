/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <tests/strings/utilities.h>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <thrust/iterator/transform_iterator.h>

struct JoinStringsTest : public cudf::test::BaseFixture {
};

TEST_F(JoinStringsTest, Join)
{
  std::vector<const char*> h_strings{"eee", "bb", nullptr, "zzzz", "", "aaa", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto view1 = cudf::strings_column_view(strings);

  {
    auto results = cudf::strings::join_strings(view1);

    cudf::test::strings_column_wrapper expected{"eeebbzzzzaaaééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::join_strings(view1, cudf::string_scalar("+"));

    cudf::test::strings_column_wrapper expected{"eee+bb+zzzz++aaa+ééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results =
      cudf::strings::join_strings(view1, cudf::string_scalar("+"), cudf::string_scalar("___"));

    cudf::test::strings_column_wrapper expected{"eee+bb+___+zzzz++aaa+ééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(JoinStringsTest, JoinZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  auto results      = cudf::strings::join_strings(strings_view);
  cudf::test::expect_strings_empty(results->view());
}

TEST_F(JoinStringsTest, JoinAllNullStringsColumn)
{
  cudf::test::strings_column_wrapper strings({"", "", ""}, {0, 0, 0});

  auto results = cudf::strings::join_strings(cudf::strings_column_view(strings));
  cudf::test::strings_column_wrapper expected1({""}, {0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected1);

  results = cudf::strings::join_strings(
    cudf::strings_column_view(strings), cudf::string_scalar(""), cudf::string_scalar("3"));
  cudf::test::strings_column_wrapper expected2({"333"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected2);

  results = cudf::strings::join_strings(
    cudf::strings_column_view(strings), cudf::string_scalar("-"), cudf::string_scalar("*"));
  cudf::test::strings_column_wrapper expected3({"*-*-*"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected3);
}
