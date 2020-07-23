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

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/extract.hpp>

#include <tests/strings/utilities.h>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>

#include <vector>

struct ListsExtractTest : public cudf::test::BaseFixture {
};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

template <typename T>
class NumericsTest : public ListsExtractTest {
};

TYPED_TEST_CASE(NumericsTest, NumericTypesNotBool);

TYPED_TEST(NumericsTest, ExtractElement)
{
  auto validity = thrust::make_transform_iterator(
    thrust::make_counting_iterator<cudf::size_type>(0), [](auto i) { return i != 1; });
  using LCW = cudf::test::lists_column_wrapper<TypeParam>;
  LCW input({LCW{3, 2, 1}, LCW{}, LCW{30, 20, 10, 50}, LCW{100, 120}, LCW{0}}, validity);

  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 0);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({3, 0, 30, 100, 0}, {1, 0, 1, 1, 1});
    cudf::test::expect_columns_equal(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 1);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({2, 0, 20, 120, 0}, {1, 0, 1, 1, 0});
    cudf::test::expect_columns_equal(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 2);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({1, 0, 10, 0, 0}, {1, 0, 1, 0, 0});
    cudf::test::expect_columns_equal(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 3);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({0, 0, 50, 0, 0}, {0, 0, 1, 0, 0});
    cudf::test::expect_columns_equal(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 4);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({0, 0, 0, 0, 0}, {0, 0, 0, 0, 0});
    cudf::test::expect_columns_equal(expected, *result);
  }
}

TEST_F(ListsExtractTest, ExtractElementStrings)
{
  auto validity = thrust::make_transform_iterator(
    thrust::make_counting_iterator<cudf::size_type>(0), [](auto i) { return i != 1; });
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW input(
    {LCW{"", "Héllo", "thesé"}, LCW{}, LCW{"are", "some", "", "z"}, LCW{"tést", "String"}, LCW{""}},
    validity);

  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 0);
    cudf::test::strings_column_wrapper expected({"", "", "are", "tést", ""}, {1, 0, 1, 1, 1});
    cudf::test::expect_columns_equal(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 1);
    cudf::test::strings_column_wrapper expected({"Héllo", "", "some", "String", ""},
                                                {1, 0, 1, 1, 0});
    cudf::test::expect_columns_equal(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 2);
    cudf::test::strings_column_wrapper expected({"thesé", "", "", "", ""}, {1, 0, 1, 0, 0});
    cudf::test::expect_columns_equal(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 3);
    cudf::test::strings_column_wrapper expected({"", "", "z", "", ""}, {0, 0, 1, 0, 0});
    cudf::test::expect_columns_equal(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 4);
    cudf::test::strings_column_wrapper expected({"", "", "", "", ""}, {0, 0, 0, 0, 0});
    cudf::test::expect_columns_equal(expected, *result);
  }
}

CUDF_TEST_PROGRAM_MAIN()
