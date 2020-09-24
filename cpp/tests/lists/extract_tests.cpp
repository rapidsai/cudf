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
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <vector>

struct ListsExtractTest : public cudf::test::BaseFixture {
};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

template <typename T>
class ListsExtractNumericsTest : public ListsExtractTest {
};

TYPED_TEST_CASE(ListsExtractNumericsTest, NumericTypesNotBool);

TYPED_TEST(ListsExtractNumericsTest, ExtractElement)
{
  auto validity = thrust::make_transform_iterator(
    thrust::make_counting_iterator<cudf::size_type>(0), [](auto i) { return i != 1; });
  using LCW = cudf::test::lists_column_wrapper<TypeParam>;
  LCW input({LCW{3, 2, 1}, LCW{}, LCW{30, 20, 10, 50}, LCW{100, 120}, LCW{0}}, validity);

  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 0);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({3, 0, 30, 100, 0}, {1, 0, 1, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 1);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({2, 0, 20, 120, 0}, {1, 0, 1, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 2);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({1, 0, 10, 0, 0}, {1, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 3);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({0, 0, 50, 0, 0}, {0, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 4);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({0, 0, 0, 0, 0}, {0, 0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -1);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({1, 0, 50, 120, 0}, {1, 0, 1, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -2);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({2, 0, 10, 100, 0}, {1, 0, 1, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -3);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({3, 0, 20, 0, 0}, {1, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -4);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({0, 0, 30, 0, 0}, {0, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -5);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({0, 0, 0, 0, 0}, {0, 0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
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
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 1);
    cudf::test::strings_column_wrapper expected({"Héllo", "", "some", "String", ""},
                                                {1, 0, 1, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 2);
    cudf::test::strings_column_wrapper expected({"thesé", "", "", "", ""}, {1, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 3);
    cudf::test::strings_column_wrapper expected({"", "", "z", "", ""}, {0, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 4);
    cudf::test::strings_column_wrapper expected({"", "", "", "", ""}, {0, 0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -1);
    cudf::test::strings_column_wrapper expected({"thesé", "", "z", "String", ""}, {1, 0, 1, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -2);
    cudf::test::strings_column_wrapper expected({"Héllo", "", "", "tést", ""}, {1, 0, 1, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -3);
    cudf::test::strings_column_wrapper expected({"", "", "some", "", ""}, {1, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -4);
    cudf::test::strings_column_wrapper expected({"", "", "are", "", ""}, {0, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -5);
    cudf::test::strings_column_wrapper expected({"", "", "", "", ""}, {0, 0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

TYPED_TEST(ListsExtractNumericsTest, ExtractElementNestedLists)
{
  std::vector<int32_t> validity{1, 0, 1, 1};
  using LCW = cudf::test::lists_column_wrapper<TypeParam>;
  LCW list({LCW{LCW{2, 3}, LCW{4, 5}},
            LCW{LCW{}},
            LCW{LCW{6, 7, 8}, LCW{9, 10, 11}, LCW{12, 13, 14}},
            LCW{LCW{15, 16}, LCW{17, 18}, LCW{19, 20}, LCW{21, 22}, LCW{23, 24}}},
           validity.begin());

  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), 0);
    LCW expected({LCW{2, 3}, LCW{}, LCW{6, 7, 8}, LCW{15, 16}}, validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), 1);
    LCW expected({LCW{4, 5}, LCW{}, LCW{9, 10, 11}, LCW{17, 18}}, validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), 2);
    std::vector<int32_t> expected_validity{0, 0, 1, 1};
    LCW expected({LCW{}, LCW{}, LCW{12, 13, 14}, LCW{19, 20}}, expected_validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), 3);
    std::vector<int32_t> expected_validity{0, 0, 0, 1};
    LCW expected({LCW{}, LCW{}, LCW{}, LCW{21, 22}}, expected_validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), -1);
    LCW expected({LCW{4, 5}, LCW{}, LCW{12, 13, 14}, LCW{23, 24}}, validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), -2);
    LCW expected({LCW{2, 3}, LCW{}, LCW{9, 10, 11}, LCW{21, 22}}, validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), -3);
    std::vector<int32_t> expected_validity{0, 0, 1, 1};
    LCW expected({LCW{}, LCW{}, LCW{6, 7, 8}, LCW{19, 20}}, expected_validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

TEST_F(ListsExtractTest, ExtractElementEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  LCW empty{};
  auto result = cudf::lists::extract_list_element(cudf::lists_column_view(empty), 1);
  EXPECT_EQ(0, result->size());

  LCW empty_strings({LCW{"", "", ""}});
  result = cudf::lists::extract_list_element(cudf::lists_column_view(empty_strings), 1);
  cudf::test::strings_column_wrapper expected({""});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);

  LCW null_strings({LCW{"", "", ""}}, thrust::make_constant_iterator<int32_t>(0));
  result = cudf::lists::extract_list_element(cudf::lists_column_view(null_strings), 1);
  cudf::test::strings_column_wrapper expected_null({""}, {0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_null, *result);
}

TEST_F(ListsExtractTest, ExtractElementWithNulls)
{
  auto validity = thrust::make_transform_iterator(
    thrust::make_counting_iterator<cudf::size_type>(0), [](auto i) { return i != 1; });
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW input{
    {{"Héllo", "", "thesé"}, validity}, {"are"}, {{"some", ""}, validity}, {"tést", "strings"}};

  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 0);
    cudf::test::strings_column_wrapper expected({"Héllo", "are", "some", "tést"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 1);
    cudf::test::strings_column_wrapper expected({"", "", "", "strings"}, {0, 0, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -1);
    cudf::test::strings_column_wrapper expected({"thesé", "are", "", "strings"}, {1, 1, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

CUDF_TEST_PROGRAM_MAIN()
