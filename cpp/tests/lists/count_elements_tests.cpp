/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/lists/count_elements.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

struct ListsElementsTest : public cudf::test::BaseFixture {
};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

template <typename T>
class ListsElementsNumericsTest : public ListsElementsTest {
};

TYPED_TEST_CASE(ListsElementsNumericsTest, NumericTypesNotBool);

TYPED_TEST(ListsElementsNumericsTest, CountElements)
{
  auto validity = thrust::make_transform_iterator(
    thrust::make_counting_iterator<cudf::size_type>(0), [](auto i) { return i != 1; });
  using LCW = cudf::test::lists_column_wrapper<TypeParam>;
  LCW input({LCW{3, 2, 1}, LCW{}, LCW{30, 20, 10, 50}, LCW{100, 120}, LCW{0}}, validity);

  auto result = cudf::lists::count_elements(cudf::lists_column_view(input));
  cudf::test::fixed_width_column_wrapper<int32_t> expected({3, 0, 4, 2, 1}, {1, 0, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(ListsElementsTest, CountElementsStrings)
{
  auto validity = thrust::make_transform_iterator(
    thrust::make_counting_iterator<cudf::size_type>(0), [](auto i) { return i != 1; });
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW input(
    {LCW{"", "Héllo", "thesé"}, LCW{}, LCW{"are", "some", "", "z"}, LCW{"tést", "String"}, LCW{""}},
    validity);

  auto result = cudf::lists::count_elements(cudf::lists_column_view(input));
  cudf::test::fixed_width_column_wrapper<int32_t> expected({3, 0, 4, 2, 1}, {1, 0, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(ListsElementsTest, CountElementsSliced)
{
  auto validity = thrust::make_transform_iterator(
    thrust::make_counting_iterator<cudf::size_type>(0), [](auto i) { return i != 1; });
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW input(
    {LCW{"", "Héllo", "thesé"}, LCW{}, LCW{"are", "some", "", "z"}, LCW{"tést", "String"}, LCW{""}},
    validity);

  auto sliced = cudf::slice(input, {1, 4}).front();
  auto result = cudf::lists::count_elements(cudf::lists_column_view(sliced));
  cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 4, 2}, {0, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TYPED_TEST(ListsElementsNumericsTest, CountElementsNestedLists)
{
  std::vector<int32_t> validity{1, 0, 1, 1};
  using LCW = cudf::test::lists_column_wrapper<TypeParam>;
  LCW list({LCW{LCW{2, 3}, LCW{4, 5}},
            LCW{LCW{}},
            LCW{LCW{6, 7, 8}, LCW{9, 10, 11}, LCW({12, 13, 14}, validity.begin())},
            LCW{LCW{15, 16}, LCW{17, 18}, LCW{19, 20}, LCW{21, 22}, LCW{23, 24}}},
           validity.begin());

  auto result = cudf::lists::count_elements(cudf::lists_column_view(list));
  cudf::test::fixed_width_column_wrapper<int32_t> expected({2, 1, 3, 5}, {1, 0, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(ListsElementsTest, CountElementsEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  LCW empty{};
  auto result = cudf::lists::count_elements(cudf::lists_column_view(empty));
  EXPECT_EQ(0, result->size());
}
