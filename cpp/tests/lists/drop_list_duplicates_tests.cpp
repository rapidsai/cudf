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

#include <cudf/lists/drop_list_duplicates.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

using float_type = float;
using int_type   = int32_t;
using INT_LCW    = cudf::test::lists_column_wrapper<int_type>;
using FLT_LCW    = cudf::test::lists_column_wrapper<float_type>;
using STR_LCW    = cudf::test::lists_column_wrapper<cudf::string_view>;

template <bool equal_test, class LCW>
void test_once(cudf::column_view const& input,
               LCW const& expected,
               cudf::null_equality nulls_equal = cudf::null_equality::EQUAL)
{
  auto const results =
    cudf::lists::drop_list_duplicates(cudf::lists_column_view{input}, nulls_equal);
  if (equal_test) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, true);
  } else {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected, true);
  }
}

struct DropListDuplicatesTest : public cudf::test::BaseFixture {
};

TEST_F(DropListDuplicatesTest, InvalidCasesTests)
{
  // Lists of nested types are not supported
  EXPECT_THROW(
    cudf::lists::drop_list_duplicates(cudf::lists_column_view{INT_LCW{INT_LCW{{1, 2}, {3}}}}),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::lists::drop_list_duplicates(cudf::lists_column_view{FLT_LCW{FLT_LCW{{1, 2}, {3}}}}),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::lists::drop_list_duplicates(cudf::lists_column_view{STR_LCW{STR_LCW{STR_LCW{"string"}}}}),
    cudf::logic_error);
}

TEST_F(DropListDuplicatesTest, FloatingPointTestsNonNull)
{
  // Trivial cases
  test_once<false>(FLT_LCW{{}}, FLT_LCW{{}});
  test_once<false>(FLT_LCW{{0, 1, 2, 3, 4, 5}, {}}, FLT_LCW{{0, 1, 2, 3, 4, 5}, {}});

  // Multiple empty lists
  test_once<false>(FLT_LCW{{}, {}, {5, 4, 3, 2, 1, 0}, {}, {6}, {}},
                   FLT_LCW{{}, {}, {0, 1, 2, 3, 4, 5}, {}, {6}, {}});

  auto constexpr p_inf = std::numeric_limits<float_type>::infinity();
  auto constexpr m_inf = -std::numeric_limits<float_type>::infinity();

  // Lists contain inf
  // We can't test for lists containing nan because the order of nan is
  // undefined after sorting
  test_once<false>(FLT_LCW{0, 1, 2, 0, 1, 2, 0, 1, 2, p_inf, p_inf, p_inf},
                   FLT_LCW{0, 1, 2, p_inf});
  test_once<false>(FLT_LCW{p_inf, 0, m_inf, 0, p_inf, 0, m_inf, 0, p_inf, 0, m_inf},
                   FLT_LCW{m_inf, 0, p_inf});
}

TEST_F(DropListDuplicatesTest, IntegerTestsNonNull)
{
  // Trivial cases
  test_once<true>(INT_LCW{{}}, INT_LCW{{}});
  test_once<true>(INT_LCW{{0, 1, 2, 3, 4, 5}, {}}, INT_LCW{{0, 1, 2, 3, 4, 5}, {}});

  // Multiple empty lists
  test_once<true>(INT_LCW{{}, {}, {5, 4, 3, 2, 1, 0}, {}, {6}, {}},
                  INT_LCW{{}, {}, {0, 1, 2, 3, 4, 5}, {}, {6}, {}});

  // Adjacent lists containing the same entries
  test_once<true>(
    INT_LCW{{1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 2, 2, 2}, {2, 2, 2, 2, 3, 3, 3, 3}},
    INT_LCW{{1}, {1, 2}, {2, 3}});

  // Sliced list column
  auto const list0 = INT_LCW{{1, 2, 3, 2, 3, 2, 3, 2, 3}, {3, 2, 1, 4, 1}, {5}, {10, 8, 9}, {6, 7}};
  auto const list1 = cudf::slice(list0, {0, 5})[0];
  auto const list2 = cudf::slice(list0, {1, 5})[0];
  auto const list3 = cudf::slice(list0, {1, 3})[0];
  auto const list4 = cudf::slice(list0, {0, 3})[0];

  test_once<true>(list0, INT_LCW{{1, 2, 3}, {1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}});
  test_once<true>(list1, INT_LCW{{1, 2, 3}, {1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}});
  test_once<true>(list2, INT_LCW{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}});
  test_once<true>(list3, INT_LCW{{1, 2, 3, 4}, {5}});
  test_once<true>(list4, INT_LCW{{1, 2, 3}, {1, 2, 3, 4}, {5}});
}

TEST_F(DropListDuplicatesTest, IntegerTestsWithNulls)
{
  auto constexpr null = std::numeric_limits<int_type>::max();

  // null lists
  test_once<true>(INT_LCW{{{3, 2, 1, 4, 1}, {5}, {}, {}, {10, 8, 9}, {6, 7}},
                          cudf::detail::make_counting_transform_iterator(
                            0, [](auto i) { return i != 2 && i != 3; })},
                  INT_LCW{{{1, 2, 3, 4}, {5}, {}, {}, {8, 9, 10}, {6, 7}},
                          cudf::detail::make_counting_transform_iterator(
                            0, [](auto i) { return i != 2 && i != 3; })});

  // null entries are equal
  test_once<true>(
    INT_LCW{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
            cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; })},
    INT_LCW{{1, 3, 5, 7, 9, null},
            std::initializer_list<bool>{true, true, true, true, true, false}});

  // nulls entries are not equal
  test_once<true>(
    INT_LCW{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
            cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; })},
    INT_LCW{
      {1, 3, 5, 7, 9, null, null, null, null, null},
      std::initializer_list<bool>{true, true, true, true, true, false, false, false, false, false}},
    cudf::null_equality::UNEQUAL);
}

TEST_F(DropListDuplicatesTest, StringTestsNonNull)
{
  // Trivial cases
  test_once<true>(STR_LCW{{}}, STR_LCW{{}});
  test_once<true>(STR_LCW{"this", "is", "a", "string"}, STR_LCW{"a", "is", "string", "this"});

  // One list column
  test_once<true>(STR_LCW{"this", "is", "is", "is", "a", "string", "string"},
                  STR_LCW{"a", "is", "string", "this"});

  // Multiple lists column
  test_once<true>(
    STR_LCW{STR_LCW{"this", "is", "a", "no duplicate", "string"},
            STR_LCW{"this", "is", "is", "a", "one duplicate", "string"},
            STR_LCW{"this", "is", "is", "is", "a", "two duplicates", "string"},
            STR_LCW{"this", "is", "is", "is", "is", "a", "three duplicates", "string"}},
    STR_LCW{STR_LCW{"a", "is", "no duplicate", "string", "this"},
            STR_LCW{"a", "is", "one duplicate", "string", "this"},
            STR_LCW{"a", "is", "string", "this", "two duplicates"},
            STR_LCW{"a", "is", "string", "this", "three duplicates"}});
}

TEST_F(DropListDuplicatesTest, StringTestsWithNulls)
{
  auto const null = std::string("");

  // One list column with null entries
  test_once<true>(
    STR_LCW{{"this", null, "is", "is", "is", "a", null, "string", null, "string"},
            cudf::detail::make_counting_transform_iterator(
              0, [](auto i) { return i != 1 && i != 6 && i != 8; })},
    STR_LCW{{"a", "is", "string", "this", null},
            cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; })});

  // Multiple lists column with null lists and null entries
  test_once<true>(
    STR_LCW{{STR_LCW{{"this", null, "is", null, "a", null, "no duplicate", null, "string"},
                     cudf::detail::make_counting_transform_iterator(
                       0, [](auto i) { return i % 2 == 0; })},
             STR_LCW{},
             STR_LCW{"this", "is", "is", "a", "one duplicate", "string"}},
            cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })},
    STR_LCW{
      {STR_LCW{{"a", "is", "no duplicate", "string", "this", null},
               cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i <= 4; })},
       STR_LCW{},
       STR_LCW{"a", "is", "one duplicate", "string", "this"}},
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; })});
}
