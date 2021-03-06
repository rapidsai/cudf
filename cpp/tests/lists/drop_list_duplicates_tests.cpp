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

template <bool equal_test, class LCW>
void test_once(cudf::column_view const& input,
               LCW const& expected,
               cudf::null_equality nulls_equal = cudf::null_equality::EQUAL)
{
  auto const results =
    cudf::lists::drop_list_duplicates(cudf::lists_column_view{input}, nulls_equal);
  if (equal_test)
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected, true);
  else
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected, true);
}

struct DropListDuplicatesTest : public cudf::test::BaseFixture {
};

TEST_F(DropListDuplicatesTest, InvalidCasesTests)
{
  using ILCW = cudf::test::lists_column_wrapper<int32_t>;
  using FLCW = cudf::test::lists_column_wrapper<float>;
  using SLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  /* Lists of nested types are not supported */
  EXPECT_THROW(cudf::lists::drop_list_duplicates(cudf::lists_column_view{ILCW{ILCW{{1, 2}, {3}}}}),
               cudf::logic_error);
  EXPECT_THROW(cudf::lists::drop_list_duplicates(cudf::lists_column_view{FLCW{FLCW{{1, 2}, {3}}}}),
               cudf::logic_error);
  EXPECT_THROW(
    cudf::lists::drop_list_duplicates(cudf::lists_column_view{SLCW{SLCW{SLCW{"dummy string"}}}}),
    cudf::logic_error);
}

TEST_F(DropListDuplicatesTest, FloatingPointTestsNonNull)
{
  using float_type = double;
  using LCW        = cudf::test::lists_column_wrapper<float_type>;

  /* Trivial cases */
  test_once<false>(LCW{{}}, LCW{{}});
  test_once<false>(LCW{{0, 1, 2, 3, 4, 5}, {}}, LCW{{0, 1, 2, 3, 4, 5}, {}});

  /* Multiple empty lists */
  test_once<false>(LCW{{}, {}, {5, 4, 3, 2, 1, 0}, {}, {6}, {}},
                   LCW{{}, {}, {0, 1, 2, 3, 4, 5}, {}, {6}, {}});

  /* Lists contain inf/nan */
  auto const inf = std::numeric_limits<float_type>::infinity();
  auto const nan = std::numeric_limits<float_type>::quiet_NaN();
  test_once<false>(LCW{0, 1, 2, 0, 1, 2, 0, 1, 2, inf, inf, inf}, LCW{0, 1, 2, inf});
  test_once<false>(LCW{0, 1, 2, 0, 1, 2, 0, 1, 2, nan, nan, nan}, LCW{0, 1, 2, nan});
  test_once<false>(LCW{nan, nan, nan, 0, 1, 2, 0, 1, 2, 0, 1, 2, inf, inf, inf},
                   LCW{0, 1, 2, nan, inf});
  test_once<false>(LCW{nan, inf, nan, inf, nan, inf}, LCW{nan, inf});
}

TEST_F(DropListDuplicatesTest, IntegerTestsNonNull)
{
  using LCW = cudf::test::lists_column_wrapper<int32_t>;

  /* Trivial cases */
  test_once<true>(LCW{{}}, LCW{{}});
  test_once<true>(LCW{{0, 1, 2, 3, 4, 5}, {}}, LCW{{0, 1, 2, 3, 4, 5}, {}});

  /* Multiple empty lists */
  test_once<true>(LCW{{}, {}, {5, 4, 3, 2, 1, 0}, {}, {6}, {}},
                  LCW{{}, {}, {0, 1, 2, 3, 4, 5}, {}, {6}, {}});

  /* Sliced list column */
  auto const list0 = LCW{{1, 2, 3, 2, 3, 2, 3, 2, 3}, {3, 2, 1, 4, 1}, {5}, {10, 8, 9}, {6, 7}};
  auto const list1 = cudf::slice(list0, {1, 5})[0];
  test_once<true>(list0, LCW{{1, 2, 3}, {1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}});
  test_once<true>(list1, LCW{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}});
}

TEST_F(DropListDuplicatesTest, IntegerTestsWithNulls)
{
  using LCW       = cudf::test::lists_column_wrapper<int32_t>;
  auto const null = std::numeric_limits<int32_t>::max();

  /* null lists */
  test_once<true>(
    LCW{{{3, 2, 1, 4, 1}, {5}, {}, {}, {10, 8, 9}, {6, 7}},
        cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 2 && i != 3; })},
    LCW{
      {{1, 2, 3, 4}, {5}, {}, {}, {8, 9, 10}, {6, 7}},
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 2 && i != 3; })});

  /* null entries are equal */
  test_once<true>(
    LCW{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; })},
    LCW{{1, 3, 5, 7, 9, null}, std::initializer_list<bool>{true, true, true, true, true, false}});

  /* nulls entries are not equal */
  test_once<true>(
    LCW{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; })},
    LCW{
      {1, 3, 5, 7, 9, null, null, null, null, null},
      std::initializer_list<bool>{true, true, true, true, true, false, false, false, false, false}},
    cudf::null_equality::UNEQUAL);
}

TEST_F(DropListDuplicatesTest, StringTestsNonNull)
{
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  /* Trivial cases */
  test_once<true>(LCW{{}}, LCW{{}});
  test_once<true>(LCW{"this", "is", "a", "string"}, LCW{"a", "is", "string", "this"});

  /* One list column */
  test_once<true>(LCW{"this", "is", "is", "is", "a", "string", "string"},
                  LCW{"a", "is", "string", "this"});

  /* Multiple lists column */
  test_once<true>(LCW{LCW{"this", "is", "a", "no duplicate", "string"},
                      LCW{"this", "is", "is", "a", "one duplicate", "string"},
                      LCW{"this", "is", "is", "is", "a", "two duplicates", "string"},
                      LCW{"this", "is", "is", "is", "is", "a", "three duplicates", "string"}},
                  LCW{LCW{"a", "is", "no duplicate", "string", "this"},
                      LCW{"a", "is", "one duplicate", "string", "this"},
                      LCW{"a", "is", "string", "this", "two duplicates"},
                      LCW{"a", "is", "string", "this", "three duplicates"}});
}
