/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/reverse.hpp>
#include <cudf/null_mask.hpp>

using namespace cudf::test::iterators;

auto constexpr null{0};

using ints_lists = cudf::test::lists_column_wrapper<int32_t>;
using ints_col   = cudf::test::fixed_width_column_wrapper<int32_t>;

struct ListsReverseTest : public cudf::test::BaseFixture {
};

template <typename T>
struct ListsReverseTypedTest : public cudf::test::BaseFixture {
};

using TestTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(ListsReverseTypedTest, TestTypes);

TEST_F(ListsReverseTest, EmptyInput)
{
  // Empty input.
  {
    auto const input   = ints_lists{};
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, *results);
  }

  // Empty lists.
  {
    auto const input   = ints_lists{ints_lists{}, ints_lists{}, ints_lists{}};
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, *results);
  }

  // Empty nested lists.
  {
    auto const input   = ints_lists{ints_lists{ints_lists{}}, ints_lists{}, ints_lists{}};
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, *results);
  }
}

TYPED_TEST(ListsReverseTypedTest, InputNoNulls)
{
  using lists_col           = cudf::test::lists_column_wrapper<TypeParam>;
  auto const input_original = lists_col{{}, {1, 2, 3}, {}, {4, 5}, {6, 7, 8}, {9}};

  {
    auto const expected = lists_col{{}, {3, 2, 1}, {}, {5, 4}, {8, 7, 6}, {9}};
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input_original));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const input    = cudf::slice(input_original, {1, 4})[0];
    auto const expected = lists_col{{3, 2, 1}, {}, {5, 4}};
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const input    = cudf::slice(input_original, {2, 3})[0];
    auto const expected = lists_col{lists_col{}};
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const input    = cudf::slice(input_original, {2, 4})[0];
    auto const expected = lists_col{{}, {5, 4}};
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }
}

TYPED_TEST(ListsReverseTypedTest, InputWithNulls)
{
  using lists_col           = cudf::test::lists_column_wrapper<TypeParam>;
  auto const input_original = lists_col{{lists_col{},
                                         lists_col{1, 2, 3},
                                         lists_col{} /*null*/,
                                         lists_col{{4, 5, null}, null_at(2)},
                                         lists_col{6, 7, 8},
                                         lists_col{9}},
                                        null_at(2)};

  {
    auto const expected = lists_col{{lists_col{},
                                     lists_col{3, 2, 1},
                                     lists_col{} /*null*/,
                                     lists_col{{null, 5, 4}, null_at(0)},
                                     lists_col{8, 7, 6},
                                     lists_col{9}},
                                    null_at(2)};
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input_original));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const input    = cudf::slice(input_original, {1, 4})[0];
    auto const expected = lists_col{
      {lists_col{3, 2, 1}, lists_col{} /*null*/, lists_col{{null, 5, 4}, null_at(0)}}, null_at(1)};
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const input    = cudf::slice(input_original, {2, 3})[0];
    auto const expected = lists_col{{lists_col{} /*null*/}, null_at(0)};
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const input = cudf::slice(input_original, {2, 4})[0];
    auto const expected =
      lists_col{{lists_col{} /*null*/, lists_col{{null, 5, 4}, null_at(0)}}, null_at(0)};
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const input    = cudf::slice(input_original, {4, 6})[0];
    auto const expected = lists_col{{8, 7, 6}, {9}};
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input));
    // The result doesn't have nulls, but it is nullable.
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results);
  }
}

TYPED_TEST(ListsReverseTypedTest, InputListsOfListsNoNulls)
{
  using lists_col           = cudf::test::lists_column_wrapper<TypeParam>;
  auto const input_original = [] {
    auto child =
      lists_col{{1, 2, 3}, {4, 5, 6}, {7}, {4, 5}, {}, {4, 5, 6}, {}, {6, 7, 8}, {}, {9}}.release();
    auto offsets = ints_col{0, 0, 3, 3, 6, 9, 10, 10, 10}.release();
    return cudf::make_lists_column(8, std::move(offsets), std::move(child), 0, {});
  }();

  {
    auto const expected = [] {
      auto child =
        lists_col{{7}, {4, 5, 6}, {1, 2, 3}, {4, 5, 6}, {}, {4, 5}, {}, {6, 7, 8}, {}, {9}}
          .release();
      auto offsets = ints_col{0, 0, 3, 3, 6, 9, 10, 10, 10}.release();
      return cudf::make_lists_column(8, std::move(offsets), std::move(child), 0, {});
    }();
    auto const results = cudf::lists::reverse(cudf::lists_column_view(*input_original));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results);
  }

  {
    auto const input    = cudf::slice(*input_original, {1, 4})[0];
    auto const expected = [] {
      auto child   = lists_col{{7}, {4, 5, 6}, {1, 2, 3}, {4, 5, 6}, {}, {4, 5}}.release();
      auto offsets = ints_col{0, 3, 3, 6}.release();
      return cudf::make_lists_column(3, std::move(offsets), std::move(child), 0, {});
    }();
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results);
  }

  {
    auto const input   = cudf::slice(*input_original, {2, 3})[0];  // empty column
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, *results);
  }

  {
    auto const input    = cudf::slice(*input_original, {2, 4})[0];
    auto const expected = [] {
      auto child   = lists_col{{4, 5, 6}, {}, {4, 5}}.release();
      auto offsets = ints_col{0, 0, 3}.release();
      return cudf::make_lists_column(2, std::move(offsets), std::move(child), 0, {});
    }();
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results);
  }
}

TYPED_TEST(ListsReverseTypedTest, InputListsOfListsWithNulls)
{
  using lists_col           = cudf::test::lists_column_wrapper<TypeParam>;
  auto const input_original = [] {
    auto child = lists_col{{{1, 2, 3},
                            {4, 5, 6},
                            {7},
                            {4, 5},
                            {} /*null*/,
                            {4, 5, 6},
                            {},
                            {6, 7, 8},
                            {} /*null*/,
                            {9}},
                           nulls_at({4, 8})}
                   .release();
    auto offsets   = ints_col{0, 0, 3, 3, 6, 9, 10, 10, 10}.release();
    auto null_mask = cudf::create_null_mask(8, cudf::mask_state::ALL_VALID);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 2, 3, false);

    return cudf::make_lists_column(
      8, std::move(offsets), std::move(child), 1, std::move(null_mask));
  }();

  {
    auto const expected = [] {
      auto child = lists_col{{{7},
                              {4, 5, 6},
                              {1, 2, 3},
                              {4, 5, 6},
                              {} /*null*/,
                              {4, 5},
                              {} /*null*/,
                              {6, 7, 8},
                              {},
                              {9}},
                             nulls_at({4, 6})}
                     .release();
      auto offsets   = ints_col{0, 0, 3, 3, 6, 9, 10, 10, 10}.release();
      auto null_mask = cudf::create_null_mask(8, cudf::mask_state::ALL_VALID);
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 2, 3, false);

      return cudf::make_lists_column(
        8, std::move(offsets), std::move(child), 1, std::move(null_mask));
    }();
    auto const results = cudf::lists::reverse(cudf::lists_column_view(*input_original));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results);
  }

  {
    auto const input    = cudf::slice(*input_original, {0, 1})[0];
    auto const expected = [] {
      auto child   = lists_col{{7}, {4, 5, 6}, {1, 2, 3}}.release();
      auto offsets = ints_col{0, 3}.release();
      return cudf::make_lists_column(1, std::move(offsets), std::move(child), 0, {});
    }();
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, *results);
  }

  {
    auto const input    = cudf::slice(*input_original, {1, 4})[0];
    auto const expected = [] {
      auto child =
        lists_col{{{7}, {4, 5, 6}, {1, 2, 3}, {4, 5, 6}, {} /*null*/, {4, 5}}, null_at(4)}
          .release();
      auto offsets   = ints_col{0, 3, 3, 6}.release();
      auto null_mask = cudf::create_null_mask(3, cudf::mask_state::ALL_VALID);
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 1, 2, false);

      return cudf::make_lists_column(
        3, std::move(offsets), std::move(child), 1, std::move(null_mask));
    }();
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results);
  }

  {
    auto const input    = cudf::slice(*input_original, {2, 4})[0];
    auto const expected = [] {
      auto child     = lists_col{{{4, 5, 6}, {} /*null*/, {4, 5}}, null_at(1)}.release();
      auto offsets   = ints_col{0, 0, 3}.release();
      auto null_mask = cudf::create_null_mask(2, cudf::mask_state::ALL_VALID);
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 0, 1, false);

      return cudf::make_lists_column(
        2, std::move(offsets), std::move(child), 1, std::move(null_mask));
    }();
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    // The result doesn't have nulls, but it is nullable.
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results);
  }
}
