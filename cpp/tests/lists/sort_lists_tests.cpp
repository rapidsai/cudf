/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/null_mask.hpp>

#include <cuda/std/limits>

#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace cudf::test::iterators;

template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

auto generate_sorted_lists(cudf::lists_column_view const& input,
                           cudf::order column_order,
                           cudf::null_order null_precedence)
{
  return std::pair{cudf::lists::sort_lists(input, column_order, null_precedence),
                   cudf::lists::stable_sort_lists(input, column_order, null_precedence)};
}

template <typename T>
struct SortLists : public cudf::test::BaseFixture {};

using TypesForTest = cudf::test::Concat<cudf::test::NumericTypes, cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(SortLists, TypesForTest);

TYPED_TEST(SortLists, NoNull)
{
  using T = TypeParam;

  // List<T>
  LCW<T> list{{3, 2, 1, 4}, {5}, {10, 8, 9}, {6, 7}};

  // Ascending
  // LCW<int>  order{{2, 1, 0, 3}, {0}, {1, 2, 0},  {0, 1}};
  LCW<T> expected{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  LCW<T> expected2{{4, 3, 2, 1}, {5}, {10, 9, 8}, {7, 6}};
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected2);
  }
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected2);
  }
}

TYPED_TEST(SortLists, Null)
{
  using T = TypeParam;
  if (std::is_same_v<T, bool>) return;
  std::vector<bool> valids_o{true, true, false, true};
  std::vector<bool> valids_a{true, true, true, false};
  std::vector<bool> valids_b{false, true, true, true};

  // List<T>
  LCW<T> list{{{3, 2, 4, 1}, valids_o.begin()}, {5}, {10, 8, 9}, {6, 7}};
  // LCW<int>  order{{2, 1, 3, 0}, {0}, {1, 2, 0},  {0, 1}};

  {
    LCW<T> expected{{{1, 2, 3, 4}, valids_a.begin()}, {5}, {8, 9, 10}, {6, 7}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    LCW<T> expected{{{4, 1, 2, 3}, valids_b.begin()}, {5}, {8, 9, 10}, {6, 7}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  {
    LCW<T> expected{{{4, 3, 2, 1}, valids_b.begin()}, {5}, {10, 9, 8}, {7, 6}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    LCW<T> expected{{{3, 2, 1, 4}, valids_a.begin()}, {5}, {10, 9, 8}, {7, 6}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

using SortListsInt = SortLists<int>;

TEST_F(SortListsInt, Empty)
{
  using T = int;

  {
    LCW<T> l{};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{LCW<T>{}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{LCW<T>{}, LCW<T>{}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
}

TEST_F(SortListsInt, Single)
{
  using T = int;

  {
    LCW<T> l{1};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{{1, 2, 3}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
}

TEST_F(SortListsInt, NullRows)
{
  using T = int;
  std::vector<int> valids{0, 1, 0};
  LCW<T> l{{{1, 2, 3}, {4, 5, 6}, {7}}, valids.begin()};  // offset 0, 0, 3, 3

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
}

// A nullable but null-free input must come back non-nullable: the sort cannot add nulls, so
// neither the returned parent nor its child may keep an allocated all-valid mask.
TEST_F(SortListsInt, NullableWithZeroNullsDropsMasks)
{
  using T = int;
  cudf::test::fixed_width_column_wrapper<T> child({3, 1, 2, 5, 4}, no_nulls());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, 3, 5};
  auto const input =
    cudf::make_lists_column(2,
                            offsets.release(),
                            child.release(),
                            0,
                            cudf::create_null_mask(2, cudf::mask_state::ALL_VALID));
  ASSERT_TRUE(input->view().nullable());
  ASSERT_TRUE(cudf::lists_column_view{input->view()}.child().nullable());

  LCW<T> expected{{1, 2, 3}, {4, 5}};
  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input->view()}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  EXPECT_FALSE(sorted_lists->view().nullable());
  EXPECT_FALSE(cudf::lists_column_view{sorted_lists->view()}.child().nullable());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  EXPECT_FALSE(stable_sorted_lists->view().nullable());
  EXPECT_FALSE(cudf::lists_column_view{stable_sorted_lists->view()}.child().nullable());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

using SortListsString = SortLists<cudf::string_view>;

// Default order: element nulls sort last within a row; a null list row passes through unchanged.
TEST_F(SortListsString, Strings)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  std::vector<bool> const valids{true, true, false, true};
  StrLCW input{{StrLCW{{"pear", "apple", "fig", "kiwi"}, null_at(2)},
                StrLCW{"banana"},
                StrLCW{"unused"},
                StrLCW{"melon", "cherry"}},
               valids.begin()};

  // Null slots keep arbitrary placeholder values ("fig" here); they are never compared.
  StrLCW expected{{StrLCW{{"apple", "kiwi", "pear", "fig"}, null_at(3)},
                   StrLCW{"banana"},
                   StrLCW{"unused"},
                   StrLCW{"cherry", "melon"}},
                  valids.begin()};

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{input}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// The Prefix* tests target the string fast path: a packed leading-bytes key, iterative byte-window
// tie-breaks, then a final comparison cleanup. Each asserts the fast path (sort_lists) and the
// comparison path (stable_sort_lists) against the same expected column.

// Identical first eight bytes force the full-string tie-break; the proper prefix sorts first.
TEST_F(SortListsString, PrefixEqualFirstEightBytes)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"abcdefgh1", "abcdefgh0", "abcdefgh"}};
  StrLCW expected{{"abcdefgh", "abcdefgh0", "abcdefgh1"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Under eight bytes, zero-padding lets the packed key alone order a proper prefix first.
TEST_F(SortListsString, PrefixSubstringShorterFirst)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"aaa", "aa", "a"}};
  StrLCW expected{{"a", "aa", "aaa"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Empty strings pack an all-zero key and must sort before any non-empty string.
TEST_F(SortListsString, PrefixEmptyStringsFirst)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"banana", "", "apple", "", "fig"}};
  StrLCW expected{{"", "", "apple", "banana", "fig"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Null slots' placeholder values are never compared.
TEST_F(SortListsString, PrefixMultipleNullsLast)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{StrLCW{{"pear", "apple", "fig", "kiwi", "lime"}, nulls_at({1, 3})}};
  StrLCW expected{StrLCW{{"fig", "lime", "pear", "apple", "kiwi"}, nulls_at({3, 4})}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

TEST_F(SortListsString, PrefixAllNullListRow)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  std::vector<bool> const valids{true, false, true};
  StrLCW input{{StrLCW{"cherry", "apple", "banana"}, StrLCW{"unused"}, StrLCW{"date", "egg"}},
               valids.begin()};
  StrLCW expected{{StrLCW{"apple", "banana", "cherry"}, StrLCW{"unused"}, StrLCW{"date", "egg"}},
                  valids.begin()};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// One shared eight-byte prefix makes the whole row a single tie run resolved wholly by full-string
// comparison -- the tie-break's worst case.
TEST_F(SortListsString, PrefixWholeRowShared)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"prefix__zebra", "prefix__apple", "prefix__mango", "prefix__"}};
  StrLCW expected{{"prefix__", "prefix__apple", "prefix__mango", "prefix__zebra"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Raw unsigned-byte order: the two-/three-/four-byte UTF-8 lead bytes (0xC3 < 0xE2 < 0xF0) sort
// after ASCII and by lead-byte class among themselves.
TEST_F(SortListsString, PrefixUtf8Multibyte)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{
    {"\xE2\x82\xAC\x75\x72\x6F", "zoo", "\xF0\x9F\x98\x80", "\xC3\xA9\x70\xC3\xA9\x65", "apple"}};
  StrLCW expected{
    {"apple", "zoo", "\xC3\xA9\x70\xC3\xA9\x65", "\xE2\x82\xAC\x75\x72\x6F", "\xF0\x9F\x98\x80"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

// Per-row prefix ties must resolve without crossing row boundaries (the segment_id key field); the
// empty row checks dense segment-ordinal labeling skips a zero-length segment without misaligning
// its neighbors.
TEST_F(SortListsString, PrefixMultipleSegments)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{StrLCW{"commonpx_b", "commonpx_a", "commonpx_c"},
               StrLCW{},
               StrLCW{"melon", "apple", "cherry"},
               StrLCW{"sharedpx2", "sharedpx0", "sharedpx1"}};
  StrLCW expected{StrLCW{"commonpx_a", "commonpx_b", "commonpx_c"},
                  StrLCW{},
                  StrLCW{"apple", "cherry", "melon"},
                  StrLCW{"sharedpx0", "sharedpx1", "sharedpx2"}};

  auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
    cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
}

namespace {
// Asserts `sort_lists` (fast path) and `stable_sort_lists` (comparison path) both equal
// `expected`, so every expectation is also re-checked against the comparison semantics.
void expect_both_sort_paths_match(cudf::lists_column_view const& input,
                                  cudf::column_view const& expected,
                                  cudf::order column_order         = cudf::order::ASCENDING,
                                  cudf::null_order null_precedence = cudf::null_order::AFTER)
{
  auto const sorted = cudf::lists::sort_lists(input, column_order, null_precedence);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view(), expected);
  auto const stable_sorted = cudf::lists::stable_sort_lists(input, column_order, null_precedence);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted->view(), expected);
}

// Wraps a leaf column as a one-row LIST, so the sort orders within a single segment.
std::unique_ptr<cudf::column> as_single_row_list(std::unique_ptr<cudf::column> leaf)
{
  auto const n = static_cast<cudf::size_type>(leaf->size());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
  return cudf::make_lists_column(1, offsets.release(), std::move(leaf), 0, {});
}
}  // namespace

// All shorter than the packed key, so the key alone orders them -- including the second-byte
// distinction "aaa" < "ab".
TEST_F(SortListsString, PrefixShortStrings)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{{"aaa", "aa", "a", "ab", "b"}};
  StrLCW expected{{"a", "aa", "aaa", "ab", "b"}};
  expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
}

// Shared leading bytes make the packed key uninformative, so the tie-break does the real ordering.
TEST_F(SortListsString, PrefixSharedPrefixWithNullAndDuplicate)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  StrLCW input{
    StrLCW{{"AAAAAAAAzebra", "AAAAAAAAapple", "AAAAAAAAmango", "unused", "AAAAAAAAapple"},
           null_at(3)},
    StrLCW{"food_pear", "food_kiwi", "food_pear"}};
  StrLCW expected{
    StrLCW{{"AAAAAAAAapple", "AAAAAAAAapple", "AAAAAAAAmango", "AAAAAAAAzebra", "unused"},
           null_at(4)},
    StrLCW{"food_kiwi", "food_pear", "food_pear"}};
  expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
}

// A naive all-0xFF null sentinel would collide with genuine all-0xFF strings; the packed key's
// dedicated is_null bit must separate them. The 72- and 73-byte 0xFF strings stay tied through
// every byte window and reach the comparison cleanup as one run.
TEST_F(SortListsString, PrefixNullSentinelCollisionFF)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::string const ff72(72, '\xff');
  std::string const ff73(73, '\xff');

  std::vector<std::string> const in_strings{"apple", ff73, "mango", ff72, ""};
  std::vector<bool> const in_valids{true, true, true, true, false};

  std::vector<std::string> const ex_strings{"apple", "mango", ff72, ff73, ""};
  std::vector<bool> const ex_valids{true, true, true, true, false};

  auto const make_single_row_list = [](std::vector<std::string> const& strs,
                                       std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(in_strings, in_valids);
  auto const expected = make_single_row_list(ex_strings, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// The shared prefix outlasts every iterative window, so the whole run reaches the tie-break,
// exceeds TIE_HEAPSORT_THRESHOLD, and takes the heapsort branch; > 64 elements keep the column on
// the prefix path. Adds interleaved nulls to the plain PrefixTrueHeapsortRun shape.
TEST_F(SortListsString, PrefixSharedPrefixHeapsortRun)
{
  std::string const prefix(80, 'A');  // 80 > 7 + 8*8 = 71, so ties survive every iterative window.
  std::vector<std::string> non_null;
  for (int i = 0; i < 70; ++i) {
    non_null.push_back(prefix + std::to_string(i / 10) + std::to_string(i % 10));
  }
  non_null.push_back(prefix + "05");  // exact duplicates
  non_null.push_back(prefix + "17");

  auto shuffled = non_null;
  std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937{0xC0'FFEE});

  auto const null_pos_a = shuffled.size() / 3;
  auto const null_pos_b = (shuffled.size() * 2) / 3 + 1;
  std::vector<std::string> in_strings;
  std::vector<bool> in_valids;
  for (std::size_t i = 0; i <= shuffled.size(); ++i) {
    if (i == null_pos_a || i == null_pos_b) {
      in_strings.emplace_back("");  // null placeholder; never compared
      in_valids.push_back(false);
    }
    if (i < shuffled.size()) {
      in_strings.push_back(shuffled[i]);
      in_valids.push_back(true);
    }
  }

  auto sorted = non_null;
  std::sort(sorted.begin(), sorted.end());
  std::vector<std::string> ex_strings = sorted;
  std::vector<bool> ex_valids(sorted.size(), true);
  ex_strings.emplace_back("");
  ex_valids.push_back(false);
  ex_strings.emplace_back("");
  ex_valids.push_back(false);

  using StrCW                     = cudf::test::strings_column_wrapper;
  auto const make_single_row_list = [](std::vector<std::string> const& strs,
                                       std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(in_strings, in_valids);
  auto const expected = make_single_row_list(ex_strings, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Deep shared prefixes force several windows past the packed key. Row 0 adds a pure length tie (a
// string vs itself plus a trailing NUL) that only the cleanup's shorter-is-less rule separates;
// row 1's thousands of shared-prefix strings exercise the pass cap and the parallel re-sort.
TEST_F(SortListsString, PrefixIterativeDeepLongSharedPrefix)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::string const prefix0(20, 'A');  // Twenty shared bytes: > 2x the 8-byte key, 5x the 4-byte.
  std::vector<std::string> row0_non_null{
    prefix0 + "zebra",
    prefix0 + "ant",  // shorter tail -> mixed run lengths within the shared run
    prefix0 + "mango",
    prefix0 + "ant",                 // exact duplicate
    prefix0,                         // the shared prefix with no tail
    prefix0 + std::string(1, '\0'),  // `prefix0` zero-extended: a pure length tie
    prefix0 + "mango",               // exact duplicate
    ""};                             // all-zero key, must sort first in the row

  std::string const prefix1(24, 'Q');  // Twenty-four shared bytes -> three windows past an 8B key.
  std::vector<std::string> row1;
  row1.reserve(5'000);
  for (int i = 0; i < 5'000; ++i) {
    auto tail = std::to_string(i);
    tail.insert(tail.begin(), 4 - tail.size(), '0');
    row1.push_back(prefix1 + tail);
  }

  std::shuffle(row1.begin(), row1.end(), std::mt19937{0x5EED});

  std::vector<std::string> in_strings;
  std::vector<bool> in_valids;
  for (auto const& s : row0_non_null) {
    in_strings.push_back(s);
    in_valids.push_back(true);
  }
  in_strings.emplace_back("");  // null placeholder; never compared
  in_valids.push_back(false);
  auto const row0_len = static_cast<cudf::size_type>(in_strings.size());
  for (auto const& s : row1) {
    in_strings.push_back(s);
    in_valids.push_back(true);
  }
  auto const total_len = static_cast<cudf::size_type>(in_strings.size());

  // std::string's raw-byte order and length tie-break match cudf's string order.
  auto row0_sorted = row0_non_null;
  std::sort(row0_sorted.begin(), row0_sorted.end());
  auto row1_sorted = row1;
  std::sort(row1_sorted.begin(), row1_sorted.end());

  std::vector<std::string> ex_strings;
  std::vector<bool> ex_valids;
  for (auto const& s : row0_sorted) {
    ex_strings.push_back(s);
    ex_valids.push_back(true);
  }
  ex_strings.emplace_back("");
  ex_valids.push_back(false);
  for (auto const& s : row1_sorted) {
    ex_strings.push_back(s);
    ex_valids.push_back(true);
  }

  auto const make_two_row_list = [&](std::vector<std::string> const& strs,
                                     std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, row0_len, total_len};
    return cudf::make_lists_column(2, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_two_row_list(in_strings, in_valids);
  auto const expected = make_two_row_list(ex_strings, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// A run of byte-identical strings becomes length-uniform and exhausted once the windows cover its
// length, so the drop freezes it early instead of dragging it to the pass cap and cleanup.
TEST_F(SortListsString, PrefixExhaustedIdenticalRunDropped)
{
  using StrCW = cudf::test::strings_column_wrapper;
  std::vector<std::string> in;
  for (int i = 0; i < 40; ++i) {
    in.emplace_back("abcdefghijkl");  // a byte-identical run
  }
  in.emplace_back("abcdefghijkZ");  // singleton: differs only in the last byte
  in.emplace_back("abcdefgh");      // shorter proper prefix: its own run, sorts first

  auto expected = in;
  std::sort(expected.begin(), expected.end());

  auto const input        = as_single_row_list(StrCW{in.begin(), in.end()}.release());
  auto const expected_col = as_single_row_list(StrCW{expected.begin(), expected.end()}.release());
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected_col->view());
}

// A string and its zero-extension collide in every window, so the stable radix keeps arrival
// order. Placing the LONGER first makes that order wrong lexicographically; an exhaustion-only
// drop would freeze it, so the length-uniform guard must leave the run for the shorter-first
// cleanup.
TEST_F(SortListsString, PrefixZeroExtensionLongerFirstNotDropped)
{
  using StrCW = cudf::test::strings_column_wrapper;
  std::string const base(20, 'A');
  std::string const shorter = base;
  std::string const longer  = base + std::string(1, '\0');

  // base+"B" / base+"C" keep the run non-trivial; they resolve as singletons.
  std::vector<std::string> in{longer, shorter, base + "B", base + "C"};
  auto expected = in;
  std::sort(expected.begin(), expected.end());  // shorter < longer < base+"B" < base+"C".

  auto const input        = as_single_row_list(StrCW{in.begin(), in.end()}.release());
  auto const expected_col = as_single_row_list(StrCW{expected.begin(), expected.end()}.release());
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected_col->view());
}

// The singleton compaction must extract only the deep row's still-tied elements, re-sort just
// those, and scatter them back without disturbing the first-pass-resolved rows around it.
TEST_F(SortListsString, PrefixCompactionMixedSingletonAndDeepRows)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::vector<std::vector<std::string>> rows;
  // Every element differs in its first byte, so these rows resolve in the first pass.
  for (int r = 0; r < 12; ++r) {
    rows.push_back({std::string(1, static_cast<char>('a' + r)) + "_zzz",
                    std::string(1, static_cast<char>('A' + r)) + "_aaa",
                    std::string(1, static_cast<char>('0' + r)) + "_mmm"});
  }
  // The only row the compaction should re-sort: a 32-byte shared prefix, four windows past the key.
  std::string const deep_prefix(32, 'D');
  std::vector<std::string> deep;
  deep.reserve(1'500);
  for (int i = 0; i < 1'500; ++i) {
    auto tail = std::to_string(i);
    tail.insert(tail.begin(), 4 - tail.size(), '0');
    deep.push_back(deep_prefix + tail);
  }
  std::shuffle(deep.begin(), deep.end(), std::mt19937{0xD00D});
  // Place the deep row in the middle so resolved rows sit on both sides of the compacted block.
  rows.insert(rows.begin() + 6, deep);

  std::vector<std::string> in_strings;
  std::vector<std::string> ex_strings;
  std::vector<cudf::size_type> offsets{0};
  for (auto const& row : rows) {
    for (auto const& s : row) {
      in_strings.push_back(s);
    }
    auto sorted = row;
    std::sort(sorted.begin(), sorted.end());
    for (auto const& s : sorted) {
      ex_strings.push_back(s);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_strings.size()));
  }

  auto const num_rows  = static_cast<cudf::size_type>(rows.size());
  auto const make_list = [&](std::vector<std::string> const& strs) {
    StrCW leaf{strs.begin(), strs.end()};
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };

  auto const input    = make_list(in_strings);
  auto const expected = make_list(ex_strings);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// With one segment (S = bit_width(1) = 1) and nulls, the packed key holds P = 64 - 1 - 1 = 62
// prefix bits -- not a byte multiple -- so it proves 7 whole bytes plus byte 7's top six bits.
// The probe strings differ only in byte 7's low two bits and therefore tie on the key; the
// tie-break must resume its windows at byte 7 (a stale fixed eight-byte offset would skip past
// both eight-byte strings and leave them arbitrarily ordered).
TEST_F(SortListsString, PrefixPackedKeyPartialByteTieOffset)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::string const probe_lo = std::string("AAAAAA") + '\0' + '\x01';  // 8 bytes, byte 7 = 0x01.
  std::string const probe_hi = std::string("AAAAAA") + '\0' + '\x02';  // 8 bytes, byte 7 = 0x02.

  // The "B" fillers resolve in the first pass and never join the probe pair's run; the two nulls
  // force the key's null bit (P = 62, not 63).
  std::vector<std::string> in_strings;
  std::vector<bool> in_valids;
  in_strings.push_back(probe_hi);  // out of order, so the tie-break must reorder the pair
  in_valids.push_back(true);
  in_strings.push_back(probe_lo);
  in_valids.push_back(true);
  in_strings.emplace_back("");  // null placeholder; never compared
  in_valids.push_back(false);
  for (int i = 0; in_strings.size() < 199; ++i) {
    auto tail = std::to_string(i);
    tail.insert(tail.begin(), 4 - tail.size(), '0');
    in_strings.push_back("B" + tail);
    in_valids.push_back(true);
  }
  in_strings.emplace_back("");  // second null placeholder
  in_valids.push_back(false);

  auto const num_elements = static_cast<cudf::size_type>(in_strings.size());

  std::vector<std::string> non_null;
  for (std::size_t i = 0; i < in_strings.size(); ++i) {
    if (in_valids[i]) { non_null.push_back(in_strings[i]); }
  }
  std::sort(non_null.begin(), non_null.end());
  std::vector<std::string> ex_strings = non_null;
  std::vector<bool> ex_valids(non_null.size(), true);
  ex_strings.emplace_back("");
  ex_valids.push_back(false);
  ex_strings.emplace_back("");
  ex_valids.push_back(false);

  auto const make_single_row_list = [](std::vector<std::string> const& strs,
                                       std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(in_strings, in_valids);
  auto const expected = make_single_row_list(ex_strings, ex_valids);
  // The count only keeps the probe pair amid a bulk of first-pass singletons; one row fixes S = 1.
  EXPECT_EQ(num_elements, 200);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Nulls are position-final after the first pass and excluded from the tie-break set; adjacent
// nulls beside genuine tie runs verify the exclusion never misplaces them.
TEST_F(SortListsString, PrefixNullRunsNeverTied)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::vector<std::string> const in_strings{
    "AAAAAAAA3", "", "", "", "AAAAAAAA1", "AAAAAAAA2", "zebra", "apple", "apple"};
  std::vector<bool> const in_valids{true, false, false, false, true, true, true, true, true};

  // 'A' (0x41) < 'a' (0x61) < 'z' (0x7A) by raw byte; the three nulls collect last.
  std::vector<std::string> const ex_strings{
    "AAAAAAAA1", "AAAAAAAA2", "AAAAAAAA3", "apple", "apple", "zebra", "", "", ""};
  std::vector<bool> const ex_valids{true, true, true, true, true, true, false, false, false};

  auto const make_single_row_list = [](std::vector<std::string> const& strs,
                                       std::vector<bool> const& valids) {
    StrCW leaf{strs.begin(), strs.end(), valids.begin()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(in_strings, in_valids);
  auto const expected = make_single_row_list(ex_strings, ex_valids);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// A whole segment of exact duplicates collapses to one segment-spanning tie run beside ordinary
// rows.
TEST_F(SortListsString, PrefixWholeSegmentDuplicate)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::vector<std::string> const dup_row(10, "duplicate");
  std::vector<std::vector<std::string>> const rows{
    {"cherry", "apple", "banana"}, dup_row, {"melon", "date", "fig"}};

  std::vector<std::string> in_strings;
  std::vector<std::string> ex_strings;
  std::vector<cudf::size_type> offsets{0};
  for (auto const& row : rows) {
    for (auto const& s : row) {
      in_strings.push_back(s);
    }
    auto sorted = row;
    std::sort(sorted.begin(), sorted.end());
    for (auto const& s : sorted) {
      ex_strings.push_back(s);
    }
    offsets.push_back(static_cast<cudf::size_type>(in_strings.size()));
  }

  auto const num_rows  = static_cast<cudf::size_type>(rows.size());
  auto const make_list = [&](std::vector<std::string> const& strs) {
    StrCW leaf{strs.begin(), strs.end()};
    cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
    return cudf::make_lists_column(num_rows, off.release(), leaf.release(), 0, {});
  };

  auto const input    = make_list(in_strings);
  auto const expected = make_list(ex_strings);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// The comparison cleanup's heapsort branch: the first pass proves 7 whole bytes (P = 63) and the
// 8 windows of 8 bytes reach byte 7 + 64 = 71, so an 80-byte shared prefix survives every window
// and the run (>= TIE_HEAPSORT_THRESHOLD) is resolved only by the cleanup.
TEST_F(SortListsString, PrefixTrueHeapsortRun)
{
  using StrCW = cudf::test::strings_column_wrapper;

  std::string const prefix(80, 'Z');  // 80 > 7 + 8*8 = 71, so ties survive every iterative window.
  std::vector<std::string> non_null;
  // > 64 elements rejects the graduated-warp path, keeping the column on the prefix path.
  for (int i = 0; i < 70; ++i) {
    non_null.push_back(prefix + std::to_string(i / 10) + std::to_string(i % 10));
  }
  non_null.push_back(prefix + "05");  // exact duplicates
  non_null.push_back(prefix + "17");

  auto shuffled = non_null;
  std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937{0x8EA7});

  auto sorted = non_null;
  std::sort(sorted.begin(), sorted.end());

  auto const make_single_row_list = [](std::vector<std::string> const& strs) {
    StrCW leaf{strs.begin(), strs.end()};
    auto const n = static_cast<cudf::size_type>(strs.size());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, n};
    return cudf::make_lists_column(1, offsets.release(), leaf.release(), 0, {});
  };

  auto const input    = make_single_row_list(shuffled);
  auto const expected = make_single_row_list(sorted);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
}

// Sliced string input with a retained null: the packed-key builder, window tie-break, and
// null handling must all honor the nonzero leaf offset.
TEST_F(SortListsString, PrefixSlicedWithNulls)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  StrLCW l{StrLCW{"zz", "aa"},
           StrLCW{{"banana", "apple", "x", "cherry"}, null_at(2)},
           StrLCW{"b", "a"},
           StrLCW{"x"}};
  auto const sliced_list = cudf::slice(l, {1, 4})[0];  // drops row 0 -> nonzero child offset
  StrLCW expected{
    StrLCW{{"apple", "banana", "cherry", "x"}, null_at(3)}, StrLCW{"a", "b"}, StrLCW{"x"}};
  expect_both_sort_paths_match(cudf::lists_column_view{sliced_list}, expected);
}

TEST_F(SortListsInt, NestedListElement)
{
  using T = int;
  // Column of LIST<LIST<int>>: each row's inner lists are reordered as whole elements. The third
  // row's inner lists tie on their first element, so ordering falls through to the second.
  LCW<T> input{LCW<T>{{3, 1}, {2, 0}}, LCW<T>{{5, 5}, {4, 9}}, LCW<T>{{1, 3}, {1, 2}}};
  {
    // Ascending.
    LCW<T> expected{LCW<T>{{2, 0}, {3, 1}}, LCW<T>{{4, 9}, {5, 5}}, LCW<T>{{1, 2}, {1, 3}}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
  {
    // Descending reverses each row's ascending order.
    LCW<T> expected{LCW<T>{{3, 1}, {2, 0}}, LCW<T>{{5, 5}, {4, 9}}, LCW<T>{{1, 3}, {1, 2}}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{input}, cudf::order::DESCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

// E = LIST<STRUCT<int, int>>: a list-of-struct element type sorts; struct ranks are computed
// internally before the lexicographic comparison.
TEST_F(SortListsInt, ListOfStructElement)
{
  // One row with two elements [{3, 30}] and [{1, 10}]; ascending reorders to [{1, 10}], [{3, 30}].
  cudf::test::fixed_width_column_wrapper<int> in_f0{3, 1};
  cudf::test::fixed_width_column_wrapper<int> in_f1{30, 10};
  cudf::test::structs_column_wrapper in_structs{{in_f0, in_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> in_inner_off{0, 1, 2};
  auto in_inner = cudf::make_lists_column(2, in_inner_off.release(), in_structs.release(), 0, {});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> in_outer_off{0, 2};
  auto in_outer = cudf::make_lists_column(1, in_outer_off.release(), std::move(in_inner), 0, {});

  cudf::test::fixed_width_column_wrapper<int> ex_f0{1, 3};
  cudf::test::fixed_width_column_wrapper<int> ex_f1{10, 30};
  cudf::test::structs_column_wrapper ex_structs{{ex_f0, ex_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ex_inner_off{0, 1, 2};
  auto ex_inner = cudf::make_lists_column(2, ex_inner_off.release(), ex_structs.release(), 0, {});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ex_outer_off{0, 2};
  auto ex_outer = cudf::make_lists_column(1, ex_outer_off.release(), std::move(ex_inner), 0, {});

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{in_outer->view()}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), ex_outer->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), ex_outer->view());
}

// E = STRUCT<int, LIST<int>>: a struct-with-list-field element type sorts.
TEST_F(SortListsInt, StructOfListElement)
{
  // One row with two struct elements {2, [9, 0]} and {1, [8, 7]}; ascending reorders them to
  // {1, [8, 7]}, {2, [9, 0]}.
  cudf::test::fixed_width_column_wrapper<int> in_f0{2, 1};
  cudf::test::lists_column_wrapper<int, int32_t> in_f1{{9, 0}, {8, 7}};
  cudf::test::structs_column_wrapper in_structs{{in_f0, in_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> in_off{0, 2};
  auto in_list = cudf::make_lists_column(1, in_off.release(), in_structs.release(), 0, {});

  cudf::test::fixed_width_column_wrapper<int> ex_f0{1, 2};
  cudf::test::lists_column_wrapper<int, int32_t> ex_f1{{8, 7}, {9, 0}};
  cudf::test::structs_column_wrapper ex_structs{{ex_f0, ex_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ex_off{0, 2};
  auto ex_list = cudf::make_lists_column(1, ex_off.release(), ex_structs.release(), 0, {});

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{in_list->view()}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), ex_list->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), ex_list->view());
}

TEST_F(SortListsInt, Sliced)
{
  using T = int;
  LCW<T> l{{3, 2, 1, 4}, {7, 5, 6}, {8, 9}, {10}};

  {
    auto const sliced_list = cudf::slice(l, {0, 4})[0];
    auto const expected    = LCW<T>{{1, 2, 3, 4}, {5, 6, 7}, {8, 9}, {10}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {1, 4})[0];
    auto const expected    = LCW<T>{{5, 6, 7}, {8, 9}, {10}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {1, 2})[0];
    auto const expected    = LCW<T>{{5, 6, 7}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {0, 2})[0];
    auto const expected    = LCW<T>{{1, 2, 3, 4}, {5, 6, 7}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

using SortListsDouble = SortLists<double>;
TEST_F(SortListsDouble, InfinityAndNaN)
{
  auto constexpr NaN = std::numeric_limits<double>::quiet_NaN();
  auto constexpr Inf = std::numeric_limits<double>::infinity();

  using LCW = cudf::test::lists_column_wrapper<double>;
  {
    LCW input{-0.0, -NaN, -NaN, NaN, Inf, -Inf, 7, 5, 6, NaN, Inf, -Inf, -NaN, -NaN, -0.0};
    auto [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{input}, {}, {});
    LCW expected{-Inf, -Inf, -0, -0, 5, 6, 7, Inf, Inf, -NaN, -NaN, NaN, NaN, -NaN, -NaN};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(stable_sorted_lists->view(), expected);
  }
  // Over 200 no-null elements route the unstable `sort_lists` through the packed-radix engine;
  // the EQUIVALENT assertions tolerate either engine's ordering.
  {
    // clang-format off
    LCW input{0.0, -0.0, -NaN, -NaN, NaN, Inf, -Inf,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
              NaN, Inf, -Inf, -NaN, -NaN, -0.0, 0.0};
    LCW expected{-Inf, -Inf, 0.0, -0.0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0, 0,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
               3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
               4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
               5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
               6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
               8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
               9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
              Inf, Inf, -NaN, -NaN, NaN, NaN, -NaN, -NaN};
    // clang-format on
    auto [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{input}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(stable_sorted_lists->view(), expected);
  }
}
