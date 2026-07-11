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

// EQUIVALENT variant for floating point: -0.0/+0.0 and distinct NaN bit patterns are
// interchangeable under the unstable contract, so per-slot bytes may differ within such groups.
void expect_both_sort_paths_equivalent(cudf::lists_column_view const& input,
                                       cudf::column_view const& expected,
                                       cudf::order column_order         = cudf::order::ASCENDING,
                                       cudf::null_order null_precedence = cudf::null_order::AFTER)
{
  auto const sorted = cudf::lists::sort_lists(input, column_order, null_precedence);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted->view(), expected);
  auto const stable_sorted = cudf::lists::stable_sort_lists(input, column_order, null_precedence);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(stable_sorted->view(), expected);
}

// Host oracle for one list row. cudf's null_order is comparator-level and DESCENDING swaps the
// comparison operands, inverting null placement too: nulls land first iff (BEFORE) != (DESCENDING).
template <typename T>
std::pair<std::vector<T>, std::vector<bool>> host_sorted_row(std::vector<T> const& vals,
                                                             std::vector<bool> const& valids,
                                                             cudf::order column_order,
                                                             cudf::null_order null_precedence)
{
  std::vector<T> nn;
  for (std::size_t i = 0; i < vals.size(); ++i) {
    if (valids[i]) { nn.push_back(vals[i]); }
  }
  std::sort(nn.begin(), nn.end());
  if (column_order == cudf::order::DESCENDING) { std::reverse(nn.begin(), nn.end()); }
  auto const num_nulls = vals.size() - nn.size();
  auto const nulls_first =
    (null_precedence == cudf::null_order::BEFORE) != (column_order == cudf::order::DESCENDING);
  std::vector<T> out;
  std::vector<bool> out_valids;
  auto const push_nulls = [&] {
    for (std::size_t k = 0; k < num_nulls; ++k) {
      out.push_back(T{});
      out_valids.push_back(false);
    }
  };
  if (nulls_first) { push_nulls(); }
  for (auto const& v : nn) {
    out.push_back(v);
    out_valids.push_back(true);
  }
  if (not nulls_first) { push_nulls(); }
  return {std::move(out), std::move(out_valids)};
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

namespace {
// Builds a LIST<STRING> column from per-row (string, validity) data. Embedded NUL bytes ride
// through unchanged, so callers pass explicit-length `std::string`s. The leaf takes a mask only
// when a null exists: the sort returns null-free columns non-nullable, and expected columns built
// here must match that contract.
std::unique_ptr<cudf::column> make_string_lists(std::vector<std::vector<std::string>> const& rows,
                                                std::vector<std::vector<bool>> const& valids)
{
  std::vector<std::string> flat;
  std::vector<bool> flat_v;
  std::vector<cudf::size_type> offsets{0};
  for (std::size_t r = 0; r < rows.size(); ++r) {
    flat.insert(flat.end(), rows[r].begin(), rows[r].end());
    flat_v.insert(flat_v.end(), valids[r].begin(), valids[r].end());
    offsets.push_back(static_cast<cudf::size_type>(flat.size()));
  }
  auto const has_null_element = std::find(flat_v.cbegin(), flat_v.cend(), false) != flat_v.cend();
  auto leaf =
    has_null_element
      ? cudf::test::strings_column_wrapper(flat.begin(), flat.end(), flat_v.begin()).release()
      : cudf::test::strings_column_wrapper(flat.begin(), flat.end()).release();
  cudf::test::fixed_width_column_wrapper<cudf::size_type> off(offsets.begin(), offsets.end());
  return cudf::make_lists_column(
    static_cast<cudf::size_type>(rows.size()), off.release(), std::move(leaf), 0, {});
}

// Drives the rows through all four (order, null_order) combos against a per-row host oracle;
// `std::string`'s unsigned-byte order and length tie-break match cudf's string order.
void expect_string_polarity_matrix(std::vector<std::vector<std::string>> const& rows,
                                   std::vector<std::vector<bool>> const& valids)
{
  constexpr std::array<std::pair<cudf::order, cudf::null_order>, 4> combos{
    {{cudf::order::ASCENDING, cudf::null_order::AFTER},
     {cudf::order::DESCENDING, cudf::null_order::AFTER},
     {cudf::order::ASCENDING, cudf::null_order::BEFORE},
     {cudf::order::DESCENDING, cudf::null_order::BEFORE}}};
  auto const input = make_string_lists(rows, valids);
  for (auto const& [ord, np] : combos) {
    std::vector<std::vector<std::string>> ex_rows(rows.size());
    std::vector<std::vector<bool>> ex_valids(rows.size());
    for (std::size_t r = 0; r < rows.size(); ++r) {
      auto sorted_row = host_sorted_row(rows[r], valids[r], ord, np);
      ex_rows[r]      = std::move(sorted_row.first);
      ex_valids[r]    = std::move(sorted_row.second);
    }
    auto const expected = make_string_lists(ex_rows, ex_valids);
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view(), ord, np);
  }
}
}  // namespace

// ===== graduated-warp string path: per-band polarity coverage (segments within the 64-element cap)
// The default string fast path once every segment fits a warp tile; its bands are (0,8] one item
// per lane, then (8,16] W8, W16, and W32.

// Sizes at and across every band boundary (8/9, 15/16/17, 31/32/33, 63/64) plus an empty row, side
// by side in one column; the 8/16/32/64 rows fill their tiles exactly (zero pads).
TEST_F(SortListsString, GradPolarityMatrixBandBoundarySizes)
{
  std::vector<cudf::size_type> const sizes{1, 2, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 0};
  std::mt19937 rng{0xBEEF};
  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (auto const n : sizes) {
    std::vector<std::string> row(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      // Distinct within a row: 97 is coprime to 1000, so the numeric part never repeats for i < 64.
      row[i] = "k" + std::to_string(i * 97 % 1'000) + static_cast<char>('a' + i % 26);
    }
    std::shuffle(row.begin(), row.end(), rng);
    valids.emplace_back(row.size(), true);
    rows.push_back(std::move(row));
  }
  expect_string_polarity_matrix(rows, valids);
}

// The nulls-first combos put the valid class at ordinal 1: a comparator hardcoding it to 0 would
// collapse valid-vs-valid pairs to "equal" and keep the shuffled order. Row 1 (2 valid / 15 null /
// 15 pad) stresses the pad class ranking above tier_null; row 5 is all null.
TEST_F(SortListsString, GradPolarityMatrixNullsAcrossBands)
{
  std::vector<cudf::size_type> const sizes{12, 17, 20, 40, 64, 16, 5, 8};
  std::mt19937 rng{0xD00D};
  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (std::size_t r = 0; r < sizes.size(); ++r) {
    auto const n = sizes[r];
    std::vector<std::pair<std::string, bool>> cells(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      // Row 1: only elements 3 and 11 valid (2 valid / 15 null). Row 5: all null. Others: every
      // (i % 5 == 2) element is null.
      bool const v = (r == 1) ? (i == 3 || i == 11) : (r == 5 ? false : (i % 5 != 2));
      cells[i]     = {"v" + std::to_string(i * 97 % 1'000), v};
    }
    if (n > 1 && cells[0].second && cells[1].second) {
      cells[1].first = cells[0].first;  // A duplicate among the valids.
    }
    std::shuffle(cells.begin(), cells.end(), rng);  // Interleave nulls; don't pre-group them.
    std::vector<std::string> row(n);
    std::vector<bool> valid(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      row[i]   = cells[i].first;
      valid[i] = cells[i].second;
    }
    rows.push_back(std::move(row));
    valids.push_back(std::move(valid));
  }
  expect_string_polarity_matrix(rows, valids);
}

// Zero-extension pairs (`S` vs `S + "\0"`) collide on the zero-filled window and resolve only by
// the length tie-break, whose side the descending complement flips; the shared eight-byte prefixes
// force every pair onto the suffix fallback in each band. Row 3 is pure duplicates.
TEST_F(SortListsString, GradPolarityMatrixTiesAndZeroExtension)
{
  using namespace std::string_literals;
  std::mt19937 rng{0xACE5};
  std::vector<std::string> row0{""s, "a"s, "ab"s, "ab"s, "ab\0"s, "ab\0c"s, "abc"s, "abcd"s, "b"s};
  std::vector<std::string> row1;
  for (int i = 0; i < 16; ++i) {
    row1.push_back("PPPPPPPP" + std::to_string(i));
  }
  row1.push_back("PPPPPPPP"s);
  row1.push_back("PPPPPPPP\0"s);
  row1.push_back("PPPPPPPP3"s);  // duplicate
  row1.push_back("PPPPPPPP7"s);  // duplicate
  std::vector<std::string> row2;
  for (int i = 0; i < 38; ++i) {
    row2.push_back("SAMEPREF" + std::string(1, static_cast<char>('0' + (i % 10))) +
                   std::to_string(i));
  }
  row2.push_back("SAMEPREF"s);
  row2.push_back("SAMEPREF\0"s);
  std::vector<std::string> row3(10, "dup"s);

  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (auto* row : {&row0, &row1, &row2, &row3}) {
    std::shuffle(row->begin(), row->end(), rng);
    valids.emplace_back(row->size(), true);
    rows.push_back(*row);
  }
  expect_string_polarity_matrix(rows, valids);
}

// Raw unsigned-byte order across the 2-/3-/4-byte UTF-8 lead classes; row 1 shares a multibyte
// prefix so the prekey ties past the lead bytes.
TEST_F(SortListsString, GradPolarityMatrixUtf8Multibyte)
{
  std::mt19937 rng{0xFACE};
  std::vector<std::string> row0{
    "\xE2\x82\xAC\x75\x72\x6F", "zoo", "\xF0\x9F\x98\x80", "\xC3\xA9\x70\xC3\xA9\x65", "apple"};
  std::vector<std::string> row1;
  for (int i = 0; i < 36; ++i) {
    row1.push_back("\xC3\xA9" + std::to_string(i * 97 % 1'000));
  }
  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (auto* row : {&row0, &row1}) {
    std::shuffle(row->begin(), row->end(), rng);
    valids.emplace_back(row->size(), true);
    rows.push_back(*row);
  }
  expect_string_polarity_matrix(rows, valids);
}

// A few hundred segments cycling every band size in (0, 64] drive the string instantiations of
// the shared warp-band kernel across block boundaries -- hundreds of virtual warps over multiple
// blocks per band -- where the small tests above stay within two blocks. Scattered nulls and a
// sprinkle of empty rows ride along, all against the host oracle under every combo.
TEST_F(SortListsString, GradManySegmentsAcrossBands)
{
  std::vector<std::vector<std::string>> rows;
  std::vector<std::vector<bool>> valids;
  for (int seg = 0; seg < 300; ++seg) {
    if (seg % 50 == 49) {  // Interleave empty rows; no band owns them.
      rows.emplace_back();
      valids.emplace_back();
      continue;
    }
    auto const size = seg % 64 + 1;
    std::vector<std::string> row(size);
    std::vector<bool> valid(size);
    for (int i = 0; i < size; ++i) {
      row[i]   = "s" + std::to_string((seg * 31 + i * 7) % 97) + std::string(i % 9, 'x');
      valid[i] = (seg * 13 + i) % 11 != 0;
    }
    rows.push_back(std::move(row));
    valids.push_back(std::move(valid));
  }
  expect_string_polarity_matrix(rows, valids);
}

// Slicing off row 0 gives the leaf strings a genuine nonzero offset, which the graduated key
// builders and comparators must honor under every combo.
TEST_F(SortListsString, GradSlicedWithNulls)
{
  using StrLCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  StrLCW l{StrLCW{"zz", "aa"},
           StrLCW{{"banana", "apple", "x", "cherry"}, null_at(2)},  // element 2 ("x") is null
           StrLCW{"b", "a"},
           StrLCW{"x"}};
  auto const sliced = cudf::slice(l, {1, 4})[0];  // drops row 0 -> nonzero child offset
  {                                               // ASC / AFTER: values ascending, null last
    StrLCW expected{
      StrLCW{{"apple", "banana", "cherry", "x"}, null_at(3)}, StrLCW{"a", "b"}, StrLCW{"x"}};
    expect_both_sort_paths_match(cudf::lists_column_view{sliced}, expected);
  }
  {  // ASC / BEFORE: null first, then values ascending
    StrLCW expected{
      StrLCW{{"x", "apple", "banana", "cherry"}, null_at(0)}, StrLCW{"a", "b"}, StrLCW{"x"}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{sliced}, expected, cudf::order::ASCENDING, cudf::null_order::BEFORE);
  }
  {  // DESC / AFTER: the comparator swap places the null first, then values descending
    StrLCW expected{
      StrLCW{{"x", "cherry", "banana", "apple"}, null_at(0)}, StrLCW{"b", "a"}, StrLCW{"x"}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{sliced}, expected, cudf::order::DESCENDING, cudf::null_order::AFTER);
  }
  {  // DESC / BEFORE: values descending, then the null last
    StrLCW expected{
      StrLCW{{"cherry", "banana", "apple", "x"}, null_at(3)}, StrLCW{"b", "a"}, StrLCW{"x"}};
    expect_both_sort_paths_match(
      cudf::lists_column_view{sliced}, expected, cudf::order::DESCENDING, cudf::null_order::BEFORE);
  }
}

// One 65-element segment exceeds the warp cap and disqualifies the whole column, so the prefix
// path must produce the result under every combo.
TEST_F(SortListsString, GradOversizedSegmentFallsThrough)
{
  std::mt19937 rng{0xF00D};
  std::vector<std::string> big(65);
  for (int i = 0; i < 65; ++i) {
    big[i] = "q" + std::to_string(i * 97 % 1'000);
  }
  std::shuffle(big.begin(), big.end(), rng);
  std::vector<std::vector<std::string>> const rows{big, {"b", "a", "c"}};
  std::vector<std::vector<bool>> const valids{std::vector<bool>(big.size(), true),
                                              {true, true, true}};
  expect_string_polarity_matrix(rows, valids);
}

// Sign-flip correctness at INT_MIN / INT_MAX for both tiered key widths (int32 -> uint64,
// int64 -> unsigned __int128); the nulls route the rows to the tiered kernel.
TEST_F(SortListsInt, NumericPackedNegativesAndBounds)
{
  {  // int32: the <= 4-byte uint64 tiered key.
    auto constexpr lo = std::numeric_limits<int32_t>::min();
    auto constexpr hi = std::numeric_limits<int32_t>::max();
    std::vector<bool> const in_valids{true, true, false, true, true, true, false};
    LCW<int32_t> input{{{hi, -1, 0, lo, 0, 1, 0}, in_valids.begin()}};
    std::vector<bool> const ex_valids{true, true, true, true, true, false, false};
    LCW<int32_t> expected{{{lo, -1, 0, 1, hi, 0, 0}, ex_valids.begin()}};
    expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
  }
  {  // int64: the 8-byte unsigned __int128 key. The file-wide `LCW` sources from `int32_t`, which
     // cannot hold the 64-bit bounds, hence the direct `int64_t` wrapper.
    using LCW64       = cudf::test::lists_column_wrapper<int64_t>;
    auto constexpr lo = std::numeric_limits<int64_t>::min();
    auto constexpr hi = std::numeric_limits<int64_t>::max();
    std::vector<bool> const in_valids{true, true, false, true, true, true, false};
    LCW64 input{{{hi, -1, 0, lo, 0, 1, 0}, in_valids.begin()}};
    std::vector<bool> const ex_valids{true, true, true, true, true, false, false};
    LCW64 expected{{{lo, -1, 0, 1, hi, 0, 0}, ex_valids.begin()}};
    expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
  }
}

// Degenerate shapes (empty, single-element, all-duplicate rows) through the tiered network tier;
// the null element is what routes the column there.
TEST_F(SortListsInt, NumericPackedEmptyAndSingleRows)
{
  std::vector<bool> const in_valids{true, false, true};
  LCW<int32_t> input{LCW<int32_t>{},
                     LCW<int32_t>{5},
                     LCW<int32_t>{-3, -3},
                     LCW<int32_t>{{2, 7, -1}, in_valids.begin()}};
  std::vector<bool> const ex_valids{true, true, false};
  LCW<int32_t> expected{LCW<int32_t>{},
                        LCW<int32_t>{5},
                        LCW<int32_t>{-3, -3},
                        LCW<int32_t>{{-1, 2, 7}, ex_valids.begin()}};
  expect_both_sort_paths_match(cudf::lists_column_view{input}, expected);
}

// Pre-/post-epoch timestamps for both rep widths guard the is_timestamp rep-extraction branch;
// the signed flip must sort pre-epoch (negative rep) values first.
TEST_F(SortListsInt, NumericPackedTimestamps)
{
  {  // TIMESTAMP_DAYS: int32 rep, the <= 4-byte uint64 tiered key.
    std::vector<int32_t> const in{100, -50, 0, 0, 25};
    std::vector<int32_t> const ex{-50, 0, 25, 100, 0};
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_D>(in.begin(), in.end(), null_at(2))
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_D>(ex.begin(), ex.end(), null_at(4))
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // TIMESTAMP_MILLISECONDS: int64 rep exceeding int32, the 8-byte unsigned __int128 tiered key.
    auto constexpr big = int64_t{5} * 1'000 * 1'000 * 1'000;  // 5e9 > INT32_MAX, post-epoch
    std::vector<int64_t> const in{big, -big, 0, 1, -1};
    std::vector<int64_t> const ex{-big, -1, 1, big, 0};
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms>(in.begin(), in.end(), null_at(2))
        .release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms>(ex.begin(), ex.end(), null_at(4))
        .release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
}

// BOOL8 is packed-radix-supported but not tiered, so a null routes it to the packed radix at any
// size -- the only test reaching the bool `radix_encode_u32` branch with the key's null bit.
TEST_F(SortListsInt, NumericPackedBoolWithNulls)
{
  std::vector<bool> const in_valids{true, true, false, true, true, false, true};
  auto input =
    as_single_row_list(cudf::test::fixed_width_column_wrapper<bool>(
                         {true, false, true, true, false, false, false}, in_valids.begin())
                         .release());
  std::vector<bool> const ex_valids{true, true, true, true, true, false, false};
  auto expected =
    as_single_row_list(cudf::test::fixed_width_column_wrapper<bool>(
                         {false, false, false, true, true, false, false}, ex_valids.begin())
                         .release());
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

// Pins the packed-radix gate's avg >= 100 boundary exactly: average 100 is the first packed-radix
// shape, average 99 the last tiered one.
TEST_F(SortListsInt, NumericPackedRadixAvgGateBoundary)
{
  auto const check_single_row = [](cudf::size_type n) {
    std::vector<int32_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = n / 2 - i;
    }
    std::vector<int32_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int32_t>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int32_t>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check_single_row(200);  // average exactly 100 -> the first packed-radix cell
  check_single_row(198);  // average 99 -> the last tiered cell
}

// The long-list band (avg at/above the packed-radix cutoff) belongs to the one-shot packed radix;
// all three key widths span the sign boundary.
TEST_F(SortListsInt, NumericPackedRadixLongLists)
{
  cudf::size_type const n = 220;  // single row, average 110 -> packed radix
  {                               // int32
    std::vector<int32_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = n / 2 - i;
    }
    std::vector<int32_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int32_t>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int32_t>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // int64 beyond the 32-bit range
    auto constexpr big = int64_t{5} * 1'000 * 1'000 * 1'000;
    std::vector<int64_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = (static_cast<int64_t>(n / 2) - i) * big;
    }
    std::vector<int64_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int64_t>(in.begin(), in.end()).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_width_column_wrapper<int64_t>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
  {  // decimal128 into the high value words
    auto constexpr scale = numeric::scale_type{0};
    auto const k         = [] {
      __int128_t v = 1;
      for (int i = 0; i < 30; ++i) {
        v *= 10;
      }
      return v;
    }();
    std::vector<__int128_t> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<__int128_t>(n / 2 - i) * k;
    }
    std::vector<__int128_t> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(in.begin(), in.end(), scale).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<__int128_t>(ex.begin(), ex.end(), scale).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  }
}

// No-null non-tiered types reach the packed radix only when num_rows >= 1<<18 AND avg >= 100 -- a
// route no other test takes; the no-null key drops the null-class bit for the full value budget.
TEST_F(SortListsInt, NumericPackedRadixNoNullNarrowTypes)
{
  cudf::size_type const n = 1 << 18;
  auto const check        = [&](auto tag) {
    using T = decltype(tag);
    std::vector<T> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<T>((static_cast<int64_t>(n) - i) % 60'000);
    }
    std::vector<T> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input =
      as_single_row_list(cudf::test::fixed_width_column_wrapper<T>(in.begin(), in.end()).release());
    auto expected =
      as_single_row_list(cudf::test::fixed_width_column_wrapper<T>(ex.begin(), ex.end()).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check(uint16_t{});
  check(int16_t{});
  check(uint64_t{});

  // DECIMAL32/64 (non-tiered) take the same no-null packed radix over their int32/int64 rep.
  auto const check_decimal = [&](auto rep_tag) {
    using Rep            = decltype(rep_tag);
    auto constexpr scale = numeric::scale_type{0};
    std::vector<Rep> in(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      in[i] = static_cast<Rep>((static_cast<int64_t>(n) - i) % 60'000);
    }
    std::vector<Rep> ex(in);
    std::sort(ex.begin(), ex.end());
    auto input = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<Rep>(in.begin(), in.end(), scale).release());
    auto expected = as_single_row_list(
      cudf::test::fixed_point_column_wrapper<Rep>(ex.begin(), ex.end(), scale).release());
    expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
  };
  check_decimal(int32_t{});  // DECIMAL32.
  check_decimal(int64_t{});  // DECIMAL64.
}

// The CUB-preference gate is a disjunction: a high average alone must not force the packed radix
// while the total row count stays under `MAX_LIST_SIZE_FOR_FAST_SORT`. This shape (avg 149,
// 150'000 rows) holds only through the row-count disjunct, exercising the CUB fast path at a
// narrow-type cell no other test reaches; fast and comparison paths must agree.
TEST_F(SortListsInt, NumericPackedRadixNarrowTypeHighAvgLowRowCount)
{
  cudf::size_type const num_segments = 1'000;
  cudf::size_type const seg_size     = 150;  // avg = 150'000 / 1'001 = 149 >= 100
  std::vector<uint16_t> in_vals(static_cast<std::size_t>(num_segments) * seg_size);
  for (std::size_t i = 0; i < in_vals.size(); ++i) {
    in_vals[i] = static_cast<uint16_t>((i * 31) % 60'000);
  }
  std::vector<cudf::size_type> offsets(num_segments + 1);
  for (cudf::size_type i = 0; i <= num_segments; ++i) {
    offsets[i] = i * seg_size;
  }
  std::vector<uint16_t> ex_vals(in_vals);
  for (cudf::size_type s = 0; s < num_segments; ++s) {
    std::sort(ex_vals.begin() + offsets[s], ex_vals.begin() + offsets[s + 1]);
  }
  auto const make_list = [&](std::vector<uint16_t> const& vals) {
    return cudf::make_lists_column(
      num_segments,
      cudf::test::fixed_width_column_wrapper<cudf::size_type>(offsets.begin(), offsets.end())
        .release(),
      cudf::test::fixed_width_column_wrapper<uint16_t>(vals.begin(), vals.end()).release(),
      0,
      {});
  };
  auto const input    = make_list(in_vals);
  auto const expected = make_list(ex_vals);
  expect_both_sort_paths_match(cudf::lists_column_view{input->view()}, expected->view());
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

// cudf orders -Inf < finite < +Inf < NaN (all payloads equal, last), nulls after NaN. Every NaN
// canonicalizes to the all-ones key, so an un-canonicalized -NaN cannot sort near -Inf; the two
// rows land in the warp tier, covering both key widths.
TEST_F(SortListsDouble, NumericPackedFloatNaNInfinity)
{
  {  // FLOAT64: the 8-byte unsigned __int128 tiered key; thirteen elements land in the warp tier.
    auto constexpr NaN    = std::numeric_limits<double>::quiet_NaN();
    auto constexpr Inf    = std::numeric_limits<double>::infinity();
    auto constexpr denorm = std::numeric_limits<double>::denorm_min();
    auto constexpr Max    = std::numeric_limits<double>::max();  // DBL_MAX boundary vs. Inf
    using LCWd            = cudf::test::lists_column_wrapper<double>;
    LCWd input{
      {{NaN, -Inf, -0.0, 3.5, -NaN, Inf, 0.0, denorm, -2.0, 0.0, Inf, Max, -Max}, null_at(9)}};
    LCWd expected{
      {{-Inf, -Max, -2.0, -0.0, 0.0, denorm, 3.5, Max, Inf, Inf, NaN, -NaN, 0.0}, null_at(12)}};
    expect_both_sort_paths_equivalent(cudf::lists_column_view{input}, expected);
  }
  {  // FLOAT32: the <= 4-byte uint64 tiered key; ten elements land in the warp tier.
    auto constexpr NaN    = std::numeric_limits<float>::quiet_NaN();
    auto constexpr Inf    = std::numeric_limits<float>::infinity();
    auto constexpr denorm = std::numeric_limits<float>::denorm_min();
    auto constexpr Max    = std::numeric_limits<float>::max();  // FLT_MAX boundary vs. Inf
    using LCWf            = cudf::test::lists_column_wrapper<float>;
    LCWf input{{{-NaN, Inf, -0.0f, 2.5f, -Inf, denorm, 0.0f, NaN, Max, -Max}, null_at(6)}};
    LCWf expected{{{-Inf, -Max, -0.0f, denorm, 2.5f, Max, Inf, NaN, -NaN, 0.0f}, null_at(9)}};
    expect_both_sort_paths_equivalent(cudf::lists_column_view{input}, expected);
  }
}
