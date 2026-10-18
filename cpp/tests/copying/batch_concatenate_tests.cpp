/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <numeric>
#include <vector>

/**
 * @brief Test fixture for batch_concatenate tests.
 *
 * These tests focus on edge cases for the bitmask concatenation kernel,
 * particularly around word boundaries (32-bit alignment).
 */
struct BatchConcatenateTest : public cudf::test::BaseFixture {};

// Helper to create a column with specific validity pattern
template <typename T>
cudf::test::fixed_width_column_wrapper<T> make_column_with_validity(
  std::vector<T> const& values, std::vector<bool> const& validity)
{
  return cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end(), validity.begin());
}

// ============================================================================
// Basic functionality tests
// ============================================================================

// ============================================================================
// Word boundary edge cases (32-bit alignment)
// ============================================================================

// ============================================================================
// Empty column edge cases
// ============================================================================

// ============================================================================
// Mixed null mask scenarios
// ============================================================================

// ============================================================================
// Source bit offset tests (sliced columns)
// ============================================================================

// ============================================================================
// Large column tests
// ============================================================================

// ============================================================================
// Specific bit pattern tests
// ============================================================================

// ============================================================================
// Struct column tests
// ============================================================================

TEST_F(BatchConcatenateTest, SmallPayloadFallsBackThroughPublicAPI)
{
  // An all-flat input under the 8 MiB payload threshold delegates to the standard
  // concatenate; this owns that decision for the column overload.
  auto col1 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}, {1, 0, 1});
  auto col2 = cudf::test::fixed_width_column_wrapper<int32_t>({4, 5, 6}, {0, 1, 0});
  std::vector<cudf::column_view> columns = {col1, col2};

  auto result = cudf::batch_concatenate(columns);

  auto expected =
    cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3, 4, 5, 6}, {1, 0, 1, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(BatchConcatenateTest, StructColumnWithNulls)
{
  // Struct with two int32 children
  auto child1_col1 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3}, {1, 1, 0});
  auto child2_col1 = cudf::test::fixed_width_column_wrapper<int32_t>({10, 20, 30}, {1, 0, 1});
  auto struct_col1 = cudf::test::structs_column_wrapper({child1_col1, child2_col1}, {1, 0, 1});

  auto child1_col2 = cudf::test::fixed_width_column_wrapper<int32_t>({4, 5}, {0, 1});
  auto child2_col2 = cudf::test::fixed_width_column_wrapper<int32_t>({40, 50}, {1, 1});
  auto struct_col2 = cudf::test::structs_column_wrapper({child1_col2, child2_col2}, {1, 1});

  std::vector<cudf::column_view> columns = {struct_col1, struct_col2};

  auto result = cudf::batch_concatenate(
    columns, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  // Concatenate the children and the parent validity in order
  auto exp_child1 =
    cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3, 4, 5}, {1, 1, 0, 0, 1});
  auto exp_child2 =
    cudf::test::fixed_width_column_wrapper<int32_t>({10, 20, 30, 40, 50}, {1, 0, 1, 1, 1});
  auto expected = cudf::test::structs_column_wrapper({exp_child1, exp_child2}, {1, 0, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(BatchConcatenateTest, SlicedListOfStruct)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a{0, 1, 2, 3, 4, 5, 6, 7};
  cudf::test::strings_column_wrapper b{"v", "w", "x", "y", "z", "p", "q", "r"};
  cudf::test::structs_column_wrapper child{{a, b}, {1, 1, 0, 1, 1, 1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, 2, 3, 3, 6, 8};
  auto col = cudf::make_lists_column(5,
                                     offsets.release(),
                                     child.release(),
                                     0,
                                     cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  auto const slices = cudf::slice(col->view(), {1, 4});
  std::vector<cudf::column_view> const sources{slices[0], col->view()};

  // Sources: rows {1, 2, 3} of `col` (child rows: {a=null,b=x}; empty; {a=3,4,5},{b=p,q,r}),
  // then all 5 rows of `col`. Concatenated: 8 list rows, 12 child rows.
  auto exp_a = cudf::test::fixed_width_column_wrapper<int32_t>(
    {2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1});
  auto exp_b =
    cudf::test::strings_column_wrapper({"x", "y", "z", "p", "v", "w", "x", "y", "z", "p", "q", "r"},
                                       {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1});
  auto exp_child =
    cudf::test::structs_column_wrapper({exp_a, exp_b}, {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> exp_offsets{0, 1, 1, 4, 6, 7, 7, 10, 12};
  auto expected = cudf::make_lists_column(8,
                                          exp_offsets.release(),
                                          exp_child.release(),
                                          0,
                                          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  auto result = cudf::batch_concatenate(
    sources, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, SlicedStructOfList)
{
  cudf::test::lists_column_wrapper<int32_t> inner{{1, 2}, {}, {3, 4, 5}, {6}, {7, 8}, {9}};
  cudf::test::fixed_width_column_wrapper<int32_t> flat{10, 11, 12, 13, 14, 15};
  cudf::test::structs_column_wrapper outer{{inner, flat}, {1, 1, 1, 0, 1, 1}};

  auto const slices = cudf::slice(outer, {2, 5});
  std::vector<cudf::column_view> const sources{slices[0], slices[0]};

  // Sources: rows {2, 3, 4} of `outer`, twice. Concatenated inner child rows: 2..7, twice;
  // `flat` rows 2..4, twice; outer validity rows 2..4, twice.
  auto exp_inner =
    cudf::test::lists_column_wrapper<int32_t>({{3, 4, 5}, {6}, {7, 8}, {3, 4, 5}, {6}, {7, 8}});
  auto exp_flat = cudf::test::fixed_width_column_wrapper<int32_t>({12, 13, 14, 12, 13, 14});
  auto expected = cudf::test::structs_column_wrapper({exp_inner, exp_flat}, {1, 0, 1, 1, 0, 1});

  auto result = cudf::batch_concatenate(
    sources, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(BatchConcatenateTest, SlicedListOfList)
{
  cudf::test::lists_column_wrapper<int32_t> col{
    {{1, 2}, {3}}, {{4, 5, 6}}, {}, {{7}, {8, 9}, {10}}, {{11, 12}}};

  auto const slices = cudf::slice(col, {1, 4});
  std::vector<cudf::column_view> const sources{slices[0], slices[0]};

  // Sources: rows {1, 2, 3} of `col`, twice. Concatenation is the 6-row list-of-lists:
  // {{4,5,6}}, {}, {{7},{8,9},{10}} followed by the same three lists again.
  auto expected = cudf::test::lists_column_wrapper<int32_t>(
    {{{4, 5, 6}}, {}, {{7}, {8, 9}, {10}}, {{4, 5, 6}}, {}, {{7}, {8, 9}, {10}}});

  auto result = cudf::batch_concatenate(
    sources, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(BatchConcatenateTest, SlicedStringsUnderSlicedStruct)
{
  cudf::test::strings_column_wrapper s{"alpha", "beta", "gamma", "delta", "epsilon", "zeta"};
  cudf::test::fixed_width_column_wrapper<int32_t> n{1, 2, 3, 4, 5, 6};
  cudf::test::structs_column_wrapper outer{{s, n}, {1, 0, 1, 1, 1, 0}};

  auto const slices = cudf::slice(outer, {1, 5});
  std::vector<cudf::column_view> const sources{slices[0], slices[0]};

  // Sources: rows {1, 2, 3, 4} of `outer`, twice.
  auto exp_s = cudf::test::strings_column_wrapper(
    {"beta", "gamma", "delta", "epsilon", "beta", "gamma", "delta", "epsilon"});
  auto exp_n    = cudf::test::fixed_width_column_wrapper<int32_t>({2, 3, 4, 5, 2, 3, 4, 5});
  auto expected = cudf::test::structs_column_wrapper({exp_s, exp_n}, {0, 1, 1, 1, 0, 1, 1, 1});

  auto result = cudf::batch_concatenate(
    sources, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(BatchConcatenateTest, MixedSlicedAndUnslicedListSources)
{
  cudf::test::lists_column_wrapper<int32_t> a{{1, 2}, {3}, {}, {4, 5, 6}, {7}};
  cudf::test::lists_column_wrapper<int32_t> b{{8}, {9, 10}, {11}};

  auto const sliced = cudf::slice(a, {1, 4});
  std::vector<cudf::column_view> const sources{b, sliced[0], b};

  // Sources: `b`, rows {1, 2, 3} of `a` ({3}, {}, {4,5,6}), then `b` again.
  auto expected = cudf::test::lists_column_wrapper<int32_t>(
    {{8}, {9, 10}, {11}, {3}, {}, {4, 5, 6}, {8}, {9, 10}, {11}});

  // One source takes the host fast path and the other cannot, within a single call.
  auto result = cudf::batch_concatenate(
    sources, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(BatchConcatenateTest, SlicedListChildRangeStartsAtZero)
{
  // Offset zero but a short size, so the view is not verbatim even though its first offset is: this
  // is the branch `get_sliced_child` handles with a split rather than a slice.
  cudf::test::lists_column_wrapper<int32_t> col{{1, 2}, {3, 4}, {5, 6}, {7, 8}};

  auto const slices = cudf::slice(col, {0, 2});
  std::vector<cudf::column_view> const sources{slices[0], slices[0]};

  // Sources: rows {0, 1} of `col`, twice.
  auto expected = cudf::test::lists_column_wrapper<int32_t>({{1, 2}, {3, 4}, {1, 2}, {3, 4}});

  auto result = cudf::batch_concatenate(
    sources, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(BatchConcatenateTest, AllEmptyListSlice)
{
  cudf::test::lists_column_wrapper<int32_t> col{{1, 2}, {}, {}, {}, {3}};

  auto const slices = cudf::slice(col, {1, 4});
  std::vector<cudf::column_view> const sources{slices[0], slices[0]};

  // Sources: rows {1, 2, 3} of `col` (all empty lists), twice.
  auto expected = cudf::test::lists_column_wrapper<int32_t>({{}, {}, {}, {}, {}, {}});

  auto result = cudf::batch_concatenate(
    sources, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(BatchConcatenateTest, PlainListOfStruct)
{
  // Unsliced list-of-struct sources with child-row nulls (existing coverage is sliced-only)
  auto a1     = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto b1     = cudf::test::strings_column_wrapper{"x", "y", "z"};
  auto child1 = cudf::test::structs_column_wrapper({a1, b1}, {1, 0, 1}).release();
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets1{0, 2, 3};
  auto col1 = cudf::make_lists_column(2,
                                      offsets1.release(),
                                      std::move(child1),
                                      0,
                                      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  auto a2     = cudf::test::fixed_width_column_wrapper<int32_t>{4, 5};
  auto b2     = cudf::test::strings_column_wrapper{"p", "q"};
  auto child2 = cudf::test::structs_column_wrapper({a2, b2}, {1, 1}).release();
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets2{0, 1, 2};
  auto col2 = cudf::make_lists_column(2,
                                      offsets2.release(),
                                      std::move(child2),
                                      0,
                                      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  std::vector<cudf::column_view> const sources{col1->view(), col2->view()};

  auto exp_a     = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5};
  auto exp_b     = cudf::test::strings_column_wrapper{"x", "y", "z", "p", "q"};
  auto exp_child = cudf::test::structs_column_wrapper({exp_a, exp_b}, {1, 0, 1, 1, 1}).release();
  cudf::test::fixed_width_column_wrapper<cudf::size_type> exp_offsets{0, 2, 3, 4, 5};
  auto expected = cudf::make_lists_column(4,
                                          exp_offsets.release(),
                                          std::move(exp_child),
                                          0,
                                          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  auto result = cudf::batch_concatenate(
    sources, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(BatchConcatenateTest, StringsTopLevel)
{
  // Strings as direct top-level sources (existing coverage is strings-inside-tables/structs)
  auto col1 = cudf::test::strings_column_wrapper({"a", "bb", "ccc"}, {1, 0, 1});
  auto col2 = cudf::test::strings_column_wrapper({"dddd", "e"}, {0, 1});
  std::vector<cudf::column_view> const sources{col1, col2};

  auto expected =
    cudf::test::strings_column_wrapper({"a", "bb", "ccc", "dddd", "e"}, {1, 0, 1, 0, 1});
  auto result = cudf::batch_concatenate(
    sources, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(BatchConcatenateTest, ListWithParentNulls)
{
  // Null list ROWS (parent validity), not null children. Null rows are empty, so the child
  // contribution is unambiguous; the engine must concatenate the parent mask.
  using LCW = cudf::test::lists_column_wrapper<int32_t>;
  auto col1 = LCW{{{{}}, {{}}}, cudf::test::iterators::nulls_at({0})};
  auto col2 = LCW{{{{3}}, {{}}}, cudf::test::iterators::nulls_at({1})};
  std::vector<cudf::column_view> const sources{col1, col2};

  // Rows: NULL {}, {}, {3}, NULL {} — child holds only {3}
  auto expected = LCW{{{{}}, {{}}, {{3}}, {{}}}, cudf::test::iterators::nulls_at({0, 3})};
  auto result   = cudf::batch_concatenate(
    sources, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(BatchConcatenateTest, DictionaryColumns)
{
  // DICTIONARY32 delegates to cudf::dictionary::detail::concatenate
  auto col1 = cudf::test::dictionary_column_wrapper<int32_t>{1, 2, 1, 3};
  auto col2 = cudf::test::dictionary_column_wrapper<int32_t>{3, 1, 1, 1};
  std::vector<cudf::column_view> const sources{col1, col2};

  // Independent expectation: concatenated rows 1,2,1,3,3,1,1,1
  auto expected = cudf::test::dictionary_column_wrapper<int32_t>{1, 2, 1, 3, 3, 1, 1, 1};
  auto result   = cudf::batch_concatenate(
    sources, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(BatchConcatenateTest, ListChildLongerThanOffsetsSpan)
{
  // A child holding more elements than the offsets reach, which `make_lists_column` permits and
  // does not check. Consumers must bound by the offsets: taking the copy length from
  // `child().size()` instead pulls the unreachable trailing element into the output. This input
  // class had no coverage, and a host-side bounds shortcut that assumed the child was exact was
  // written and withdrawn on the strength of it.
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, 2, 4};
  cudf::test::fixed_width_column_wrapper<int32_t> child{1, 2, 3, 4, 99};
  auto col = cudf::make_lists_column(2,
                                     offsets.release(),
                                     child.release(),
                                     0,
                                     cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  // Sources: `col` twice. The output child must contain only the offsets-reachable rows
  // ({1, 2, 3, 4}), twice; the trailing unreachable element never appears.
  std::vector<cudf::column_view> const sources{col->view(), col->view()};
  auto exp_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 4, 6, 8};
  auto exp_child   = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 1, 2, 3, 4};
  auto expected    = cudf::make_lists_column(4,
                                          exp_offsets.release(),
                                          exp_child.release(),
                                          0,
                                          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  auto result = cudf::batch_concatenate(
    sources, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// ============================================================================
// Table batch_concatenate tests
// ============================================================================

struct BatchConcatenateTableTest : public cudf::test::BaseFixture {};

