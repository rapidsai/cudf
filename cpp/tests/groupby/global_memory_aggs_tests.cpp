/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cstddef>
#include <memory>
#include <vector>

/*
 * The hash groupby uses its global memory kernels whenever a value column is a dictionary or a
 * SUM_OVERFLOW aggregation is requested.  Null keys select the nullable hash set, and more than two
 * single-pass aggregations select the dense output kernel instead of the sparse one.  The nullable
 * kernels are split by value column type, so a table with both dictionary and non-dictionary
 * values launches once per type.  These tests run every combination of those paths against the
 * sort-based implementation, with null keys both excluded and included.
 */

namespace {

struct request_spec {
  cudf::column_view values;
  std::vector<cudf::aggregation::Kind> kinds;
};

std::unique_ptr<cudf::groupby_aggregation> make_aggregation(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::SUM: return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::MIN: return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::MAX: return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::COUNT_VALID:
      return cudf::make_count_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::SUM_OVERFLOW:
      return cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();
    default: CUDF_FAIL("Aggregation kind is not used by these tests.");
  }
}

enum class implementation { HASH, SORT };

// Result rows sorted by key, so the unordered hash output can be compared with the sorted output.
struct sorted_result {
  std::unique_ptr<cudf::table> keys;
  std::vector<std::unique_ptr<cudf::table>> values;
};

sorted_result aggregate(cudf::column_view const& keys,
                        std::vector<request_spec> const& specs,
                        cudf::null_policy null_handling,
                        implementation impl)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  for (auto const& spec : specs) {
    auto& request  = requests.emplace_back();
    request.values = spec.values;
    for (auto kind : spec.kinds) {
      request.aggregations.push_back(make_aggregation(kind));
    }
  }
  // nth_element is not a hash aggregation, so adding it forces the sort-based implementation.  It
  // is appended last so the positions of the requested results are unchanged.
  if (impl == implementation::SORT) {
    requests.front().aggregations.push_back(
      cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));
  }

  cudf::groupby::groupby gb_obj(cudf::table_view{{keys}}, null_handling);
  auto [result_keys, results] = gb_obj.aggregate(requests);

  auto const order = cudf::sorted_order(result_keys->view());
  sorted_result sorted;
  sorted.keys = cudf::gather(result_keys->view(), *order);
  for (std::size_t i = 0; i < specs.size(); ++i) {
    for (std::size_t j = 0; j < specs[i].kinds.size(); ++j) {
      sorted.values.push_back(
        cudf::gather(cudf::table_view{{results[i].results[j]->view()}}, *order));
    }
  }
  return sorted;
}

void expect_hash_matches_sort(cudf::column_view const& keys, std::vector<request_spec> const& specs)
{
  for (auto null_handling : {cudf::null_policy::EXCLUDE, cudf::null_policy::INCLUDE}) {
    auto const hash = aggregate(keys, specs, null_handling, implementation::HASH);
    auto const sort = aggregate(keys, specs, null_handling, implementation::SORT);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort.keys->view(), hash.keys->view());
    ASSERT_EQ(sort.values.size(), hash.values.size());
    for (std::size_t i = 0; i < sort.values.size(); ++i) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sort.values[i]->get_column(0),
                                          hash.values[i]->get_column(0));
    }
  }
}

}  // namespace

struct groupby_global_memory_aggs_test : public cudf::test::BaseFixture {
  // Two null keys and one null in each value column, so null handling is exercised on both sides
  // of every aggregation.
  cudf::test::fixed_width_column_wrapper<int32_t> keys{{1, 2, 3, 1, 2, 2, 1, 3, 3, 2},
                                                       {1, 0, 1, 1, 1, 1, 0, 1, 1, 1}};
  cudf::test::dictionary_column_wrapper<int32_t> dictionary_values{
    {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}, {1, 1, 1, 1, 0, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int64_t> plain_values{{5, 4, 3, 2, 1, 0, -1, -2, -3, -4},
                                                               {1, 1, 1, 1, 1, 1, 1, 1, 0, 1}};
};

TEST_F(groupby_global_memory_aggs_test, DictionaryOnlySparse)
{
  expect_hash_matches_sort(keys, {{dictionary_values, {cudf::aggregation::SUM}}});
}

TEST_F(groupby_global_memory_aggs_test, DictionaryOnlyDense)
{
  expect_hash_matches_sort(
    keys,
    {{dictionary_values,
      {cudf::aggregation::SUM, cudf::aggregation::MIN, cudf::aggregation::MAX}}});
}

TEST_F(groupby_global_memory_aggs_test, NonDictionaryOnlySparse)
{
  expect_hash_matches_sort(keys, {{plain_values, {cudf::aggregation::SUM_OVERFLOW}}});
}

TEST_F(groupby_global_memory_aggs_test, NonDictionaryOnlyDense)
{
  expect_hash_matches_sort(
    keys,
    {{plain_values,
      {cudf::aggregation::SUM_OVERFLOW, cudf::aggregation::MIN, cudf::aggregation::MAX}}});
}

TEST_F(groupby_global_memory_aggs_test, MixedSparse)
{
  expect_hash_matches_sort(
    keys,
    {{dictionary_values, {cudf::aggregation::SUM}}, {plain_values, {cudf::aggregation::MAX}}});
}

TEST_F(groupby_global_memory_aggs_test, MixedDense)
{
  expect_hash_matches_sort(
    keys,
    {{dictionary_values, {cudf::aggregation::SUM, cudf::aggregation::MIN}},
     {plain_values, {cudf::aggregation::MAX, cudf::aggregation::COUNT_VALID}}});
}
