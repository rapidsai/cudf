/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/sorting.hpp>

using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64s_col  = cudf::test::fixed_width_column_wrapper<int64_t>;
using structs_col = cudf::test::structs_column_wrapper;

auto groupby_histogram(cudf::column_view const& keys,
                       cudf::column_view const& values,
                       cudf::aggregation::Kind agg_kind)
{
  CUDF_EXPECTS(
    agg_kind == cudf::aggregation::HISTOGRAM || agg_kind == cudf::aggregation::MERGE_HISTOGRAM,
    "Aggregation must be either HISTOGRAM or MERGE_HISTOGRAM.");

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = values;
  if (agg_kind == cudf::aggregation::HISTOGRAM) {
    requests[0].aggregations.push_back(
      cudf::make_histogram_aggregation<cudf::groupby_aggregation>());
  } else {
    requests[0].aggregations.push_back(
      cudf::make_merge_histogram_aggregation<cudf::groupby_aggregation>());
  }

  auto gb_obj              = cudf::groupby::groupby(cudf::table_view({keys}));
  auto const agg_results   = gb_obj.aggregate(requests, cudf::test::get_default_stream());
  auto const agg_histogram = agg_results.second[0].results[0]->view();
  EXPECT_EQ(agg_histogram.type().id(), cudf::type_id::LIST);
  EXPECT_EQ(agg_histogram.null_count(), 0);

  auto const histograms = cudf::lists_column_view{agg_histogram}.child();
  EXPECT_EQ(histograms.num_children(), 2);
  EXPECT_EQ(histograms.null_count(), 0);
  EXPECT_EQ(histograms.child(1).null_count(), 0);

  auto const key_sort_order = cudf::sorted_order(agg_results.first->view(), {}, {});
  auto sorted_keys =
    std::move(cudf::gather(agg_results.first->view(), *key_sort_order)->release().front());
  auto const sorted_vals =
    std::move(cudf::gather(cudf::table_view{{agg_histogram}}, *key_sort_order)->release().front());
  auto sorted_histograms = cudf::lists::sort_lists(cudf::lists_column_view{*sorted_vals},
                                                   cudf::order::ASCENDING,
                                                   cudf::null_order::BEFORE,
                                                   rmm::mr::get_current_device_resource());

  return std::pair{std::move(sorted_keys), std::move(sorted_histograms)};
}

template <typename T>
struct GroupbyHistogramTest : public cudf::test::BaseFixture {};

template <typename T>
struct GroupbyMergeHistogramTest : public cudf::test::BaseFixture {};

// Avoid unsigned types, as the tests below have negative values in their input.
using HistogramTestTypes = cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t>,
                                              cudf::test::FloatingPointTypes,
                                              cudf::test::FixedPointTypes,
                                              cudf::test::ChronoTypes>;
TYPED_TEST_SUITE(GroupbyHistogramTest, HistogramTestTypes);
TYPED_TEST_SUITE(GroupbyMergeHistogramTest, HistogramTestTypes);

TYPED_TEST(GroupbyHistogramTest, EmptyInput)
{
  using col_data = cudf::test::fixed_width_column_wrapper<TypeParam, int>;

  auto const keys   = int32s_col{};
  auto const values = col_data{};
  auto const [res_keys, res_histogram] =
    groupby_histogram(keys, values, cudf::aggregation::HISTOGRAM);

  // The structure of the output is already verified in the function `groupby_histogram`.
  ASSERT_EQ(res_histogram->size(), 0);
}

TYPED_TEST(GroupbyHistogramTest, SimpleInputNoNull)
{
  using col_data = cudf::test::fixed_width_column_wrapper<TypeParam, int>;

  // key = 0: values = [2, 2, -3, -2, 2]
  // key = 1: values = [2, 0, 5, 2, 1]
  // key = 2: values = [-3, 1, 1, 2, 2]
  auto const keys               = int32s_col{2, 0, 2, 1, 1, 1, 0, 0, 0, 1, 2, 2, 1, 0, 2};
  auto const values             = col_data{-3, 2, 1, 2, 0, 5, 2, -3, -2, 2, 1, 2, 1, 2, 2};
  auto const expected_keys      = int32s_col{0, 1, 2};
  auto const expected_histogram = [] {
    auto structs = [] {
      auto values = col_data{-3, -2, 2, 0, 1, 2, 5, -3, 1, 2};
      auto counts = int64s_col{1, 1, 3, 1, 1, 2, 1, 1, 2, 2};
      return structs_col{{values, counts}};
    }();
    return cudf::make_lists_column(
      3, int32s_col{0, 3, 7, 10}.release(), structs.release(), 0, rmm::device_buffer{});
  }();
  auto const [res_keys, res_histogram] =
    groupby_histogram(keys, values, cudf::aggregation::HISTOGRAM);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *res_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_histogram, *res_histogram);
}

TYPED_TEST(GroupbyHistogramTest, SlicedInputNoNull)
{
  using col_data = cudf::test::fixed_width_column_wrapper<TypeParam, int>;

  auto const keys_original = int32s_col{2, 0, 2, 1, 0, 2, 0, 2, 1, 1, 1, 0, 0, 0, 1, 2, 2, 1, 0, 2};
  auto const values_original =
    col_data{1, 2, 0, 2, 1, -3, 2, 1, 2, 0, 5, 2, -3, -2, 2, 1, 2, 1, 2, 2};

  // key = 0: values = [2, 2, -3, -2, 2]
  // key = 1: values = [2, 0, 5, 2, 1]
  // key = 2: values = [-3, 1, 1, 2, 2]
  auto const keys   = cudf::slice(keys_original, {5, 20})[0];
  auto const values = cudf::slice(values_original, {5, 20})[0];

  auto const expected_keys      = int32s_col{0, 1, 2};
  auto const expected_histogram = [] {
    auto structs = [] {
      auto values = col_data{-3, -2, 2, 0, 1, 2, 5, -3, 1, 2};
      auto counts = int64s_col{1, 1, 3, 1, 1, 2, 1, 1, 2, 2};
      return structs_col{{values, counts}};
    }();
    return cudf::make_lists_column(
      3, int32s_col{0, 3, 7, 10}.release(), structs.release(), 0, rmm::device_buffer{});
  }();
  auto const [res_keys, res_histogram] =
    groupby_histogram(keys, values, cudf::aggregation::HISTOGRAM);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *res_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_histogram, *res_histogram);
}

TYPED_TEST(GroupbyHistogramTest, InputWithNulls)
{
  using col_data = cudf::test::fixed_width_column_wrapper<TypeParam, int>;
  using namespace cudf::test::iterators;
  auto constexpr null{0};

  // key = 0: values = [-3, null, 2, null, 2]
  // key = 1: values = [1, 2, null, 5, 2, -3, 1, 1]
  // key = 2: values = [null, 2, 0, -2, 2, null, 2]
  auto const keys = int32s_col{2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 1, 2, 0, 0, 1, 2, 2, 1, 0, 2};
  auto const values =
    col_data{{null, -3, 2, 1, 2, null, 0, 5, 2, null, -3, -2, 2, null, 1, 2, null, 1, 2, 2},
             nulls_at({0, 5, 9, 13, 16})};
  auto const expected_keys      = int32s_col{0, 1, 2};
  auto const expected_histogram = [] {
    auto structs = [] {
      auto values = col_data{{null, -3, 2, null, -3, 1, 2, 5, null, -2, 0, 2}, nulls_at({0, 3, 8})};
      auto counts = int64s_col{2, 1, 2, 1, 1, 3, 2, 1, 2, 1, 1, 3};
      return structs_col{{values, counts}};
    }();
    return cudf::make_lists_column(
      3, int32s_col{0, 3, 8, 12}.release(), structs.release(), 0, rmm::device_buffer{});
  }();
  auto const [res_keys, res_histogram] =
    groupby_histogram(keys, values, cudf::aggregation::HISTOGRAM);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *res_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_histogram, *res_histogram);
}

TYPED_TEST(GroupbyHistogramTest, SlicedInputWithNulls)
{
  using col_data = cudf::test::fixed_width_column_wrapper<TypeParam, int>;
  using namespace cudf::test::iterators;
  auto constexpr null{0};

  auto const keys_original =
    int32s_col{1, 0, 2, 2, 0, 2, 0, 2, 1, 1, 1, 2, 1, 1, 0, 1, 2, 0, 0, 1, 2, 2, 1, 0, 2, 0, 1, 2};
  auto const values_original =
    col_data{{null, 1,  1,  2, 1,    null, -3, 2,    1, 2, null, 0,    5, 2,
              null, -3, -2, 2, null, 1,    2,  null, 1, 2, 2,    null, 1, 2},
             nulls_at({0, 5, 10, 14, 18, 21, 25})};

  // key = 0: values = [-3, null, 2, null, 2]
  // key = 1: values = [1, 2, null, 5, 2, -3, 1, 1]
  // key = 2: values = [null, 2, 0, -2, 2, null, 2]
  auto const keys   = cudf::slice(keys_original, {5, 25})[0];
  auto const values = cudf::slice(values_original, {5, 25})[0];

  auto const expected_keys      = int32s_col{0, 1, 2};
  auto const expected_histogram = [] {
    auto structs = [] {
      auto values = col_data{{null, -3, 2, null, -3, 1, 2, 5, null, -2, 0, 2}, nulls_at({0, 3, 8})};
      auto counts = int64s_col{2, 1, 2, 1, 1, 3, 2, 1, 2, 1, 1, 3};
      return structs_col{{values, counts}};
    }();
    return cudf::make_lists_column(
      3, int32s_col{0, 3, 8, 12}.release(), structs.release(), 0, rmm::device_buffer{});
  }();
  auto const [res_keys, res_histogram] =
    groupby_histogram(keys, values, cudf::aggregation::HISTOGRAM);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *res_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_histogram, *res_histogram);
}

#if 0
TYPED_TEST(GroupbyMergeHistogramTest, MergeHistogram)
{
  using col_data    = cudf::test::fixed_width_column_wrapper<TypeParam>;
  using int64s_col  = cudf::test::fixed_width_column_wrapper<int64_t>;
  using structs_col = cudf::test::structs_column_wrapper;

  auto const agg = cudf::make_merge_histogram_aggregation<reduce_aggregation>();

  // Empty input.
  {
    auto const input = [] {
      auto child1 = col_data{};
      auto child2 = int64s_col{};
      return structs_col{{child1, child2}};
    }();
    auto const expected = [] {
      auto child1 = col_data{};
      auto child2 = int64s_col{};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test without nulls.
  {
    auto const input = [] {
      auto child1 = col_data{-3, 2, 1, 2, 0, 5, 2, -3, -2, 2, 1};
      auto child2 = int64s_col{2, 1, 1, 2, 4, 1, 2, 3, 5, 3, 4};
      return structs_col{{child1, child2}};
    }();

    auto const expected = [] {
      auto child1 = col_data{-3, -2, 0, 1, 2, 5};
      auto child2 = int64s_col{5, 5, 4, 5, 8, 1};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test without nulls, sliced input.
  {
    auto const input_original = [] {
      auto child1 = col_data{-3, 2, 1, 2, 0, 5, 2, -3, -2, 2, 1};
      auto child2 = int64s_col{2, 1, 1, 2, 4, 1, 2, 3, 5, 3, 4};
      return structs_col{{child1, child2}};
    }();
    auto const input = cudf::slice(input_original, {0, 7})[0];

    auto const expected = [] {
      auto child1 = col_data{-3, 0, 1, 2, 5};
      auto child2 = int64s_col{2, 4, 1, 5, 1};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test with nulls.
  using namespace cudf::test::iterators;
  auto constexpr null{0};
  {
    auto const input = [] {
      auto child1 = col_data{{-3, 2, null, 1, 2, null, 0, 5, null, 2, -3, null, -2, 2, 1, null},
                             nulls_at({2, 5, 8, 11, 15})};
      auto child2 = int64s_col{2, 1, 12, 1, 2, 11, 4, 1, 10, 2, 3, 15, 5, 3, 4, 19};
      return structs_col{{child1, child2}};
    }();

    auto const expected = [] {
      auto child1 = col_data{{null, -3, -2, 0, 1, 2, 5}, null_at(0)};
      auto child2 = int64s_col{67, 5, 5, 4, 5, 8, 1};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test with nulls, sliced input.
  {
    auto const input_original = [] {
      auto child1 = col_data{{-3, 2, null, 1, 2, null, 0, 5, null, 2, -3, null, -2, 2, 1, null},
                             nulls_at({2, 5, 8, 11, 15})};
      auto child2 = int64s_col{2, 1, 12, 1, 2, 11, 4, 1, 10, 2, 3, 15, 5, 3, 4, 19};
      return structs_col{{child1, child2}};
    }();
    auto const input = cudf::slice(input_original, {0, 9})[0];

    auto const expected = [] {
      auto child1 = col_data{{null, -3, 0, 1, 2, 5}, null_at(0)};
      auto child2 = int64s_col{33, 2, 4, 1, 3, 1};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

#endif
