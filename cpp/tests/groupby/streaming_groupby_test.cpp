/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <vector>

using namespace cudf::test;

static std::vector<cudf::size_type> const KEY_COL{0};

namespace {

void sort_and_compare(std::unique_ptr<cudf::table>& lhs_keys,
                      std::vector<cudf::groupby::aggregation_result>& lhs_results,
                      std::unique_ptr<cudf::table>& rhs_keys,
                      std::vector<cudf::groupby::aggregation_result>& rhs_results,
                      std::vector<cudf::null_order> const& null_prec = {cudf::null_order::AFTER})
{
  auto const lhs_order = cudf::sorted_order(lhs_keys->view(), {}, null_prec);
  auto const rhs_order = cudf::sorted_order(rhs_keys->view(), {}, null_prec);

  EXPECT_EQ(lhs_keys->num_rows(), rhs_keys->num_rows());

  ASSERT_EQ(lhs_results.size(), rhs_results.size());
  for (size_t r = 0; r < lhs_results.size(); ++r) {
    ASSERT_EQ(lhs_results[r].results.size(), rhs_results[r].results.size());
    for (size_t c = 0; c < lhs_results[r].results.size(); ++c) {
      auto const lhs_sorted =
        cudf::gather(cudf::table_view{{lhs_results[r].results[c]->view()}}, *lhs_order);
      auto const rhs_sorted =
        cudf::gather(cudf::table_view{{rhs_results[r].results[c]->view()}}, *rhs_order);

      auto const& lhs_col = lhs_sorted->get_column(0);
      auto const& rhs_col = rhs_sorted->get_column(0);

      if (lhs_col.type() == rhs_col.type()) {
        CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lhs_col, rhs_col);
      } else if (cudf::is_fixed_width(lhs_col.type()) && cudf::is_fixed_width(rhs_col.type())) {
        auto const lhs_cast = cudf::cast(lhs_col, rhs_col.type());
        CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*lhs_cast, rhs_col);
      } else {
        CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lhs_col, rhs_col);
      }
    }
  }
}

void verify_against_groupby(
  std::unique_ptr<cudf::table>& streaming_keys,
  std::vector<cudf::groupby::aggregation_result>& streaming_results,
  std::vector<cudf::table_view> const& batches,
  std::vector<cudf::size_type> const& key_indices,
  std::vector<cudf::groupby::streaming_aggregation_request> const& requests,
  cudf::null_policy null_handling = cudf::null_policy::EXCLUDE)
{
  auto const all_data = cudf::concatenate(batches);

  std::vector<cudf::column_view> key_cols;
  for (auto idx : key_indices) {
    key_cols.push_back(all_data->view().column(idx));
  }
  cudf::groupby::groupby reference_groupby{cudf::table_view{key_cols}, null_handling};

  std::vector<cudf::groupby::aggregation_request> ref_requests;
  for (auto const& req : requests) {
    cudf::groupby::aggregation_request ref_req;
    ref_req.values = all_data->view().column(req.column_index);
    for (auto const& agg : req.aggregations) {
      auto cloned = agg->clone();
      ref_req.aggregations.push_back(std::unique_ptr<cudf::groupby_aggregation>{
        dynamic_cast<cudf::groupby_aggregation*>(cloned.release())});
    }
    ref_requests.push_back(std::move(ref_req));
  }

  auto [ref_keys, ref_results] = reference_groupby.aggregate(ref_requests);

  sort_and_compare(streaming_keys, streaming_results, ref_keys, ref_results);
}

void check(std::unique_ptr<cudf::table>& keys,
           std::vector<cudf::groupby::aggregation_result>& results,
           cudf::table_view expect_keys,
           std::vector<cudf::column_view> const& expect_vals)
{
  auto const order       = cudf::sorted_order(keys->view());
  auto const sorted_keys = cudf::gather(keys->view(), *order);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expect_keys, sorted_keys->view());

  size_t val_idx = 0;
  for (auto& agg_res : results) {
    for (auto& col : agg_res.results) {
      auto const sorted = cudf::gather(cudf::table_view{{col->view()}}, *order);
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_vals[val_idx], sorted->get_column(0));
      ++val_idx;
    }
  }
}

cudf::groupby::streaming_aggregation_request make_req(
  cudf::size_type col_idx, std::vector<std::unique_ptr<cudf::groupby_aggregation>>&& aggs)
{
  cudf::groupby::streaming_aggregation_request req;
  req.column_index = col_idx;
  req.aggregations = std::move(aggs);
  return req;
}

std::vector<cudf::groupby::streaming_aggregation_request> single_agg_req(
  cudf::size_type col_idx, std::unique_ptr<cudf::groupby_aggregation>&& agg)
{
  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
  aggs.push_back(std::move(agg));
  reqs.push_back(make_req(col_idx, std::move(aggs)));
  return reqs;
}

}  // namespace

// -- Fixed-type tests --

struct StreamingGroupbyTest : public cudf::test::BaseFixture {};

TEST_F(StreamingGroupbyTest, SumTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  fixed_width_column_wrapper<K> keys1{1, 2, 3, 1};
  fixed_width_column_wrapper<V> vals1{10, 20, 30, 40};

  fixed_width_column_wrapper<K> keys2{2, 3, 1, 4};
  fixed_width_column_wrapper<V> vals2{5, 15, 25, 35};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MinMaxTwoBatches)
{
  using K = int32_t;
  using V = double;

  fixed_width_column_wrapper<K> keys1{1, 2, 1};
  fixed_width_column_wrapper<V> vals1{5.0, 2.0, 8.0};

  fixed_width_column_wrapper<K> keys2{1, 2, 3};
  fixed_width_column_wrapper<V> vals2{3.0, 9.0, 1.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
  aggs.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  aggs.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  reqs.push_back(make_req(1, std::move(aggs)));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, CountValidTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  fixed_width_column_wrapper<K> keys1{1, 2, 1, 2};
  fixed_width_column_wrapper<V> vals1{{10, 20, 30, 40}, {true, false, true, true}};

  fixed_width_column_wrapper<K> keys2{1, 2};
  fixed_width_column_wrapper<V> vals2{{50, 60}, {false, true}};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MeanTwoBatches)
{
  using K = int32_t;
  using V = double;

  fixed_width_column_wrapper<K> keys1{1, 2, 1};
  fixed_width_column_wrapper<V> vals1{10.0, 20.0, 30.0};

  fixed_width_column_wrapper<K> keys2{1, 2};
  fixed_width_column_wrapper<V> vals2{50.0, 40.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, ProductTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  fixed_width_column_wrapper<K> keys1{1, 2};
  fixed_width_column_wrapper<V> vals1{3, 5};

  fixed_width_column_wrapper<K> keys2{1, 2};
  fixed_width_column_wrapper<V> vals2{4, 2};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_product_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MaxMinOnIntegers)
{
  using K = int32_t;

  fixed_width_column_wrapper<K> keys1{1, 2};
  fixed_width_column_wrapper<int32_t> vals1{0, 1};

  fixed_width_column_wrapper<K> keys2{1, 2};
  fixed_width_column_wrapper<int32_t> vals2{1, 1};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
  aggs.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  aggs.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  reqs.push_back(make_req(1, std::move(aggs)));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

// -- Merge, release/restore, edge cases --

TEST_F(StreamingGroupbyTest, MergeTwoObjects)
{
  using K = int32_t;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  fixed_width_column_wrapper<K> keys1{1, 2, 1};
  fixed_width_column_wrapper<V> vals1{10, 20, 30};

  fixed_width_column_wrapper<K> keys2{2, 3};
  fixed_width_column_wrapper<V> vals2{40, 50};

  auto reqs1 = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker1(KEY_COL, reqs1);
  worker1.aggregate(cudf::table_view{{keys1, vals1}});

  auto reqs2 = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker2(KEY_COL, reqs2);
  worker2.aggregate(cudf::table_view{{keys2, vals2}});

  worker1.merge(worker2);
  auto [keys, results] = worker1.finalize();

  fixed_width_column_wrapper<K> ek{1, 2, 3};
  fixed_width_column_wrapper<R> ev{40, 60, 50};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, ReleaseAndRestore)
{
  using K = int32_t;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  fixed_width_column_wrapper<K> keys1{1, 2};
  fixed_width_column_wrapper<V> vals1{10, 20};

  fixed_width_column_wrapper<K> keys2{1, 3};
  fixed_width_column_wrapper<V> vals2{30, 40};

  auto reqs1 = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby first_pass(KEY_COL, reqs1);
  first_pass.aggregate(cudf::table_view{{keys1, vals1}});

  auto released = first_pass.release();

  auto reqs2 = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby restored(std::move(released), KEY_COL, reqs2);
  restored.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = restored.finalize();

  fixed_width_column_wrapper<K> ek{1, 2, 3};
  fixed_width_column_wrapper<R> ev{40, 20, 40};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, EmptyBatch)
{
  using K = int32_t;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  fixed_width_column_wrapper<K> keys1{1, 2};
  fixed_width_column_wrapper<V> vals1{10, 20};

  fixed_width_column_wrapper<K> keys_empty{};
  fixed_width_column_wrapper<V> vals_empty{};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(cudf::table_view{{keys_empty, vals_empty}});
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  auto [keys, results] = streaming_agg.finalize();

  fixed_width_column_wrapper<K> ek{1, 2};
  fixed_width_column_wrapper<R> ev{10, 20};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, SingleBatch)
{
  using K = int32_t;
  using V = int32_t;

  fixed_width_column_wrapper<K> keys1{1, 2, 1, 3, 2};
  fixed_width_column_wrapper<V> vals1{10, 20, 30, 40, 50};

  cudf::table_view batch1{{keys1, vals1}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(batch1);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, NewKeysInLaterBatches)
{
  using K = int32_t;
  using V = int32_t;

  fixed_width_column_wrapper<K> keys1{1, 2};
  fixed_width_column_wrapper<V> vals1{10, 20};

  fixed_width_column_wrapper<K> keys2{3, 4};
  fixed_width_column_wrapper<V> vals2{30, 40};

  fixed_width_column_wrapper<K> keys3{1, 4};
  fixed_width_column_wrapper<V> vals3{50, 60};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  cudf::table_view batch3{{keys3, vals3}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  streaming_agg.aggregate(batch3);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2, batch3}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MultipleRequestsOnDifferentColumns)
{
  using K = int32_t;

  fixed_width_column_wrapper<K> keys1{1, 2, 1};
  fixed_width_column_wrapper<int32_t> col_a1{10, 20, 30};
  fixed_width_column_wrapper<double> col_b1{1.0, 2.0, 3.0};

  fixed_width_column_wrapper<K> keys2{2, 1};
  fixed_width_column_wrapper<int32_t> col_a2{40, 50};
  fixed_width_column_wrapper<double> col_b2{4.0, 5.0};

  cudf::table_view batch1{{keys1, col_a1, col_b1}};
  cudf::table_view batch2{{keys2, col_a2, col_b2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  {
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
    aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    reqs.push_back(make_req(1, std::move(aggs)));
  }
  {
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
    aggs.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
    reqs.push_back(make_req(2, std::move(aggs)));
  }

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, FinalizeDoesNotModifyState)
{
  using K = int32_t;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  fixed_width_column_wrapper<K> keys1{1, 2};
  fixed_width_column_wrapper<V> vals1{10, 20};

  fixed_width_column_wrapper<K> keys2{1};
  fixed_width_column_wrapper<V> vals2{30};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});

  {
    auto [k1, r1] = streaming_agg.finalize();
  }

  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = streaming_agg.finalize();

  fixed_width_column_wrapper<K> ek{1, 2};
  fixed_width_column_wrapper<R> ev{40, 20};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

// -- Null key handling --

TEST_F(StreamingGroupbyTest, NullKeysExcluded)
{
  using K = int32_t;
  using V = int32_t;

  fixed_width_column_wrapper<K> keys1{{1, 2, 3}, {true, false, true}};
  fixed_width_column_wrapper<V> vals1{10, 20, 30};

  fixed_width_column_wrapper<K> keys2{{1, 2}, {true, false}};
  fixed_width_column_wrapper<V> vals2{40, 50};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, cudf::null_policy::EXCLUDE);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(
    keys, results, {batch1, batch2}, KEY_COL, reqs, cudf::null_policy::EXCLUDE);
}

TEST_F(StreamingGroupbyTest, NullKeysIncluded)
{
  using K = int32_t;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  fixed_width_column_wrapper<K> keys1{{1, 2, 3}, {true, false, true}};
  fixed_width_column_wrapper<V> vals1{10, 20, 30};

  fixed_width_column_wrapper<K> keys2{{1, 2}, {true, false}};
  fixed_width_column_wrapper<V> vals2{40, 50};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, cudf::null_policy::INCLUDE);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = streaming_agg.finalize();

  EXPECT_EQ(keys->num_rows(), 3);

  auto const order       = cudf::sorted_order(keys->view(), {}, {cudf::null_order::AFTER});
  auto const sorted_keys = cudf::gather(keys->view(), *order);
  auto const sorted_vals = cudf::gather(cudf::table_view{{results[0].results[0]->view()}}, *order);

  fixed_width_column_wrapper<K> ek{{1, 3, 2}, {true, true, false}};
  fixed_width_column_wrapper<R> ev{50, 30, 70};
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{{ek}}, sorted_keys->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(ev, sorted_vals->get_column(0));
}

TEST_F(StreamingGroupbyTest, AllNullKeysExcluded)
{
  using K = int32_t;
  using V = int32_t;

  fixed_width_column_wrapper<K> keys1{{1, 2}, {false, false}};
  fixed_width_column_wrapper<V> vals1{10, 20};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, cudf::null_policy::EXCLUDE);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  auto [keys, results] = streaming_agg.finalize();

  EXPECT_EQ(keys->num_rows(), 0);
  EXPECT_EQ(results[0].results[0]->size(), 0);
}

// -- Typed SUM test across multiple value types --

template <typename V>
struct StreamingGroupbySumTypedTest : public cudf::test::BaseFixture {};

using SumSupportedTypes =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>,
                     cudf::test::DurationTypes>;

TYPED_TEST_SUITE(StreamingGroupbySumTypedTest, SumSupportedTypes);

TYPED_TEST(StreamingGroupbySumTypedTest, TwoBatches)
{
  using K = int32_t;
  using V = TypeParam;

  fixed_width_column_wrapper<K> keys1{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  fixed_width_column_wrapper<V> vals1{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  fixed_width_column_wrapper<K> keys2{1, 2, 3};
  fixed_width_column_wrapper<V> vals2{10, 20, 30};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

// -- Typed MIN test across multiple value types --

template <typename V>
struct StreamingGroupbyMinTypedTest : public cudf::test::BaseFixture {};

using MinSupportedTypes =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>,
                     cudf::test::DurationTypes>;

TYPED_TEST_SUITE(StreamingGroupbyMinTypedTest, MinSupportedTypes);

TYPED_TEST(StreamingGroupbyMinTypedTest, TwoBatches)
{
  using K = int32_t;
  using V = TypeParam;

  fixed_width_column_wrapper<K> keys1{1, 2, 1};
  fixed_width_column_wrapper<V> vals1{5, 2, 8};

  fixed_width_column_wrapper<K> keys2{1, 2};
  fixed_width_column_wrapper<V> vals2{3, 9};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}
