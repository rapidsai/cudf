/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

using namespace cudf::test;

static std::vector<cudf::size_type> const KEY_COL{0};

namespace {

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

}  // namespace

struct StreamingGroupbyTest : public cudf::test::BaseFixture {};

TEST_F(StreamingGroupbyTest, SumTwoBatches)
{
  using K = int32_t;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  fixed_width_column_wrapper<K> keys1{1, 2, 3, 1};
  fixed_width_column_wrapper<V> vals1{10, 20, 30, 40};

  fixed_width_column_wrapper<K> keys2{2, 3, 1, 4};
  fixed_width_column_wrapper<V> vals2{5, 15, 25, 35};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
  aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  reqs.push_back(make_req(1, std::move(aggs)));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = streaming_agg.finalize();

  fixed_width_column_wrapper<K> ek{1, 2, 3, 4};
  fixed_width_column_wrapper<R> ev{75, 25, 45, 35};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, MinMaxTwoBatches)
{
  using K = int32_t;
  using V = double;

  fixed_width_column_wrapper<K> keys1{1, 2, 1};
  fixed_width_column_wrapper<V> vals1{5.0, 2.0, 8.0};

  fixed_width_column_wrapper<K> keys2{1, 2, 3};
  fixed_width_column_wrapper<V> vals2{3.0, 9.0, 1.0};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
  aggs.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  aggs.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  reqs.push_back(make_req(1, std::move(aggs)));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = streaming_agg.finalize();

  fixed_width_column_wrapper<K> ek{1, 2, 3};
  fixed_width_column_wrapper<V> e_min{3.0, 2.0, 1.0};
  fixed_width_column_wrapper<V> e_max{8.0, 9.0, 1.0};
  check(keys, results, cudf::table_view{{ek}}, {e_min, e_max});
}

TEST_F(StreamingGroupbyTest, CountValidTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  fixed_width_column_wrapper<K> keys1{1, 2, 1, 2};
  fixed_width_column_wrapper<V> vals1{{10, 20, 30, 40}, {true, false, true, true}};

  fixed_width_column_wrapper<K> keys2{1, 2};
  fixed_width_column_wrapper<V> vals2{{50, 60}, {false, true}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
  aggs.push_back(
    cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
  reqs.push_back(make_req(1, std::move(aggs)));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = streaming_agg.finalize();

  fixed_width_column_wrapper<K> ek{1, 2};
  fixed_width_column_wrapper<int64_t> ev{2, 2};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, MeanTwoBatches)
{
  using K = int32_t;
  using V = double;

  fixed_width_column_wrapper<K> keys1{1, 2, 1};
  fixed_width_column_wrapper<V> vals1{10.0, 20.0, 30.0};

  fixed_width_column_wrapper<K> keys2{1, 2};
  fixed_width_column_wrapper<V> vals2{50.0, 40.0};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
  aggs.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  reqs.push_back(make_req(1, std::move(aggs)));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = streaming_agg.finalize();

  fixed_width_column_wrapper<K> ek{1, 2};
  fixed_width_column_wrapper<double> ev{30.0, 30.0};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, ProductTwoBatches)
{
  using K = int32_t;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::PRODUCT>;

  fixed_width_column_wrapper<K> keys1{1, 2};
  fixed_width_column_wrapper<V> vals1{3, 5};

  fixed_width_column_wrapper<K> keys2{1, 2};
  fixed_width_column_wrapper<V> vals2{4, 2};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
  aggs.push_back(cudf::make_product_aggregation<cudf::groupby_aggregation>());
  reqs.push_back(make_req(1, std::move(aggs)));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = streaming_agg.finalize();

  fixed_width_column_wrapper<K> ek{1, 2};
  fixed_width_column_wrapper<R> ev{12, 10};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, MaxMinOnBooleans)
{
  using K = int32_t;

  fixed_width_column_wrapper<K> keys1{1, 2};
  fixed_width_column_wrapper<int32_t> vals1{0, 1};

  fixed_width_column_wrapper<K> keys2{1, 2};
  fixed_width_column_wrapper<int32_t> vals2{1, 1};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
  aggs.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  aggs.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  reqs.push_back(make_req(1, std::move(aggs)));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = streaming_agg.finalize();

  fixed_width_column_wrapper<K> ek{1, 2};
  fixed_width_column_wrapper<int32_t> e_max{1, 1};
  fixed_width_column_wrapper<int32_t> e_min{0, 1};
  check(keys, results, cudf::table_view{{ek}}, {e_max, e_min});
}

TEST_F(StreamingGroupbyTest, MergeTwoObjects)
{
  using K = int32_t;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  fixed_width_column_wrapper<K> keys1{1, 2, 1};
  fixed_width_column_wrapper<V> vals1{10, 20, 30};

  fixed_width_column_wrapper<K> keys2{2, 3};
  fixed_width_column_wrapper<V> vals2{40, 50};

  auto make_reqs = []() {
    std::vector<cudf::groupby::streaming_aggregation_request> reqs;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
    aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    reqs.push_back(make_req(1, std::move(aggs)));
    return reqs;
  };

  auto reqs1 = make_reqs();
  cudf::groupby::streaming_groupby worker1(KEY_COL, reqs1);
  worker1.aggregate(cudf::table_view{{keys1, vals1}});

  auto reqs2 = make_reqs();
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

  auto make_reqs = []() {
    std::vector<cudf::groupby::streaming_aggregation_request> reqs;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
    aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    reqs.push_back(make_req(1, std::move(aggs)));
    return reqs;
  };

  auto reqs1 = make_reqs();
  cudf::groupby::streaming_groupby first_pass(KEY_COL, reqs1);
  first_pass.aggregate(cudf::table_view{{keys1, vals1}});

  auto released = first_pass.release();

  auto reqs2 = make_reqs();
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

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
  aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  reqs.push_back(make_req(1, std::move(aggs)));

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
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  fixed_width_column_wrapper<K> keys1{1, 2, 1, 3, 2};
  fixed_width_column_wrapper<V> vals1{10, 20, 30, 40, 50};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
  aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  reqs.push_back(make_req(1, std::move(aggs)));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  auto [keys, results] = streaming_agg.finalize();

  fixed_width_column_wrapper<K> ek{1, 2, 3};
  fixed_width_column_wrapper<R> ev{40, 70, 40};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, NewKeysInLaterBatches)
{
  using K = int32_t;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  fixed_width_column_wrapper<K> keys1{1, 2};
  fixed_width_column_wrapper<V> vals1{10, 20};

  fixed_width_column_wrapper<K> keys2{3, 4};
  fixed_width_column_wrapper<V> vals2{30, 40};

  fixed_width_column_wrapper<K> keys3{1, 4};
  fixed_width_column_wrapper<V> vals3{50, 60};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
  aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  reqs.push_back(make_req(1, std::move(aggs)));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  streaming_agg.aggregate(cudf::table_view{{keys3, vals3}});
  auto [keys, results] = streaming_agg.finalize();

  fixed_width_column_wrapper<K> ek{1, 2, 3, 4};
  fixed_width_column_wrapper<R> ev{60, 20, 30, 100};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, MultipleRequestsOnDifferentColumns)
{
  using K = int32_t;
  using R = cudf::detail::target_type_t<int32_t, cudf::aggregation::SUM>;

  fixed_width_column_wrapper<K> keys1{1, 2, 1};
  fixed_width_column_wrapper<int32_t> col_a1{10, 20, 30};
  fixed_width_column_wrapper<double> col_b1{1.0, 2.0, 3.0};

  fixed_width_column_wrapper<K> keys2{2, 1};
  fixed_width_column_wrapper<int32_t> col_a2{40, 50};
  fixed_width_column_wrapper<double> col_b2{4.0, 5.0};

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
  streaming_agg.aggregate(cudf::table_view{{keys1, col_a1, col_b1}});
  streaming_agg.aggregate(cudf::table_view{{keys2, col_a2, col_b2}});
  auto [keys, results] = streaming_agg.finalize();

  fixed_width_column_wrapper<K> ek{1, 2};
  fixed_width_column_wrapper<R> e_sum{90, 60};
  fixed_width_column_wrapper<double> e_min{1.0, 2.0};
  check(keys, results, cudf::table_view{{ek}}, {e_sum, e_min});
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

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
  aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  reqs.push_back(make_req(1, std::move(aggs)));

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
