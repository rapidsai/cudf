/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

static std::vector<cudf::size_type> const KEY_COL{0};
static cudf::size_type constexpr DEFAULT_MAX_DISTINCT_KEYS = 1024;

namespace {

void sort_and_compare(std::unique_ptr<cudf::table>& lhs_keys,
                      std::vector<cudf::groupby::aggregation_result>& lhs_results,
                      std::unique_ptr<cudf::table>& rhs_keys,
                      std::vector<cudf::groupby::aggregation_result>& rhs_results,
                      std::vector<cudf::null_order> const& null_prec = {})
{
  auto const lhs_order = cudf::sorted_order(lhs_keys->view(), {}, null_prec);
  auto const rhs_order = cudf::sorted_order(rhs_keys->view(), {}, null_prec);

  EXPECT_EQ(lhs_keys->num_rows(), rhs_keys->num_rows());
  auto const lhs_sorted_keys = cudf::gather(lhs_keys->view(), *lhs_order);
  auto const rhs_sorted_keys = cudf::gather(rhs_keys->view(), *rhs_order);
  CUDF_TEST_EXPECT_TABLES_EQUAL(lhs_sorted_keys->view(), rhs_sorted_keys->view());

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
    ref_req.aggregations.push_back(std::unique_ptr<cudf::groupby_aggregation>{
      dynamic_cast<cudf::groupby_aggregation*>(req.aggregation->clone().release())});
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
  cudf::size_type col_idx, std::unique_ptr<cudf::groupby_aggregation>&& agg)
{
  cudf::groupby::streaming_aggregation_request req;
  req.column_index = col_idx;
  req.aggregation  = std::move(agg);
  return req;
}

std::vector<cudf::groupby::streaming_aggregation_request> single_agg_req(
  cudf::size_type col_idx, std::unique_ptr<cudf::groupby_aggregation>&& agg)
{
  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(col_idx, std::move(agg)));
  return reqs;
}

}  // namespace

struct StreamingGroupbyTest : public cudf::test::BaseFixture {};

TEST_F(StreamingGroupbyTest, MemoryResource)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{1, 2, 3, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> values{10, 20, 30, 40};
  cudf::table_view batch{{keys, values}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  auto mr   = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());

  cudf::groupby::streaming_groupby streaming_agg(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::EXCLUDE, mr);
  streaming_agg.aggregate(batch);

  EXPECT_GT(mr.get_bytes_counter().peak, 0);
}

TEST_F(StreamingGroupbyTest, SumTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 3, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30, 40};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 3, 1, 4};
  cudf::test::fixed_width_column_wrapper<V> vals2{5, 15, 25, 35};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MinMaxTwoBatches)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{5.0, 2.0, 8.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> vals2{3.0, 9.0, 1.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, CountValidTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{{10, 20, 30, 40}, {true, false, true, true}};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{{50, 60}, {false, true}};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MeanTwoBatches)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{10.0, 20.0, 30.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{50.0, 40.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, ProductTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{3, 5};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{4, 2};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_product_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MaxMinOnIntegers)
{
  using K = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> vals1{0, 1};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> vals2{1, 1};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MergeTwoObjects)
{
  using K = int32_t;
  using V = int32_t;
  using R = int64_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 3};
  cudf::test::fixed_width_column_wrapper<V> vals2{40, 50};

  auto reqs1 = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker1(KEY_COL, reqs1, DEFAULT_MAX_DISTINCT_KEYS);
  worker1.aggregate(cudf::table_view{{keys1, vals1}});

  auto reqs2 = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker2(KEY_COL, reqs2, DEFAULT_MAX_DISTINCT_KEYS);
  worker2.aggregate(cudf::table_view{{keys2, vals2}});

  worker1.merge(worker2);
  auto [keys, results] = worker1.finalize();

  cudf::test::fixed_width_column_wrapper<K> ek{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> ev{40, 60, 50};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, ConcurrentAggregate)
{
  using K = int32_t;
  using V = int32_t;

  constexpr int num_batches = 8;

  // Every batch re-hits keys 0 and 1 and introduces one key of its own, so concurrent calls
  // both collide on existing groups and discover new keys at the same time.
  std::vector<cudf::test::fixed_width_column_wrapper<K>> keys;
  std::vector<cudf::test::fixed_width_column_wrapper<V>> vals;
  keys.reserve(num_batches);
  vals.reserve(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    keys.emplace_back(std::initializer_list<K>{0, 1, static_cast<K>(i + 2)});
    vals.emplace_back(std::initializer_list<V>{1, 10, 100});
  }

  std::vector<cudf::table_view> batches;
  batches.reserve(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    batches.push_back(cudf::table_view{{keys[i], vals[i]}});
  }

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  auto const stream_device = cuda::device_ref{device};
  std::vector<std::unique_ptr<cuda::stream>> streams;
  streams.reserve(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    streams.push_back(std::make_unique<cuda::stream>(stream_device));
  }

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);

  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(num_batches);
  // `ready` lets the main thread wait until every worker is spinning, and `start` then releases
  // them together, so the aggregate() calls actually overlap.
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  threads.reserve(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    threads.emplace_back([&, i] {
      CUDF_CUDA_TRY(cudaSetDevice(device));
      ready.fetch_add(1, std::memory_order_relaxed);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      try {
        streaming_agg.aggregate(batches[i], *streams[i]);
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  while (ready.load(std::memory_order_relaxed) != num_batches) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  for (auto const& error : errors) {
    EXPECT_FALSE(error);
  }
  for (auto const& stream : streams) {
    stream->sync();
  }

  auto [out_keys, results] = streaming_agg.finalize();
  verify_against_groupby(out_keys, results, batches, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, EmptyBatch)
{
  using K = int32_t;
  using V = int32_t;
  using R = int64_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::fixed_width_column_wrapper<K> keys_empty{};
  cudf::test::fixed_width_column_wrapper<V> vals_empty{};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(cudf::table_view{{keys_empty, vals_empty}});
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  auto [keys, results] = streaming_agg.finalize();

  cudf::test::fixed_width_column_wrapper<K> ek{1, 2};
  cudf::test::fixed_width_column_wrapper<R> ev{10, 20};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, SingleBatch)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30, 40, 50};

  cudf::table_view batch1{{keys1, vals1}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, NewKeysInLaterBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::fixed_width_column_wrapper<K> keys2{3, 4};
  cudf::test::fixed_width_column_wrapper<V> vals2{30, 40};

  cudf::test::fixed_width_column_wrapper<K> keys3{1, 4};
  cudf::test::fixed_width_column_wrapper<V> vals3{50, 60};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  cudf::table_view batch3{{keys3, vals3}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  streaming_agg.aggregate(batch3);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2, batch3}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MultipleRequestsOnDifferentColumns)
{
  using K = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> col_a1{10, 20, 30};
  cudf::test::fixed_width_column_wrapper<double> col_b1{1.0, 2.0, 3.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> col_a2{40, 50};
  cudf::test::fixed_width_column_wrapper<double> col_b2{4.0, 5.0};

  cudf::table_view batch1{{keys1, col_a1, col_b1}};
  cudf::table_view batch2{{keys2, col_a2, col_b2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(2, cudf::make_min_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, FinalizeDoesNotModifyState)
{
  using K = int32_t;
  using V = int32_t;
  using R = int64_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::fixed_width_column_wrapper<K> keys2{1};
  cudf::test::fixed_width_column_wrapper<V> vals2{30};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});

  {
    auto [k1, r1] = streaming_agg.finalize();
  }

  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = streaming_agg.finalize();

  cudf::test::fixed_width_column_wrapper<K> ek{1, 2};
  cudf::test::fixed_width_column_wrapper<R> ev{40, 20};
  check(keys, results, cudf::table_view{{ek}}, {ev});
}

TEST_F(StreamingGroupbyTest, NullKeysExcluded)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{{1, 2, 3}, {true, false, true}};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30};

  cudf::test::fixed_width_column_wrapper<K> keys2{{1, 2}, {true, false}};
  cudf::test::fixed_width_column_wrapper<V> vals2{40, 50};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::EXCLUDE);
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
  using R = int64_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{{1, 2, 3}, {true, false, true}};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30};

  cudf::test::fixed_width_column_wrapper<K> keys2{{1, 2}, {true, false}};
  cudf::test::fixed_width_column_wrapper<V> vals2{40, 50};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::INCLUDE);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = streaming_agg.finalize();

  EXPECT_EQ(keys->num_rows(), 3);

  auto const order       = cudf::sorted_order(keys->view(), {}, {cudf::null_order::AFTER});
  auto const sorted_keys = cudf::gather(keys->view(), *order);
  auto const sorted_vals = cudf::gather(cudf::table_view{{results[0].results[0]->view()}}, *order);

  cudf::test::fixed_width_column_wrapper<K> ek{{1, 3, 2}, {true, true, false}};
  cudf::test::fixed_width_column_wrapper<R> ev{50, 30, 70};
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{{ek}}, sorted_keys->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(ev, sorted_vals->get_column(0));
}

TEST_F(StreamingGroupbyTest, AllNullKeysExcluded)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{{1, 2}, {false, false}};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::EXCLUDE);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  auto [keys, results] = streaming_agg.finalize();

  EXPECT_EQ(keys->num_rows(), 0);
  EXPECT_EQ(results[0].results[0]->size(), 0);
}

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

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> vals2{10, 20, 30};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, Decimal128SupportedAggregations)
{
  auto const type = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  EXPECT_TRUE(cudf::groupby::is_streaming_groupby_supported(type, cudf::aggregation::SUM));
  EXPECT_FALSE(cudf::groupby::is_streaming_groupby_supported(type, cudf::aggregation::MIN));
  EXPECT_FALSE(cudf::groupby::is_streaming_groupby_supported(type, cudf::aggregation::MAX));
}

struct StreamingGroupbyDecimal128Test
  : public cudf::test::BaseFixture,
    public testing::WithParamInterface<std::tuple<int32_t, cudf::null_policy>> {
  using fp128                      = cudf::test::fixed_point_column_wrapper<__int128_t>;
  static constexpr __int128_t high = static_cast<__int128_t>(1) << 80;
  static constexpr __int128_t word = static_cast<__int128_t>(1) << 64;
  static constexpr auto key_scale  = numeric::scale_type{-3};

  numeric::scale_type scale() const { return numeric::scale_type{std::get<0>(GetParam())}; }
  cudf::null_policy null_handling() const { return std::get<1>(GetParam()); }

  // Expected rows are sorted, with the null key first.
  void check_result(cudf::groupby::streaming_groupby const& streaming_agg,
                    cudf::table_view expected) const
  {
    auto [keys, results] = streaming_agg.finalize();
    EXPECT_EQ(keys->get_column(0).type().id(), cudf::type_id::DECIMAL128);
    EXPECT_EQ(keys->get_column(0).type().scale(), static_cast<int32_t>(key_scale));
    ASSERT_EQ(results.size(), 1);
    ASSERT_EQ(results[0].results.size(), 1);
    EXPECT_EQ(results[0].results[0]->type().id(), cudf::type_id::DECIMAL128);
    EXPECT_EQ(results[0].results[0]->type().scale(), static_cast<int32_t>(scale()));
    if (null_handling() == cudf::null_policy::EXCLUDE) {
      expected = cudf::slice(expected, {1, expected.num_rows()})[0];
    }
    auto const order  = cudf::sorted_order(keys->view());
    auto const sorted = cudf::gather(
      cudf::table_view{{keys->view().column(0), results[0].results[0]->view()}}, *order);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, sorted->view());
  }
};

TEST_P(StreamingGroupbyDecimal128Test, SumSlicedBatches)
{
  // Distinct keys share their low word. Values cross the low-word boundary in both directions.
  auto const a = high + 3;
  auto const b = 2 * high + 3;
  auto const c = -high + 3;
  fp128 keys1{
    {0, a, b, a, c, c, 0, 0}, {true, true, true, true, true, true, false, true}, key_scale};
  fp128 vals1{{0, high + word - 1, -high - word + 1, 2, 0, 0, high + 9, 0},
              {true, true, true, true, false, false, true, true},
              scale()};
  fp128 keys2{{0, a, b, b, c, 0, 0}, {true, true, true, true, true, false, true}, key_scale};
  fp128 vals2{
    {0, high + 3, -2, 0, 0, -high - 4, 0}, {true, true, true, false, false, true, true}, scale()};
  auto const batch1 = cudf::slice(cudf::table_view{{keys1, vals1}}, {1, 7})[0];
  auto const batch2 = cudf::slice(cudf::table_view{{keys2, vals2}}, {1, 6})[0];
  auto reqs         = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, null_handling());
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);

  fp128 expected_keys{{0, c, a, b}, {false, true, true, true}, key_scale};
  fp128 expected_vals{
    {5, 0, 2 * high + word + 4, -high - word - 1}, {true, false, true, true}, scale()};
  check_result(streaming_agg, cudf::table_view{{expected_keys, expected_vals}});
}

TEST_P(StreamingGroupbyDecimal128Test, SumMergeAndContinue)
{
  auto const a = -high + 7;
  auto const b = high + 7;
  auto const c = 2 * high + 7;
  auto const d = 3 * high + 7;
  fp128 keys1{{a, b, d, 0}, {true, true, true, false}, key_scale};
  fp128 vals1{{high + word - 1, -high - word + 1, 0, high + 1}, {true, true, false, true}, scale()};
  fp128 keys2{{a, b, c, d, 0}, {true, true, true, true, false}, key_scale};
  fp128 vals2{{2, -2, 2 * high + 3, 0, -high + 4}, {true, true, true, false, true}, scale()};
  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby destination(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, null_handling());
  cudf::groupby::streaming_groupby source(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, null_handling());
  destination.aggregate(cudf::table_view{{keys1, vals1}});
  source.aggregate(cudf::table_view{{keys2, vals2}});
  destination.merge(source);

  fp128 expected_keys{{0, a, b, c, d}, {false, true, true, true, true}, key_scale};
  fp128 expected_vals{{5, high + word + 1, -high - word - 1, 2 * high + 3, 0},
                      {true, true, true, true, false},
                      scale()};
  check_result(destination, cudf::table_view{{expected_keys, expected_vals}});
  fp128 source_vals{{-high + 4, 2, -2, 2 * high + 3, 0}, {true, true, true, true, false}, scale()};
  check_result(source, cudf::table_view{{expected_keys, source_vals}});

  // Merging and finalizing leave the source usable; further input does not affect the destination.
  source.aggregate(cudf::table_view{{keys1, vals1}});
  check_result(source, cudf::table_view{{expected_keys, expected_vals}});
  check_result(destination, cudf::table_view{{expected_keys, expected_vals}});
}

INSTANTIATE_TEST_SUITE_P(Decimal128,
                         StreamingGroupbyDecimal128Test,
                         testing::Combine(testing::Values(-2, 0, 2),
                                          testing::Values(cudf::null_policy::INCLUDE,
                                                          cudf::null_policy::EXCLUDE)));

TEST_F(StreamingGroupbyTest, Decimal128SumRepeatedKeys)
{
  using fp128                        = cudf::test::fixed_point_column_wrapper<__int128_t>;
  constexpr cudf::size_type num_rows = 65536;
  constexpr __int128_t high          = static_cast<__int128_t>(1) << 80;
  constexpr __int128_t word          = static_cast<__int128_t>(1) << 64;
  auto const scale                   = numeric::scale_type{-2};
  std::vector<__int128_t> key_data(num_rows), value_data(num_rows), sums(2, 0);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    auto const group = i % 2;
    key_data[i]      = (group + 1) * high + 3;
    value_data[i]    = high + word - 1 + i % 7;
    if (i % 3 == 0) { value_data[i] = -value_data[i]; }
    sums[group] += value_data[i];
  }
  fp128 input_keys(key_data.begin(), key_data.end(), scale);
  fp128 input_vals(value_data.begin(), value_data.end(), scale);
  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, num_rows);
  streaming_agg.aggregate(cudf::table_view{{input_keys, input_vals}});
  streaming_agg.aggregate(cudf::table_view{{input_keys, input_vals}});
  auto [keys, results] = streaming_agg.finalize();

  fp128 expected_keys{{high + 3, 2 * high + 3}, scale};
  fp128 expected_vals{{2 * sums[0], 2 * sums[1]}, scale};
  check(keys, results, cudf::table_view{{expected_keys}}, {expected_vals});
}

TEST_F(StreamingGroupbyTest, Decimal128SumNullsInLaterBatch)
{
  using fp128               = cudf::test::fixed_point_column_wrapper<__int128_t>;
  constexpr __int128_t high = static_cast<__int128_t>(1) << 80;
  auto const scale          = numeric::scale_type{-2};
  fp128 keys1{{high + 1}, scale};
  fp128 vals1{{high}, scale};
  fp128 keys2{{high + 1, high + 2}, scale};
  fp128 vals2{{0, 0}, {false, false}, scale};
  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(cudf::table_view{{keys1, vals1}});
  streaming_agg.aggregate(cudf::table_view{{keys2, vals2}});
  auto [keys, results] = streaming_agg.finalize();

  fp128 expected_vals{{high, 0}, {true, false}, scale};
  check(keys, results, cudf::table_view{{keys2}}, {expected_vals});

  cudf::groupby::streaming_groupby destination(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  cudf::groupby::streaming_groupby source(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  destination.aggregate(cudf::table_view{{keys1, vals1}});
  source.aggregate(cudf::table_view{{keys2, vals2}});
  destination.merge(source);
  auto [merged_keys, merged_results] = destination.finalize();
  check(merged_keys, merged_results, cudf::table_view{{keys2}}, {expected_vals});

  fp128 valid_vals{{2, -high - 7}, scale};
  fp128 expected_valid_vals{{high + 2, -high - 7}, scale};
  for (auto* worker : {&streaming_agg, &destination}) {
    worker->aggregate(cudf::table_view{{keys2, valid_vals}});
    auto [valid_keys, valid_results] = worker->finalize();
    check(valid_keys, valid_results, cudf::table_view{{keys2}}, {expected_valid_vals});
  }
}

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

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{5, 2, 8};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{3, 9};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, VarianceBasic)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 3, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{0, 1, 2, 3, 4};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{5, 6, 7, 8, 9};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_variance_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StdBasic)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 3, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{0, 1, 2, 3, 4};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{5, 6, 7, 8, 9};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_std_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, UnsupportedAggThrows)
{
  auto reqs = single_agg_req(1, cudf::make_collect_list_aggregation<cudf::groupby_aggregation>());
  EXPECT_THROW(cudf::groupby::streaming_groupby(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS),
               std::invalid_argument);
}

TEST_F(StreamingGroupbyTest, BatchExceedsMaxDistinctKeysThrows)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<V> vals{10, 20, 30, 40, 50};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, 3);
  EXPECT_THROW(streaming_agg.aggregate(cudf::table_view{{keys, vals}}), std::invalid_argument);
}

TEST_F(StreamingGroupbyTest, DisjointKeysAcrossBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::fixed_width_column_wrapper<K> keys2{3, 4};
  cudf::test::fixed_width_column_wrapper<V> vals2{30, 40};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, AllDuplicateKeysAcrossBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{30, 40};

  cudf::test::fixed_width_column_wrapper<K> keys3{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals3{50, 60};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  cudf::table_view batch3{{keys3, vals3}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  streaming_agg.aggregate(batch3);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2, batch3}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, SingleRowBatches)
{
  using K = int32_t;
  using V = int32_t;

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);

  std::vector<cudf::table_view> batches;
  std::vector<std::unique_ptr<cudf::column>> key_owners;
  std::vector<std::unique_ptr<cudf::column>> val_owners;

  for (int32_t i = 0; i < 10; ++i) {
    auto k = std::make_unique<cudf::column>(cudf::test::fixed_width_column_wrapper<K>{i % 3});
    auto v = std::make_unique<cudf::column>(cudf::test::fixed_width_column_wrapper<V>{i * 10});
    cudf::table_view batch{{k->view(), v->view()}};
    streaming_agg.aggregate(batch);
    key_owners.push_back(std::move(k));
    val_owners.push_back(std::move(v));
  }

  auto [keys, results] = streaming_agg.finalize();
  EXPECT_EQ(keys->num_rows(), 3);
}

// Regression test for staging-offset corruption: when a batch has internal duplicate keys,
// the last distinct key's canonical position in the staging buffer equals the distinct-key
// count, not the end of the written range.  Without the fix, the next batch writes at the
// wrong offset and corrupts the canonical entry for that key.
TEST_F(StreamingGroupbyTest, InternalDuplicatesDoNotCorruptStaging)
{
  using K = int32_t;
  using V = int32_t;

  // Batch 1: key 1 is a duplicate at position 1, so key 3 ends up at position 3
  // (_num_unique_keys == 3).  Without the fix, batch 2 writes at offset 3, overwriting
  // the canonical slot for key 3 with key 4, making them appear identical.
  cudf::test::fixed_width_column_wrapper<K> keys1{1, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 10, 20, 30};

  cudf::test::fixed_width_column_wrapper<K> keys2{4, 5};
  cudf::test::fixed_width_column_wrapper<V> vals2{40, 50};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, SumAndMeanOnSameColumn)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{10.0, 20.0, 30.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals2{40.0, 50.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

// Test that many small batches accumulate correctly within the fixed-capacity key table.
TEST_F(StreamingGroupbyTest, ManySmallBatches)
{
  using K = int32_t;
  using V = int32_t;

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  // 10 batches of 4 rows = 40 total rows; set max_distinct_keys=40 to fit all rows.
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, 40);

  std::vector<cudf::table_view> batches;
  std::vector<std::unique_ptr<cudf::column>> key_owners;
  std::vector<std::unique_ptr<cudf::column>> val_owners;

  for (int32_t b = 0; b < 10; ++b) {
    auto k = std::make_unique<cudf::column>(
      cudf::test::fixed_width_column_wrapper<K>{b % 8, (b + 1) % 8, (b + 2) % 8, (b + 3) % 8});
    auto v = std::make_unique<cudf::column>(
      cudf::test::fixed_width_column_wrapper<V>{b * 10, b * 10 + 1, b * 10 + 2, b * 10 + 3});
    cudf::table_view batch{{k->view(), v->view()}};
    streaming_agg.aggregate(batch);
    batches.push_back(batch);
    key_owners.push_back(std::move(k));
    val_owners.push_back(std::move(v));
  }

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, batches, KEY_COL, reqs);
}

// Test that exceeding distinct-key capacity throws.
TEST_F(StreamingGroupbyTest, ExceedsDistinctKeyCapacityThrows)
{
  using K = int32_t;
  using V = int32_t;

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  // max_distinct_keys=4: can hold at most 4 distinct keys.
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, 4);

  // Batch with 4 distinct keys fills distinct-key capacity.
  cudf::test::fixed_width_column_wrapper<K> k1{0, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> v1{10, 20, 30, 40};
  streaming_agg.aggregate(cudf::table_view{{k1, v1}});

  // Any further batch with a new distinct key exceeds distinct-key capacity (4 + 1 > 4).
  cudf::test::fixed_width_column_wrapper<K> k2{4};
  cudf::test::fixed_width_column_wrapper<V> v2{50};
  EXPECT_THROW(streaming_agg.aggregate(cudf::table_view{{k2, v2}}), cudf::logic_error);

  // The object is now invalidated: even an empty batch is rejected, while finalize() still
  // recovers the groups inserted before the failure.
  cudf::test::fixed_width_column_wrapper<K> k_empty{};
  cudf::test::fixed_width_column_wrapper<V> v_empty{};
  EXPECT_THROW(streaming_agg.aggregate(cudf::table_view{{k_empty, v_empty}}), cudf::logic_error);
  auto [keys, results] = streaming_agg.finalize();
  EXPECT_EQ(keys->num_rows(), 4);
}

// Test that sliced input columns with non-zero offsets work correctly.
TEST_F(StreamingGroupbyTest, SlicedInputColumns)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> full_keys{0, 1, 2, 3, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> full_vals{10, 20, 30, 40, 50, 60};

  // Slice to get a view with offset=2: keys={2,3,1,2}, vals={30,40,50,60}
  auto sliced = cudf::slice(cudf::table_view{{full_keys, full_vals}}, {2, 6});
  ASSERT_EQ(sliced[0].num_rows(), 4);

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(sliced[0]);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {sliced[0]}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, FinalizeAndRelease)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{1, 1, 2, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> values{{2, 4, 0, 0}, {true, true, false, false}};
  cudf::table_view const batch{{keys, values}};
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>()));
  cudf::groupby::streaming_groupby agg{KEY_COL, requests, DEFAULT_MAX_DISTINCT_KEYS};
  agg.aggregate(batch);
  std::ignore = agg.finalize();
  // Non-destructive finalization must still allow subsequent updates.
  agg.aggregate(batch);
  auto [expected_keys, expected_results] = agg.finalize();
  auto [out_keys, out_results]           = std::move(agg).finalize_and_release();
  sort_and_compare(out_keys, out_results, expected_keys, expected_results);
  EXPECT_EQ(agg.distinct_keys(), 0);
  EXPECT_THROW(agg.aggregate(batch), cudf::logic_error);
  EXPECT_THROW(static_cast<void>(agg.finalize()), cudf::logic_error);
  EXPECT_THROW(static_cast<void>(std::move(agg).finalize_and_release()), cudf::logic_error);
  cudf::groupby::streaming_groupby other{KEY_COL, requests, DEFAULT_MAX_DISTINCT_KEYS};
  other.aggregate(batch);
  EXPECT_THROW(other.merge(agg), cudf::logic_error);
  EXPECT_THROW(agg.merge(other), cudf::logic_error);
}

TEST_F(StreamingGroupbyTest, FinalizeAndReleaseDifferentStream)
{
  cudf::test::strings_column_wrapper keys1{{"a", "b", "c"}, {true, false, true}};
  cudf::test::strings_column_wrapper keys2{"a", "d"};
  cudf::test::fixed_width_column_wrapper<int32_t> values1{{2, 4, 0}, {true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> values2{3, 5};
  cudf::table_view const batch1{{keys1, values1}};
  cudf::table_view const batch2{{keys2, values2}};
  auto requests = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  cuda::stream aggregate_stream{cuda::device_ref{device}};
  cuda::stream finalize_stream{cuda::device_ref{device}};
  cudf::get_default_stream().sync();
  cudf::groupby::streaming_groupby agg{
    KEY_COL, requests, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::INCLUDE};
  agg.aggregate(batch1, aggregate_stream);
  agg.aggregate(batch2, aggregate_stream);
  aggregate_stream.sync();

  auto [out_keys, out_results] = std::move(agg).finalize_and_release(finalize_stream);
  finalize_stream.sync();
  verify_against_groupby(
    out_keys, out_results, {batch1, batch2}, KEY_COL, requests, cudf::null_policy::INCLUDE);
  EXPECT_EQ(agg.distinct_keys(), 0);
}

TEST_F(StreamingGroupbyTest, FinalizeAndReleaseEmpty)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{};
  cudf::test::fixed_width_column_wrapper<int32_t> values{};
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  cudf::groupby::streaming_groupby agg{KEY_COL, requests, DEFAULT_MAX_DISTINCT_KEYS};
  agg.aggregate(cudf::table_view{{keys, values}});
  EXPECT_THROW(static_cast<void>(std::move(agg).finalize_and_release()), cudf::logic_error);

  cudf::test::fixed_width_column_wrapper<int32_t> nonempty_keys{1};
  cudf::test::fixed_width_column_wrapper<int32_t> nonempty_values{2};
  agg.aggregate(cudf::table_view{{nonempty_keys, nonempty_values}});
  EXPECT_EQ(agg.distinct_keys(), 1);
}

TEST_F(StreamingGroupbyTest, FinalizeAndReleaseBeforeAggregateThrows)
{
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  cudf::groupby::streaming_groupby agg{KEY_COL, requests, DEFAULT_MAX_DISTINCT_KEYS};
  EXPECT_THROW(static_cast<void>(std::move(agg).finalize_and_release()), cudf::logic_error);
  EXPECT_EQ(agg.distinct_keys(), 0);
  EXPECT_THROW(static_cast<void>(agg.finalize()), cudf::logic_error);
}

// Test that finalize() before any aggregate() throws.
TEST_F(StreamingGroupbyTest, FinalizeBeforeAggregateThrows)
{
  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  EXPECT_THROW(static_cast<void>(streaming_agg.finalize()), cudf::logic_error);
}

// Test merge with MEAN aggregation (compound: SUM + COUNT intermediates).
TEST_F(StreamingGroupbyTest, MergeMeanTwoBatches)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{10.0, 20.0, 30.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1, 3};
  cudf::test::fixed_width_column_wrapper<V> vals2{40.0, 50.0, 60.0};

  auto reqs1 = single_agg_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker1(KEY_COL, reqs1, DEFAULT_MAX_DISTINCT_KEYS);
  worker1.aggregate(cudf::table_view{{keys1, vals1}});

  auto reqs2 = single_agg_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker2(KEY_COL, reqs2, DEFAULT_MAX_DISTINCT_KEYS);
  worker2.aggregate(cudf::table_view{{keys2, vals2}});

  worker1.merge(worker2);
  auto [keys, results] = worker1.finalize();

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs1);
}

// Test merge with COUNT_VALID aggregation (counts must be summed, not incremented).
TEST_F(StreamingGroupbyTest, MergeCountTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30, 40};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{50, 60, 70};

  auto reqs1 = single_agg_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
  cudf::groupby::streaming_groupby worker1(KEY_COL, reqs1, DEFAULT_MAX_DISTINCT_KEYS);
  worker1.aggregate(cudf::table_view{{keys1, vals1}});

  auto reqs2 = single_agg_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
  cudf::groupby::streaming_groupby worker2(KEY_COL, reqs2, DEFAULT_MAX_DISTINCT_KEYS);
  worker2.aggregate(cudf::table_view{{keys2, vals2}});

  worker1.merge(worker2);
  auto [keys, results] = worker1.finalize();

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs1);
}

// Test merge with VARIANCE aggregation (compound: SUM_OF_SQUARES + SUM + COUNT intermediates).
TEST_F(StreamingGroupbyTest, MergeVarianceTwoBatches)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 3, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{0, 1, 2, 3, 4};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{5, 6, 7, 8, 9};

  auto reqs1 = single_agg_req(1, cudf::make_variance_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker1(KEY_COL, reqs1, DEFAULT_MAX_DISTINCT_KEYS);
  worker1.aggregate(cudf::table_view{{keys1, vals1}});

  auto reqs2 = single_agg_req(1, cudf::make_variance_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby worker2(KEY_COL, reqs2, DEFAULT_MAX_DISTINCT_KEYS);
  worker2.aggregate(cudf::table_view{{keys2, vals2}});

  worker1.merge(worker2);
  auto [keys, results] = worker1.finalize();

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs1);
}

TEST_F(StreamingGroupbyTest, SumOfSquaresBasic)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{3.0, 4.0, 5.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals2{6.0, 7.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, M2Basic)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{1.0, 2.0, 3.0, 4.0};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals2{5.0, 6.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_m2_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StdWithNullValues)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> vals1{{1.0, 2.0, 3.0, 4.0}, {true, true, false, true}};

  cudf::test::fixed_width_column_wrapper<K> keys2{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals2{{5.0, 6.0, 7.0}, {true, false, true}};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_std_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

// ===== String key tests =====

TEST_F(StreamingGroupbyTest, StringKeySumTwoBatches)
{
  using V = int32_t;

  cudf::test::strings_column_wrapper keys1{"a", "b", "c", "a"};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30, 40};

  cudf::test::strings_column_wrapper keys2{"b", "c", "a", "d"};
  cudf::test::fixed_width_column_wrapper<V> vals2{5, 15, 25, 35};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StringKeyNonAsciiUtf8)
{
  using V = int32_t;

  cudf::test::strings_column_wrapper keys1{"αλφα", "βητα", "γαμμα", "αλφα"};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30, 40};

  cudf::test::strings_column_wrapper keys2{"βητα", "δελτα", "αλφα", "🙂"};
  cudf::test::fixed_width_column_wrapper<V> vals2{5, 15, 25, 35};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StringKeyMinMaxTwoBatches)
{
  cudf::test::strings_column_wrapper keys1{"cat", "dog", "cat"};
  cudf::test::fixed_width_column_wrapper<double> vals1{5.0, 2.0, 8.0};

  cudf::test::strings_column_wrapper keys2{"cat", "dog", "bird"};
  cudf::test::fixed_width_column_wrapper<double> vals2{3.0, 9.0, 1.0};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  std::vector<cudf::groupby::streaming_aggregation_request> reqs;
  reqs.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  reqs.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StringKeyManySmallBatches)
{
  using V = int32_t;

  std::vector<std::string> key_universe{"alpha", "beta", "gamma", "delta"};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);

  std::vector<cudf::table_view> batches;
  std::vector<std::unique_ptr<cudf::column>> key_owners;
  std::vector<std::unique_ptr<cudf::column>> val_owners;

  for (int32_t b = 0; b < 8; ++b) {
    auto k = std::make_unique<cudf::column>(
      cudf::test::strings_column_wrapper{key_universe[b % 4], key_universe[(b + 1) % 4]});
    auto v =
      std::make_unique<cudf::column>(cudf::test::fixed_width_column_wrapper<V>{b * 10, b * 10 + 1});
    cudf::table_view batch{{k->view(), v->view()}};
    streaming_agg.aggregate(batch);
    batches.push_back(batch);
    key_owners.push_back(std::move(k));
    val_owners.push_back(std::move(v));
  }

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, batches, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StringKeyDisjointBatches)
{
  using V = int32_t;

  cudf::test::strings_column_wrapper keys1{"x", "y"};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::strings_column_wrapper keys2{"z", "w"};
  cudf::test::fixed_width_column_wrapper<V> vals2{30, 40};

  cudf::test::strings_column_wrapper keys3{"x", "w"};
  cudf::test::fixed_width_column_wrapper<V> vals3{50, 60};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};
  cudf::table_view batch3{{keys3, vals3}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  streaming_agg.aggregate(batch3);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2, batch3}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StringKeyNullKeysExcluded)
{
  using V = int32_t;

  cudf::test::strings_column_wrapper keys1(std::initializer_list<std::string>{"a", "b", "c"},
                                           std::initializer_list<bool>{true, false, true});
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20, 30};

  cudf::test::strings_column_wrapper keys2(std::initializer_list<std::string>{"a", "b"},
                                           std::initializer_list<bool>{true, false});
  cudf::test::fixed_width_column_wrapper<V> vals2{40, 50};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(
    KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::EXCLUDE);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(
    keys, results, {batch1, batch2}, KEY_COL, reqs, cudf::null_policy::EXCLUDE);
}

TEST_F(StreamingGroupbyTest, StringKeyMerge)
{
  using V = int32_t;

  cudf::test::strings_column_wrapper keys1{"a", "b"};
  cudf::test::fixed_width_column_wrapper<V> vals1{10, 20};

  cudf::test::strings_column_wrapper keys2{"b", "c"};
  cudf::test::fixed_width_column_wrapper<V> vals2{30, 40};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby obj1(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  obj1.aggregate(cudf::table_view{{keys1, vals1}});

  cudf::groupby::streaming_groupby obj2(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  obj2.aggregate(cudf::table_view{{keys2, vals2}});

  obj1.merge(obj2);
  auto [keys, results] = obj1.finalize();

  verify_against_groupby(keys,
                         results,
                         {cudf::table_view{{keys1, vals1}}, cudf::table_view{{keys2, vals2}}},
                         KEY_COL,
                         reqs);
}

TEST_F(StreamingGroupbyTest, CountAllTwoBatches)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys1{1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals1{{10, 20, 30}, {true, false, true}};

  cudf::test::fixed_width_column_wrapper<K> keys2{2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals2{{40, 50}, {false, true}};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_count_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, MultiColumnKeys)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> k1a{1, 1, 2};
  cudf::test::fixed_width_column_wrapper<K> k1b{10, 20, 10};
  cudf::test::fixed_width_column_wrapper<V> v1{100, 200, 300};

  cudf::test::fixed_width_column_wrapper<K> k2a{1, 2};
  cudf::test::fixed_width_column_wrapper<K> k2b{10, 10};
  cudf::test::fixed_width_column_wrapper<V> v2{400, 500};

  cudf::table_view batch1{{k1a, k1b, v1}};
  cudf::table_view batch2{{k2a, k2b, v2}};

  std::vector<cudf::size_type> key_cols{0, 1};
  auto reqs = single_agg_req(2, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(key_cols, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, key_cols, reqs);
}

// A single batch larger than `max_distinct_keys` cannot be encoded because transient
// batch values (`max_distinct_keys + row_idx`) would collide with stored dense IDs.
TEST_F(StreamingGroupbyTest, BatchExceedingMaxDistinctKeysThrows)
{
  using K = int32_t;
  using V = int32_t;

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, 3);

  cudf::test::fixed_width_column_wrapper<K> k{0, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> v{10, 20, 30, 40};
  EXPECT_THROW(streaming_agg.aggregate(cudf::table_view{{k, v}}), std::invalid_argument);
}

// Cumulative input rows are not bounded by `max_distinct_keys` — only cumulative distinct
// keys are.  Re-feeding the same batch many times keeps distinct_keys constant and
// must never throw, regardless of how many cumulative rows have been processed.
TEST_F(StreamingGroupbyTest, CumulativeRowsCanExceedMaxDistinctKeys)
{
  using K = int32_t;
  using V = int32_t;

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, 3);

  cudf::test::fixed_width_column_wrapper<K> k{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<V> v{10, 20, 30};
  cudf::table_view batch{{k, v}};

  // Five repeats: 15 cumulative rows >> max_distinct_keys=3, distinct_keys stays at 3.
  streaming_agg.aggregate(batch);
  streaming_agg.aggregate(batch);
  streaming_agg.aggregate(batch);
  streaming_agg.aggregate(batch);
  streaming_agg.aggregate(batch);

  EXPECT_EQ(streaming_agg.distinct_keys(), 3);

  auto [keys, results] = streaming_agg.finalize();
  verify_against_groupby(keys, results, {batch, batch, batch, batch, batch}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, StructKeySumTwoBatches)
{
  using V = int32_t;

  // Struct key: {int, int}
  cudf::test::fixed_width_column_wrapper<int32_t> s1a{1, 1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> s1b{10, 20, 10};
  auto keys1 = cudf::test::structs_column_wrapper{{s1a, s1b}};
  cudf::test::fixed_width_column_wrapper<V> vals1{100, 200, 300};

  cudf::test::fixed_width_column_wrapper<int32_t> s2a{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> s2b{10, 10};
  auto keys2 = cudf::test::structs_column_wrapper{{s2a, s2b}};
  cudf::test::fixed_width_column_wrapper<V> vals2{400, 500};

  cudf::table_view batch1{{keys1, vals1}};
  cudf::table_view batch2{{keys2, vals2}};

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, DEFAULT_MAX_DISTINCT_KEYS);
  streaming_agg.aggregate(batch1);
  streaming_agg.aggregate(batch2);
  auto [keys, results] = streaming_agg.finalize();

  verify_against_groupby(keys, results, {batch1, batch2}, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, DenseAggregationsNullableFlatAndNestedKeys)
{
  // Stable key nullability layouts isolate launcher delegation from cross-batch key schema
  // transitions.
  cudf::test::fixed_width_column_wrapper<int32_t> keys1{{1, 2, 1}, {true, true, true}};
  cudf::test::fixed_width_column_wrapper<int32_t> keys2{{2, 3, 0}, {true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> values1{{10, 0, 30}, {true, false, true}};
  cudf::test::fixed_width_column_wrapper<int32_t> values2{40, 50, 60};
  cudf::test::structs_column_wrapper nested_keys1{{keys1}, {true, true, true}};
  cudf::test::structs_column_wrapper nested_keys2{{keys2}, {true, true, false}};

  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{1, 2, 3};
  cudf::test::structs_column_wrapper expected_nested_keys{{expected_keys}};
  cudf::test::fixed_width_column_wrapper<int64_t> expected_sum{40, 40, 50};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_min{10, 40, 50};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_max{30, 40, 50};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_count{2, 1, 1};

  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE)));

  // Multiple kinds exercise concurrent updates by streaming's dense-output kernel.
  // Explicit expectations make the regression independent of the reference implementation.
  struct {
    char const* name;
    cudf::column_view first_keys;
    cudf::column_view second_keys;
    cudf::column_view expected_keys;
  } const key_parameters[] = {
    {"top-level", keys1, keys2, expected_keys},
    {"nested", nested_keys1, nested_keys2, expected_nested_keys},
  };
  for (auto const& [name, first_keys, second_keys, expected_key_view] : key_parameters) {
    SCOPED_TRACE(name);
    cudf::table_view const batch1{{first_keys, values1}};
    cudf::table_view const batch2{{second_keys, values2}};
    // The null-key row must contribute to none of the expected aggregates.
    cudf::groupby::streaming_groupby streaming_agg(
      KEY_COL, requests, DEFAULT_MAX_DISTINCT_KEYS, cudf::null_policy::EXCLUDE);
    streaming_agg.aggregate(batch1);
    streaming_agg.aggregate(batch2);
    auto [keys, results] = streaming_agg.finalize();

    ASSERT_EQ(results.size(), requests.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
      ASSERT_EQ(results[i].results.size(), 1) << "aggregation request " << i;
    }
    check(keys,
          results,
          cudf::table_view{{expected_key_view}},
          {expected_sum, expected_min, expected_max, expected_count});
  }
}

TEST_F(StreamingGroupbyTest, CSRAbsentGroupsAndNullableValues)
{
  constexpr int rows = 1 << 16;
  std::vector<int32_t> key_data(rows), value_data(rows);
  std::vector<bool> validity(rows);
  std::vector<int64_t> sums(5, 0), squared_sums(5, 0);
  std::vector<int32_t> minima(5, std::numeric_limits<int32_t>::max());
  std::vector<int32_t> maxima(5, std::numeric_limits<int32_t>::lowest());
  std::vector<int32_t> all_counts(5, 0), valid_counts(5, 0);
  auto const accumulate = [&](int group, int32_t value, bool valid, int repetitions) {
    all_counts[group] += repetitions;
    if (!valid) { return; }
    valid_counts[group] += repetitions;
    sums[group] += static_cast<int64_t>(value) * repetitions;
    squared_sums[group] += static_cast<int64_t>(value) * value * repetitions;
    minima[group] = std::min(minima[group], value);
    maxima[group] = std::max(maxima[group], value);
  };
  for (int i = 0; i < rows; ++i) {
    key_data[i]   = i % 4;
    value_data[i] = i % 11 - 5;
    validity[i]   = i % 4 != 0 && i % 7 != 0;
    accumulate(key_data[i], value_data[i], validity[i], 1);
  }
  cudf::test::fixed_width_column_wrapper<int32_t> first_keys(key_data.begin(), key_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> first_values(
    value_data.begin(), value_data.end(), validity.begin());
  // Later batches contain no rows for the all-null group 0. Its output must remain null.
  for (int i = 0; i < rows; ++i) {
    key_data[i] = 1 + i % 4;
    accumulate(key_data[i], value_data[i], true, 3);
  }
  cudf::test::fixed_width_column_wrapper<int32_t> later_keys(key_data.begin(), key_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> later_values(value_data.begin(),
                                                               value_data.end());

  std::vector<bool> expected_validity(5);
  std::vector<double> means(5, 0);
  for (int group = 0; group < 5; ++group) {
    expected_validity[group] = valid_counts[group] != 0;
    if (expected_validity[group]) {
      means[group] = static_cast<double>(sums[group]) / valid_counts[group];
    }
  }
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<int64_t> expected_sums(
    sums.begin(), sums.end(), expected_validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_minima(
    minima.begin(), minima.end(), expected_validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_maxima(
    maxima.begin(), maxima.end(), expected_validity.begin());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_squared_sums(
    squared_sums.begin(), squared_sums.end(), expected_validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_all_counts(all_counts.begin(),
                                                                      all_counts.end());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_valid_counts(valid_counts.begin(),
                                                                        valid_counts.end());
  cudf::test::fixed_width_column_wrapper<double> expected_means(
    means.begin(), means.end(), expected_validity.begin());

  // Keep the compound pair together: the existing finalizer uses the direct SUM's
  // validity to produce MEAN's null mask, including the all-null group 0.
  for (bool compound : {false, true}) {
    SCOPED_TRACE(compound ? "SUM and MEAN" : "direct aggregations");
    std::vector<cudf::groupby::streaming_aggregation_request> requests;
    requests.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
    if (compound) {
      requests.push_back(make_req(1, cudf::make_mean_aggregation<cudf::groupby_aggregation>()));
    } else {
      requests.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
      requests.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
      requests.push_back(
        make_req(1, cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>()));
      requests.push_back(make_req(
        1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE)));
      requests.push_back(make_req(
        1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE)));
    }
    cudf::table_view first{{first_keys, first_values}};
    cudf::table_view later{{later_keys, later_values}};
    cudf::groupby::streaming_groupby destination(KEY_COL, requests, rows);
    cudf::groupby::streaming_groupby source(KEY_COL, requests, rows);
    destination.aggregate(first);
    destination.aggregate(later);
    source.aggregate(later);
    destination.merge(source);
    destination.aggregate(later);
    auto [keys, results] = destination.finalize();
    if (compound) {
      check(keys, results, cudf::table_view{{expected_keys}}, {expected_sums, expected_means});
    } else {
      check(keys,
            results,
            cudf::table_view{{expected_keys}},
            {expected_sums,
             expected_minima,
             expected_maxima,
             expected_squared_sums,
             expected_all_counts,
             expected_valid_counts});
    }
  }
}

TEST_F(StreamingGroupbyTest, CSRNullableNestedKeysAndExcludedBatch)
{
  constexpr int rows = 1 << 16;
  std::vector<int32_t> key_data(rows), value_data(rows, 3);
  std::vector<bool> validity(rows), all_null(rows, false);
  for (int i = 0; i < rows; ++i) {
    key_data[i] = i % 9;
    validity[i] = i % 7 != 0;
  }
  cudf::test::fixed_width_column_wrapper<int32_t> child(
    key_data.begin(), key_data.end(), validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> excluded_child(
    key_data.begin(), key_data.end(), all_null.begin());
  cudf::test::structs_column_wrapper keys{{child}, validity};
  cudf::test::structs_column_wrapper excluded_keys{{excluded_child}, all_null};
  cudf::test::fixed_width_column_wrapper<int32_t> values(value_data.begin(), value_data.end());
  auto requests = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::table_view batch{{keys, values}};
  cudf::table_view excluded{{excluded_keys, values}};
  for (auto policy : {cudf::null_policy::EXCLUDE, cudf::null_policy::INCLUDE}) {
    cudf::groupby::streaming_groupby streaming(KEY_COL, requests, rows, policy);
    streaming.aggregate(batch);
    streaming.aggregate(excluded);
    streaming.aggregate(batch);
    auto [out_keys, results] = streaming.finalize();
    verify_against_groupby(out_keys, results, {batch, excluded, batch}, KEY_COL, requests, policy);
  }
}

TEST_F(StreamingGroupbyTest, ConcurrentAggregateCSR)
{
  using K = int32_t;
  using V = int32_t;

  constexpr int num_batches = 8;
  constexpr int rows        = 1 << 16;

  // Every batch re-hits keys 0 and 1 and introduces one key of its own, so concurrent calls
  // both collide on existing groups and discover new keys at the same time.
  std::vector<cudf::test::fixed_width_column_wrapper<K>> keys;
  std::vector<cudf::test::fixed_width_column_wrapper<V>> vals;
  keys.reserve(num_batches);
  vals.reserve(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    std::vector<K> host_keys(rows);
    std::vector<V> host_values(rows);
    for (int row = 0; row < rows; ++row) {
      host_keys[row]   = row % 3 == 2 ? i + 2 : row % 3;
      host_values[row] = row % 3 == 0 ? 1 : (row % 3 == 1 ? 10 : 100);
    }
    keys.emplace_back(host_keys.begin(), host_keys.end());
    vals.emplace_back(host_values.begin(), host_values.end());
  }

  std::vector<cudf::table_view> batches;
  batches.reserve(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    batches.push_back(cudf::table_view{{keys[i], vals[i]}});
  }

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  auto const stream_device = cuda::device_ref{device};
  std::vector<std::unique_ptr<cuda::stream>> streams;
  streams.reserve(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    streams.push_back(std::make_unique<cuda::stream>(stream_device));
  }

  auto reqs = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming_agg(KEY_COL, reqs, rows);

  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(num_batches);
  // `ready` lets the main thread wait until every worker is spinning, and `start` then releases
  // them together, so the aggregate() calls actually overlap.
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  threads.reserve(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    threads.emplace_back([&, i] {
      CUDF_CUDA_TRY(cudaSetDevice(device));
      ready.fetch_add(1, std::memory_order_relaxed);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      try {
        streaming_agg.aggregate(batches[i], *streams[i]);
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  while (ready.load(std::memory_order_relaxed) != num_batches) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  for (auto const& error : errors) {
    EXPECT_FALSE(error);
  }
  for (auto const& stream : streams) {
    stream->sync();
  }

  auto [out_keys, results] = streaming_agg.finalize();
  verify_against_groupby(out_keys, results, batches, KEY_COL, reqs);
}

TEST_F(StreamingGroupbyTest, CSRDecimal128SlicedNullableBatches)
{
  using fp128                      = cudf::test::fixed_point_column_wrapper<__int128_t>;
  constexpr cudf::size_type rows   = 1 << 16;
  constexpr cudf::size_type offset = 7;
  constexpr __int128_t high        = static_cast<__int128_t>(1) << 80;
  constexpr __int128_t word        = static_cast<__int128_t>(1) << 64;
  auto const scale                 = numeric::scale_type{-3};
  std::vector<int32_t> key_data(2 * rows + 2 * offset);
  std::vector<__int128_t> value_data(key_data.size()), sums(4, 0);
  std::vector<bool> validity(key_data.size());
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(key_data.size()); ++i) {
    auto const group = i % 4;
    key_data[i]      = group;
    value_data[i]    = high + word - 1 + i % 7;
    if (i % 3 == 0) { value_data[i] = -value_data[i]; }
    // Group 2 becomes valid only in the later batch; group 3 stays all-null.
    validity[i] = group != 3 && (group != 2 || i >= offset + rows) && i % 11 != 0;
    if (i >= offset && i < offset + 2 * rows && validity[i]) { sums[group] += value_data[i]; }
  }
  cudf::test::fixed_width_column_wrapper<int32_t> input_keys(key_data.begin(), key_data.end());
  fp128 input_values(value_data.begin(), value_data.end(), validity.begin(), scale);
  auto const batches = cudf::slice(cudf::table_view{{input_keys, input_values}},
                                   {offset, offset + rows, offset + rows, offset + 2 * rows});
  auto requests      = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::streaming_groupby streaming(KEY_COL, requests, rows);
  for (auto const& batch : batches) {
    streaming.aggregate(batch);
  }
  auto [keys, results] = streaming.finalize();
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2, 3};
  fp128 expected_values(sums.begin(), sums.end(), {true, true, true, false}, scale);
  check(keys, results, cudf::table_view{{expected_keys}}, {expected_values});
}

TEST_F(StreamingGroupbyTest, CSRDictionaryValuesSlicedNullableBatches)
{
  constexpr cudf::size_type rows   = 1 << 16;
  constexpr cudf::size_type offset = 5;
  std::vector<int32_t> key_data(2 * rows + 2 * offset), value_data(key_data.size());
  std::vector<bool> validity(key_data.size());
  std::vector<int64_t> sums(4, 0);
  std::vector<int32_t> minima(4, 100), maxima(4, -100), counts(4, 0);
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(key_data.size()); ++i) {
    auto const group = i % 4;
    key_data[i]      = group;
    value_data[i]    = i % 17 - 8;
    validity[i]      = group != 3 && (group != 2 || i >= offset + rows) && i % 7 != 0;
    if (i >= offset && i < offset + 2 * rows && validity[i]) {
      sums[group] += value_data[i];
      minima[group] = std::min(minima[group], value_data[i]);
      maxima[group] = std::max(maxima[group], value_data[i]);
      ++counts[group];
    }
  }
  cudf::test::fixed_width_column_wrapper<int32_t> input_keys(key_data.begin(), key_data.end());
  cudf::test::dictionary_column_wrapper<int32_t> input_values(
    value_data.begin(), value_data.end(), validity.begin());
  auto const batches = cudf::slice(cudf::table_view{{input_keys, input_values}},
                                   {offset, offset + rows, offset + rows, offset + 2 * rows});
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE)));
  cudf::groupby::streaming_groupby streaming(KEY_COL, requests, rows);
  for (auto const& batch : batches) {
    streaming.aggregate(batch);
  }
  auto [keys, results] = streaming.finalize();
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2, 3};
  std::vector<bool> const expected_validity{true, true, true, false};
  cudf::test::fixed_width_column_wrapper<int64_t> expected_sums(
    sums.begin(), sums.end(), expected_validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_minima(
    minima.begin(), minima.end(), expected_validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_maxima(
    maxima.begin(), maxima.end(), expected_validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_counts(counts.begin(), counts.end());
  check(keys,
        results,
        cudf::table_view{{expected_keys}},
        {expected_sums, expected_minima, expected_maxima, expected_counts});
}

TEST_F(StreamingGroupbyTest, CSRFloatingCompoundNullableBatches)
{
  constexpr cudf::size_type rows   = 1 << 16;
  constexpr cudf::size_type offset = 11;
  std::vector<int32_t> key_data(2 * rows + 2 * offset);
  std::vector<double> value_data(key_data.size());
  std::vector<bool> validity(key_data.size());
  std::vector<long double> sums(5, 0), squared_sums(5, 0);
  std::vector<int32_t> counts(5, 0);
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(key_data.size()); ++i) {
    // The hot group requires long-segment reduction; the remaining groups are shorter.
    auto const group = i % 16 < 12 ? 0 : i % 16 - 11;
    key_data[i]      = group;
    value_data[i]    = (i % 29 - 14) * 0.125;
    // Group 2 becomes valid later, group 3 stays all-null, and group 4 has one valid row.
    validity[i] =
      group != 3 && (group != 2 || i >= offset + rows) && (group != 4 || i == 15) && i % 13 != 0;
    if (i >= offset && i < offset + 2 * rows && validity[i]) {
      auto const value = static_cast<long double>(value_data[i]);
      sums[group] += value;
      squared_sums[group] += value * value;
      ++counts[group];
    }
  }
  cudf::test::fixed_width_column_wrapper<int32_t> input_keys(key_data.begin(), key_data.end());
  cudf::test::fixed_width_column_wrapper<double> input_values(
    value_data.begin(), value_data.end(), validity.begin());
  auto const batches = cudf::slice(cudf::table_view{{input_keys, input_values}},
                                   {offset, offset + rows, offset + rows, offset + 2 * rows});

  std::vector<double> variances(5, 0), standard_deviations(5, 0);
  std::vector<bool> expected_validity(5, false);
  for (int group = 0; group < 5; ++group) {
    if (counts[group] <= 1) { continue; }
    auto const m2              = squared_sums[group] - sums[group] * sums[group] / counts[group];
    variances[group]           = static_cast<double>(m2 / (counts[group] - 1));
    standard_deviations[group] = std::sqrt(variances[group]);
    expected_validity[group]   = true;
  }
  ASSERT_EQ(counts[3], 0);
  ASSERT_EQ(counts[4], 1);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2, 3, 4};

  // Test each compound request independently; both reduce SUM, SUM_OF_SQUARES and COUNT_VALID.
  for (bool standard_deviation : {false, true}) {
    SCOPED_TRACE(standard_deviation ? "STD" : "VARIANCE");
    auto requests = single_agg_req(
      1,
      standard_deviation ? cudf::make_std_aggregation<cudf::groupby_aggregation>(1)
                         : cudf::make_variance_aggregation<cudf::groupby_aggregation>(1));
    cudf::groupby::streaming_groupby streaming(KEY_COL, requests, rows);
    for (auto const& batch : batches) {
      streaming.aggregate(batch);
    }
    auto [keys, results] = streaming.finalize();
    auto const& expected = standard_deviation ? standard_deviations : variances;
    cudf::test::fixed_width_column_wrapper<double> expected_values(
      expected.begin(), expected.end(), expected_validity.begin());
    check(keys, results, cudf::table_view{{expected_keys}}, {expected_values});
  }
}

TEST_F(StreamingGroupbyTest, AdaptiveGrowthPreservesNullableIdentities)
{
  constexpr int rows   = 1 << 16;
  constexpr int groups = 140;
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> batch_keys, batch_values;
  std::vector<int32_t> expected_key_data(groups),
    minima(groups, std::numeric_limits<int32_t>::max());
  std::vector<int32_t> maxima(groups, std::numeric_limits<int32_t>::lowest());
  std::vector<int64_t> sums(groups, 0), products(groups, 1);
  std::vector<bool> expected_validity(groups, false);
  auto const accumulate = [&](int group, int32_t value, bool valid) {
    if (!valid) { return; }
    sums[group] += value;
    products[group] *= value;
    minima[group]            = std::min(minima[group], value);
    maxima[group]            = std::max(maxima[group], value);
    expected_validity[group] = true;
  };

  // Prefix lengths 31, 37 and 70 cross partial mask words during successive resizes.
  int first_group = 0;
  for (int last_group : {31, 37, 70, groups}) {
    std::vector<int32_t> keys(rows), values(rows);
    std::vector<bool> validity(rows);
    for (int row = 0; row < rows; ++row) {
      auto const group = first_group + row % (last_group - first_group);
      keys[row]        = group;
      values[row]      = group % 2 == 0 ? 1 : -1;
      validity[row]    = group >= 4 && group % 7 != 0;
      accumulate(group, values[row], validity[row]);
    }
    batch_keys.emplace_back(keys.begin(), keys.end());
    batch_values.emplace_back(values.begin(), values.end(), validity.begin());
    first_group = last_group;
  }
  // Null payloads must retain reduction identities through every resize. Replacing them
  // with zero would corrupt negative MAX, positive MIN, and every PRODUCT below.
  cudf::test::fixed_width_column_wrapper<int32_t> late_keys{0, 1, 2, 7, 63};
  cudf::test::fixed_width_column_wrapper<int32_t> late_values{-5, 5, 3, -2, 4};
  for (auto const& [group, value] :
       std::vector<std::pair<int32_t, int32_t>>{{0, -5}, {1, 5}, {2, 3}, {7, -2}, {63, 4}}) {
    accumulate(group, value, true);
  }
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(1, cudf::make_product_aggregation<cudf::groupby_aggregation>()));

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  cuda::stream first_stream{cuda::device_ref{device}};
  cuda::stream second_stream{cuda::device_ref{device}};
  cudf::get_default_stream().sync();
  cudf::groupby::streaming_groupby streaming(KEY_COL, requests, rows);
  for (std::size_t batch = 0; batch < batch_keys.size(); ++batch) {
    streaming.aggregate(cudf::table_view{{batch_keys[batch], batch_values[batch]}},
                        batch % 2 == 0 ? first_stream : second_stream);
  }
  streaming.aggregate(cudf::table_view{{late_keys, late_values}}, first_stream);
  auto [keys, results] = streaming.finalize(second_stream);
  first_stream.sync();
  second_stream.sync();

  EXPECT_EQ(streaming.distinct_keys(), groups);
  for (int group = 0; group < groups; ++group) {
    expected_key_data[group] = group;
  }
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys(expected_key_data.begin(),
                                                                expected_key_data.end());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_sums(
    sums.begin(), sums.end(), expected_validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_minima(
    minima.begin(), minima.end(), expected_validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_maxima(
    maxima.begin(), maxima.end(), expected_validity.begin());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_products(
    products.begin(), products.end(), expected_validity.begin());
  check(keys,
        results,
        cudf::table_view{{expected_keys}},
        {expected_sums, expected_minima, expected_maxima, expected_products});
}

TEST_F(StreamingGroupbyTest, AdaptiveGrowthAfterCrossStreamSnapshot)
{
  constexpr int initial_groups = 1 << 15;
  constexpr int final_groups   = 1 << 16;
  std::vector<int32_t> key_data(final_groups), value_data(final_groups, 1);
  for (int group = 0; group < final_groups; ++group) {
    key_data[group] = group;
  }
  cudf::test::fixed_width_column_wrapper<int32_t> input_keys(key_data.begin(), key_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> input_values(value_data.begin(),
                                                               value_data.end());
  cudf::table_view const full_batch{{input_keys, input_values}};
  auto const first_batch = cudf::slice(full_batch, {0, initial_groups}).front();
  auto requests          = single_agg_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  cuda::stream update_stream{cuda::device_ref{device}};
  cuda::stream snapshot_stream{cuda::device_ref{device}};
  cudf::get_default_stream().sync();
  cudf::groupby::streaming_groupby streaming(KEY_COL, requests, final_groups);
  streaming.aggregate(first_batch, update_stream);
  auto [snapshot_keys, snapshot_results] = streaming.finalize(snapshot_stream);
  // No external dependency orders this growth after the queued snapshot copies.
  streaming.aggregate(full_batch, update_stream);
  auto [later_snapshot_keys, later_snapshot_results] = streaming.finalize(snapshot_stream);
  // Existing-key updates must also wait for snapshot reads when no resize provides a barrier.
  streaming.aggregate(first_batch, update_stream);
  auto [keys, results] = streaming.finalize(update_stream);
  snapshot_stream.sync();
  update_stream.sync();

  std::vector<int32_t> snapshot_counts(initial_groups, 1), later_snapshot_counts(final_groups, 1);
  std::vector<int32_t> final_counts(final_groups, 1);
  std::fill(later_snapshot_counts.begin(), later_snapshot_counts.begin() + initial_groups, 2);
  std::fill(final_counts.begin(), final_counts.begin() + initial_groups, 3);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_snapshot_counts(snapshot_counts.begin(),
                                                                           snapshot_counts.end());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_final_counts(final_counts.begin(),
                                                                        final_counts.end());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_later_snapshot_counts(
    later_snapshot_counts.begin(), later_snapshot_counts.end());
  check(snapshot_keys,
        snapshot_results,
        cudf::table_view{{first_batch.column(0)}},
        {expected_snapshot_counts});
  check(later_snapshot_keys,
        later_snapshot_results,
        cudf::table_view{{input_keys}},
        {expected_later_snapshot_counts});
  check(keys, results, cudf::table_view{{input_keys}}, {expected_final_counts});
}

TEST_F(StreamingGroupbyTest, AdaptiveGrowthAfterMergeSourceRead)
{
  constexpr int initial_groups = 33;
  constexpr int final_groups   = 129;
  std::vector<int32_t> key_data(final_groups), initial_values(initial_groups, 10);
  std::vector<int32_t> later_values(final_groups, 2);
  for (int group = 0; group < final_groups; ++group) {
    key_data[group] = group;
  }
  cudf::test::fixed_width_column_wrapper<int32_t> first_keys(key_data.begin(),
                                                             key_data.begin() + initial_groups);
  cudf::test::fixed_width_column_wrapper<int32_t> first_values(initial_values.begin(),
                                                               initial_values.end());
  cudf::test::fixed_width_column_wrapper<int32_t> later_keys(key_data.begin(), key_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> later_column(later_values.begin(),
                                                               later_values.end());
  cudf::test::fixed_width_column_wrapper<int32_t> seed_key{0}, seed_value{1};
  auto requests = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  cuda::stream source_stream{cuda::device_ref{device}};
  cuda::stream merge_stream{cuda::device_ref{device}};
  cudf::get_default_stream().sync();
  cudf::groupby::streaming_groupby source(KEY_COL, requests, final_groups);
  cudf::groupby::streaming_groupby destination(KEY_COL, requests, final_groups);
  source.aggregate(cudf::table_view{{first_keys, first_values}}, source_stream);
  destination.aggregate(cudf::table_view{{seed_key, seed_value}}, merge_stream);
  destination.merge(source, merge_stream);
  // The merge call has returned, but its source read may still be in flight on merge_stream.
  source.aggregate(cudf::table_view{{later_keys, later_column}}, source_stream);
  auto [first_merged_keys, first_merged_results] = destination.finalize(merge_stream);
  destination.merge(source, merge_stream);
  // Source writes must follow the second merge's read even when the allocation stays unchanged.
  source.aggregate(cudf::table_view{{first_keys, first_values}}, source_stream);
  auto [source_keys, source_results] = source.finalize(source_stream);
  auto [keys, results]               = destination.finalize(merge_stream);
  source_stream.sync();
  merge_stream.sync();

  std::vector<int64_t> first_merged_sums(initial_groups, 10), merged_sums(final_groups, 2);
  std::vector<int64_t> source_sums(final_groups, 2);
  first_merged_sums[0] = 11;
  std::fill(merged_sums.begin(), merged_sums.begin() + initial_groups, 22);
  merged_sums[0] = 23;
  std::fill(source_sums.begin(), source_sums.begin() + initial_groups, 22);
  cudf::test::fixed_width_column_wrapper<int64_t> expected_first_merged(first_merged_sums.begin(),
                                                                        first_merged_sums.end());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_merged(merged_sums.begin(),
                                                                  merged_sums.end());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_source(source_sums.begin(),
                                                                  source_sums.end());
  check(first_merged_keys,
        first_merged_results,
        cudf::table_view{{first_keys}},
        {expected_first_merged});
  check(keys, results, cudf::table_view{{later_keys}}, {expected_merged});
  check(source_keys, source_results, cudf::table_view{{later_keys}}, {expected_source});
}

TEST_F(StreamingGroupbyTest, AdaptiveGrowthAfterAllExcludedBatch)
{
  constexpr int rows   = 1 << 16;
  constexpr int groups = 70;
  std::vector<int32_t> excluded_key_data(rows, 0), excluded_values(rows, 123);
  std::vector<bool> excluded_validity(rows, false);
  std::vector<int32_t> key_data(groups), value_data(groups, 3);
  std::vector<bool> value_validity(groups, true);
  value_validity[0] = false;
  for (int group = 0; group < groups; ++group) {
    key_data[group] = group;
  }
  cudf::test::fixed_width_column_wrapper<int32_t> excluded_keys(
    excluded_key_data.begin(), excluded_key_data.end(), excluded_validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> excluded_column(excluded_values.begin(),
                                                                  excluded_values.end());
  cudf::test::fixed_width_column_wrapper<int32_t> valid_keys(key_data.begin(), key_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> valid_values(
    value_data.begin(), value_data.end(), value_validity.begin());
  cudf::table_view const excluded{{excluded_keys, excluded_column}};
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE)));

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  cuda::stream first_stream{cuda::device_ref{device}};
  cuda::stream second_stream{cuda::device_ref{device}};
  cudf::get_default_stream().sync();
  cudf::groupby::streaming_groupby streaming(KEY_COL, requests, rows, cudf::null_policy::EXCLUDE);
  streaming.aggregate(excluded, first_stream);
  EXPECT_EQ(streaming.distinct_keys(), 0);
  auto [empty_keys, empty_results] = streaming.finalize(first_stream);
  streaming.aggregate(cudf::table_view{{valid_keys, valid_values}}, second_stream);
  streaming.aggregate(excluded, first_stream);
  auto [keys, results] = streaming.finalize(second_stream);
  first_stream.sync();
  second_stream.sync();

  cudf::test::fixed_width_column_wrapper<int32_t> expected_empty_keys{}, expected_empty_counts{};
  cudf::test::fixed_width_column_wrapper<int64_t> expected_empty_sums{};
  check(empty_keys,
        empty_results,
        cudf::table_view{{expected_empty_keys}},
        {expected_empty_sums, expected_empty_counts});
  std::vector<int64_t> sums(groups, 3);
  std::vector<int32_t> counts(groups, 1);
  counts[0] = 0;
  cudf::test::fixed_width_column_wrapper<int64_t> expected_sums(
    sums.begin(), sums.end(), value_validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_counts(counts.begin(), counts.end());
  check(keys, results, cudf::table_view{{valid_keys}}, {expected_sums, expected_counts});
}

TEST_F(StreamingGroupbyTest, GrowingBatchSizeWithStableGroups)
{
  constexpr int groups        = 17;
  constexpr int largest_batch = 1 << 18;
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> batch_keys, batch_values;
  std::vector<int32_t> expected_key_data(groups), counts(groups, 0);
  std::vector<int32_t> minima(groups, std::numeric_limits<int32_t>::max());
  std::vector<int32_t> maxima(groups, std::numeric_limits<int32_t>::lowest());
  std::vector<int64_t> sums(groups, 0);
  for (int rows : {groups, 1 << 16, largest_batch, largest_batch}) {
    std::vector<int32_t> keys(rows), values(rows);
    for (int row = 0; row < rows; ++row) {
      auto const group = row % groups;
      keys[row]        = group;
      values[row]      = rows == groups ? group - 8 : row % 5 - 2;
      sums[group] += values[row];
      ++counts[group];
      minima[group] = std::min(minima[group], values[row]);
      maxima[group] = std::max(maxima[group], values[row]);
    }
    batch_keys.emplace_back(keys.begin(), keys.end());
    batch_values.emplace_back(values.begin(), values.end());
  }
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(1, cudf::make_count_aggregation<cudf::groupby_aggregation>()));
  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  cuda::stream first_stream{cuda::device_ref{device}};
  cuda::stream second_stream{cuda::device_ref{device}};
  cudf::get_default_stream().sync();
  cudf::groupby::streaming_groupby streaming(KEY_COL, requests, largest_batch);
  for (std::size_t batch = 0; batch < batch_keys.size(); ++batch) {
    streaming.aggregate(cudf::table_view{{batch_keys[batch], batch_values[batch]}},
                        batch % 2 == 0 ? first_stream : second_stream);
    EXPECT_EQ(streaming.distinct_keys(), groups);
  }
  auto [keys, results] = streaming.finalize(second_stream);
  first_stream.sync();
  second_stream.sync();
  for (int group = 0; group < groups; ++group) {
    expected_key_data[group] = group;
  }
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys(expected_key_data.begin(),
                                                                expected_key_data.end());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_sums(sums.begin(), sums.end());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_minima(minima.begin(), minima.end());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_maxima(maxima.begin(), maxima.end());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_counts(counts.begin(), counts.end());
  check(keys,
        results,
        cudf::table_view{{expected_keys}},
        {expected_sums, expected_minima, expected_maxima, expected_counts});
}

TEST_F(StreamingGroupbyTest, PersistentPartialBudgetGrowthSelfMergeAndRelease)
{
  constexpr int initial_rows = 16;
  constexpr int large_rows   = 1 << 16;
  constexpr int groups       = 4;
  std::vector<int32_t> key_data(large_rows), value_data(large_rows, 1);
  for (int row = 0; row < large_rows; ++row) {
    key_data[row] = row % groups;
  }
  cudf::test::fixed_width_column_wrapper<int32_t> input_keys(key_data.begin(), key_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> input_values(value_data.begin(),
                                                               value_data.end());
  cudf::table_view const large_batch{{input_keys, input_values}};
  auto const initial_batch = cudf::slice(large_batch, {0, initial_rows}).front();
  auto requests            = single_agg_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  cuda::stream update_stream{cuda::device_ref{device}};
  cuda::stream snapshot_stream{cuda::device_ref{device}};
  cudf::get_default_stream().sync();
  cudf::groupby::streaming_groupby streaming(KEY_COL, requests, large_rows);
  streaming.aggregate(initial_batch, update_stream);
  // No new keys: only the larger batch can trigger expansion from one partial to many.
  streaming.aggregate(large_batch, update_stream);
  auto [snapshot_keys, snapshot_results] = streaming.finalize(snapshot_stream);
  // Every source shard must be read once before any self-merge writes its destination shard0.
  streaming.merge(streaming, update_stream);
  streaming.aggregate(initial_batch, snapshot_stream);
  // Honor the destructive API's caller-ordering requirement. The terminal reduction runs on
  // a different stream from the original aggregation metadata allocation. COUNT has no mask
  // and therefore does not add incidental nullable-finalization synchronization.
  update_stream.sync();
  snapshot_stream.sync();
  auto [keys, results] = std::move(streaming).finalize_and_release(snapshot_stream);
  EXPECT_EQ(streaming.distinct_keys(), 0);
  snapshot_stream.sync();

  constexpr int32_t snapshot_count = (initial_rows + large_rows) / groups;
  constexpr int32_t final_count    = 2 * snapshot_count + initial_rows / groups;
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_snapshot{
    snapshot_count, snapshot_count, snapshot_count, snapshot_count};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_final{
    final_count, final_count, final_count, final_count};
  check(snapshot_keys, snapshot_results, cudf::table_view{{expected_keys}}, {expected_snapshot});
  check(keys, results, cudf::table_view{{expected_keys}}, {expected_final});
}

TEST_F(StreamingGroupbyTest, CSRAdjacentColumnsSlicedNullableBatches)
{
  constexpr cudf::size_type rows   = 1 << 16;
  constexpr cudf::size_type offset = 7;
  std::vector<int32_t> key_data(2 * rows + 2 * offset);
  std::vector<std::vector<int32_t>> value_data(4, std::vector<int32_t>(key_data.size()));
  std::vector<std::vector<bool>> validity(2, std::vector<bool>(key_data.size()));
  std::vector<std::vector<int32_t>> maxima(
    4, std::vector<int32_t>(5, std::numeric_limits<int32_t>::lowest()));
  std::vector<std::vector<bool>> expected_validity(4, std::vector<bool>(5, false));
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(key_data.size()); ++i) {
    auto const later = i >= offset + rows;
    auto const group = i % 4 + static_cast<int>(later);
    key_data[i]      = group;
    validity[0][i]   = group != 3 && (group != 2 || later) && i % 7 != 0;
    validity[1][i]   = group != 1 && group != 4 && i % 11 != 0;
    for (int column = 0; column < 4; ++column) {
      value_data[column][i] = -1 - (column + 1) * (i % 23 + 1) - (later ? 0 : 100);
      if (i >= offset && i < offset + 2 * rows && (column >= 2 || validity[column][i])) {
        maxima[column][group]            = std::max(maxima[column][group], value_data[column][i]);
        expected_validity[column][group] = true;
      }
    }
  }
  cudf::test::fixed_width_column_wrapper<int32_t> input_keys(key_data.begin(), key_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> first_values(
    value_data[0].begin(), value_data[0].end(), validity[0].begin());
  cudf::test::fixed_width_column_wrapper<int32_t> second_values(
    value_data[1].begin(), value_data[1].end(), validity[1].begin());
  cudf::test::fixed_width_column_wrapper<int32_t> third_values(value_data[2].begin(),
                                                               value_data[2].end());
  cudf::test::fixed_width_column_wrapper<int32_t> fourth_values(value_data[3].begin(),
                                                                value_data[3].end());
  auto const batches = cudf::slice(
    cudf::table_view{{input_keys, first_values, second_values, third_values, fourth_values}},
    {offset, offset + rows, offset + rows, offset + 2 * rows});
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  for (int column = 1; column <= 4; ++column) {
    requests.push_back(make_req(column, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  }
  cudf::groupby::streaming_groupby streaming(KEY_COL, requests, rows);
  for (auto const& batch : batches) {
    streaming.aggregate(batch);
  }
  auto [keys, results] = streaming.finalize();
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_first(
    maxima[0].begin(), maxima[0].end(), expected_validity[0].begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_second(
    maxima[1].begin(), maxima[1].end(), expected_validity[1].begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_third(maxima[2].begin(),
                                                                 maxima[2].end());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_fourth(maxima[3].begin(),
                                                                  maxima[3].end());
  check(keys,
        results,
        cudf::table_view{{expected_keys}},
        {expected_first, expected_second, expected_third, expected_fourth});
}

TEST_F(StreamingGroupbyTest, PeerRunsNoncontiguousTargetsNullsAndTails)
{
  constexpr cudf::size_type rows   = 270;
  constexpr cudf::size_type offset = 5;
  constexpr int groups             = 5;
  std::vector<int32_t> key_data(2 * rows + 2 * offset);
  std::vector<std::vector<int32_t>> integer_data(3, std::vector<int32_t>(key_data.size()));
  std::vector<double> floating_data(key_data.size());
  std::vector<int64_t> wide_data(key_data.size());
  std::vector<bool> key_validity(key_data.size());
  std::vector<std::vector<bool>> validity(3, std::vector<bool>(key_data.size()));
  std::vector<int32_t> max_first(groups, std::numeric_limits<int32_t>::lowest());
  std::vector<int32_t> max_second = max_first;
  std::vector<double> max_floating(groups, -std::numeric_limits<double>::infinity());
  std::vector<int64_t> sums(groups, 0), squared_sums(groups, 0), wide_sums(groups, 0);
  std::vector<int32_t> all_counts(groups, 0), valid_counts(groups, 0);
  std::vector<bool> first_valid(groups, false), second_valid(groups, false);
  std::vector<bool> aggregate_valid(groups, false), floating_valid(groups, false);
  std::vector<int> const pattern{0, 1, 0, 2, 1, 0, 3, 2};
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(key_data.size()); ++i) {
    auto const last    = i == offset + 2 * rows - 1;
    auto const group   = last ? 4 : pattern[i % pattern.size()];
    key_data[i]        = group;
    key_validity[i]    = i % 19 != 0;
    integer_data[0][i] = -1 - i % 17;
    integer_data[1][i] = -20 - i % 13;
    integer_data[2][i] = i % 11 - 5;
    floating_data[i]   = last ? std::numeric_limits<double>::quiet_NaN() : -0.5 * (i % 23 + 1);
    wide_data[i]       = 1000 + i % 29;
    validity[0][i]     = group != 3 && i % 7 != 0;
    validity[1][i]     = group != 3 && (group != 2 || i >= offset + rows) && i % 11 != 0;
    validity[2][i]     = group != 3 && i % 5 != 0;
    if (i < offset || i >= offset + 2 * rows || !key_validity[i]) { continue; }
    ++all_counts[group];
    wide_sums[group] += wide_data[i];
    if (validity[0][i]) {
      max_first[group]   = std::max(max_first[group], integer_data[0][i]);
      first_valid[group] = true;
    }
    if (validity[1][i]) {
      max_second[group]   = std::max(max_second[group], integer_data[1][i]);
      second_valid[group] = true;
    }
    if (validity[2][i]) {
      auto const value = integer_data[2][i];
      sums[group] += value;
      squared_sums[group] += static_cast<int64_t>(value) * value;
      ++valid_counts[group];
      aggregate_valid[group] = true;
      max_floating[group] =
        last ? floating_data[i] : std::max(max_floating[group], floating_data[i]);
      floating_valid[group] = true;
    }
  }
  ASSERT_EQ(all_counts[4], 1);
  ASSERT_EQ(valid_counts[3], 0);
  ASSERT_TRUE(floating_valid[4]);
  cudf::test::fixed_width_column_wrapper<int32_t> input_keys(
    key_data.begin(), key_data.end(), key_validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> first(
    integer_data[0].begin(), integer_data[0].end(), validity[0].begin());
  cudf::test::fixed_width_column_wrapper<int32_t> second(
    integer_data[1].begin(), integer_data[1].end(), validity[1].begin());
  cudf::test::fixed_width_column_wrapper<int32_t> integers(
    integer_data[2].begin(), integer_data[2].end(), validity[2].begin());
  cudf::test::fixed_width_column_wrapper<double> floating(
    floating_data.begin(), floating_data.end(), validity[2].begin());
  cudf::test::fixed_width_column_wrapper<int64_t> wide(wide_data.begin(), wide_data.end());
  auto const batches =
    cudf::slice(cudf::table_view{{input_keys, first, second, integers, floating, wide}},
                {offset, offset + rows, offset + rows, offset + 2 * rows});
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(2, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(3, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(
    make_req(3, cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(
    3, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE)));
  requests.push_back(make_req(
    3, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE)));
  requests.push_back(make_req(4, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(5, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  // B/256 is one, so five groups use R=1. Each batch ends in a partial warp.
  cudf::groupby::streaming_groupby streaming(KEY_COL, requests, rows, cudf::null_policy::EXCLUDE);
  for (auto const& batch : batches) {
    streaming.aggregate(batch);
  }
  auto [keys, results] = streaming.finalize();
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_first(
    max_first.begin(), max_first.end(), first_valid.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_second(
    max_second.begin(), max_second.end(), second_valid.begin());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_sums(
    sums.begin(), sums.end(), aggregate_valid.begin());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_squares(
    squared_sums.begin(), squared_sums.end(), aggregate_valid.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_all(all_counts.begin(),
                                                               all_counts.end());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_valid(valid_counts.begin(),
                                                                 valid_counts.end());
  cudf::test::fixed_width_column_wrapper<double> expected_floating(
    max_floating.begin(), max_floating.end(), floating_valid.begin());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_wide(wide_sums.begin(), wide_sums.end());
  check(keys,
        results,
        cudf::table_view{{expected_keys}},
        {expected_first,
         expected_second,
         expected_sums,
         expected_squares,
         expected_all,
         expected_valid,
         expected_floating,
         expected_wide});
}

template <typename T>
struct StreamingGroupbyFloatingPeerTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(StreamingGroupbyFloatingPeerTest, cudf::test::FloatingPointTypes);

TYPED_TEST(StreamingGroupbyFloatingPeerTest, MinMaxSpecialValuesNullsAndTails)
{
  using T                               = TypeParam;
  constexpr cudf::size_type first_rows  = 270;
  constexpr cudf::size_type second_rows = 271;
  constexpr cudf::size_type offset      = 5;
  constexpr int groups                  = 10;
  auto constexpr nan                    = std::numeric_limits<T>::quiet_NaN();
  auto constexpr inf                    = std::numeric_limits<T>::infinity();
  auto constexpr tiny                   = std::numeric_limits<T>::denorm_min();
  std::vector<std::vector<T>> const patterns{{nan, nan},
                                             {T{0}, T{0}},
                                             {-T{0}, -T{0}},
                                             {T{0}, -T{0}},
                                             {-inf, T{-3}, -tiny, tiny, T{3}, inf},
                                             {T{-4}, -tiny, tiny, T{2}},
                                             {tiny, T{2} * tiny},
                                             {-T{2} * tiny, -tiny},
                                             {nan},
                                             {nan, T{-2}, T{5}}};
  auto const size = first_rows + second_rows + 2 * offset;
  std::vector<int32_t> key_data(size);
  std::vector<T> values_data(size);
  std::vector<bool> key_validity(size), validity(size);
  std::vector<T> expected_min(groups, inf), expected_max(groups, -inf);
  for (cudf::size_type i = 0; i < size; ++i) {
    auto const group    = i % groups;
    auto const& pattern = patterns[group];
    key_data[i]         = group;
    key_validity[i]     = i % 19 != 0;
    validity[i]         = group != 8 && i % 7 != 0;
    auto const value    = pattern[(i / groups) % pattern.size()];
    // Invalid NaN payloads must not classify or affect a finite peer reduction.
    values_data[i] = validity[i] ? value : nan;
    if (i < offset || i >= offset + first_rows + second_rows || !key_validity[i] || !validity[i] ||
        std::isnan(value)) {
      continue;
    }
    expected_min[group] = std::min(expected_min[group], value);
    expected_max[group] = std::max(expected_max[group], value);
  }
  cudf::test::fixed_width_column_wrapper<int32_t> input_keys(
    key_data.begin(), key_data.end(), key_validity.begin());
  // Separate input columns keep this focused on MIN/MAX rather than compound finalization.
  cudf::test::fixed_width_column_wrapper<T> min_values(
    values_data.begin(), values_data.end(), validity.begin());
  cudf::test::fixed_width_column_wrapper<T> max_values(
    values_data.begin(), values_data.end(), validity.begin());
  auto const batches = cudf::slice(
    cudf::table_view{{input_keys, min_values, max_values}},
    {offset, offset + first_rows, offset + first_rows, offset + first_rows + second_rows});
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(2, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  // Both batches have R=1, interleaved same-target peers, null holes and partial final warps.
  cudf::groupby::streaming_groupby streaming(
    KEY_COL, requests, second_rows, cudf::null_policy::EXCLUDE);
  for (auto const& batch : batches) {
    streaming.aggregate(batch);
  }
  auto [keys, results] = streaming.finalize();
  ASSERT_EQ(results.size(), 2);
  auto const order       = cudf::sorted_order(keys->view());
  auto const sorted_keys = cudf::gather(keys->view(), *order);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, sorted_keys->view().column(0));
  for (std::size_t aggregation = 0; aggregation < results.size(); ++aggregation) {
    SCOPED_TRACE(::testing::Message() << "aggregation=" << aggregation);
    ASSERT_EQ(results[aggregation].results.size(), 1);
    auto const sorted =
      cudf::gather(cudf::table_view{{results[aggregation].results[0]->view()}}, *order);
    auto const [values, mask] = cudf::test::to_host<T>(sorted->view().column(0));
    ASSERT_FALSE(mask.empty());
    auto const& expected = aggregation == 0 ? expected_min : expected_max;
    for (int group = 0; group < groups; ++group) {
      SCOPED_TRACE(::testing::Message() << "group=" << group);
      EXPECT_EQ(cudf::bit_is_set(mask.data(), group), group != 8);
      if (group == 8) { continue; }
      if (group == 0) {
        EXPECT_TRUE(std::isnan(values[group]));
      } else if (group == 9) {
        // Existing MIN/MAX can return either a NaN or a finite input depending on atomic
        // order. Do not establish a new NaN propagation/ignoring policy in this test.
        EXPECT_TRUE(std::isnan(values[group]) || values[group] == T{-2} || values[group] == T{5});
      } else {
        EXPECT_EQ(values[group], expected[group]);
        if (group == 1 || group == 2) { EXPECT_EQ(std::signbit(values[group]), group == 2); }
        // Mixed-sign zero group3 intentionally has no sign-bit expectation.
      }
    }
  }
}

TEST_F(StreamingGroupbyTest, PartialLayoutPaddingGrowthAndMerge)
{
  struct expected_state {
    std::vector<int64_t> sums   = std::vector<int64_t>(8);
    std::vector<int32_t> maxima = std::vector<int32_t>(8, std::numeric_limits<int32_t>::lowest());
    std::vector<int32_t> valid_counts = std::vector<int32_t>(8);
    std::vector<int32_t> all_counts   = std::vector<int32_t>(8);
  } expected_destination, expected_source;

  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(1, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE)));
  requests.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE)));
  cudf::groupby::streaming_groupby destination(KEY_COL, requests, 16384);
  cudf::groupby::streaming_groupby source(KEY_COL, requests, 16384);

  auto append = [](auto& groupby, auto& expected, int rows, int first_key, int groups, int tag) {
    std::vector<int32_t> key_data(rows), value_data(rows);
    std::vector<bool> validity(rows);
    for (int row = 0; row < rows; ++row) {
      auto const key   = first_key + row % groups;
      auto const value = tag + row % 5;
      key_data[row]    = key;
      value_data[row]  = value;
      validity[row]    = key != 0 && row % 7 != 0;
      ++expected.all_counts[key];
      if (validity[row]) {
        expected.sums[key] += value;
        expected.maxima[key] = std::max(expected.maxima[key], value);
        ++expected.valid_counts[key];
      }
    }
    cudf::test::fixed_width_column_wrapper<int32_t> keys(key_data.begin(), key_data.end());
    cudf::test::fixed_width_column_wrapper<int32_t> values{
      value_data.begin(), value_data.end(), validity.begin()};
    groupby.aggregate(cudf::table_view{{keys, values}});
  };

  auto verify = [](auto&& result, expected_state const& expected) {
    auto& [keys, results] = result;
    std::vector<int32_t> expected_keys, maxima, valid_counts, all_counts;
    std::vector<int64_t> sums;
    std::vector<bool> validity;
    for (int key = 0; key < 8; ++key) {
      if (expected.all_counts[key] == 0) { continue; }
      expected_keys.push_back(key);
      sums.push_back(expected.sums[key]);
      maxima.push_back(expected.maxima[key]);
      valid_counts.push_back(expected.valid_counts[key]);
      all_counts.push_back(expected.all_counts[key]);
      validity.push_back(expected.valid_counts[key] != 0);
    }
    cudf::test::fixed_width_column_wrapper<int32_t> key_column(expected_keys.begin(),
                                                               expected_keys.end());
    cudf::test::fixed_width_column_wrapper<int64_t> sum_column{
      sums.begin(), sums.end(), validity.begin()};
    cudf::test::fixed_width_column_wrapper<int32_t> max_column{
      maxima.begin(), maxima.end(), validity.begin()};
    cudf::test::fixed_width_column_wrapper<int32_t> valid_column(valid_counts.begin(),
                                                                 valid_counts.end());
    cudf::test::fixed_width_column_wrapper<int32_t> all_column(all_counts.begin(),
                                                               all_counts.end());
    check(keys,
          results,
          cudf::table_view{{key_column}},
          {sum_column, max_column, valid_column, all_column});
  };

  // C3/R4 -> C6/R4 changes pitch without changing replica count. Four live groups
  // then leave padding in every shard; a live-prefix slice cannot represent them.
  append(destination, expected_destination, 3072, 0, 3, -30);
  append(destination, expected_destination, 6144, 0, 4, -20);
  verify(destination.finalize(), expected_destination);
  append(destination, expected_destination, 12288, 0, 4, -10);  // C6/R8
  append(source, expected_source, 2560, 2, 5, 10);
  append(source, expected_source, 5120, 2, 6, 20);  // C10/R2, six live groups
  verify(source.finalize(), expected_source);

  // Source and destination have different physical pitches, with padding on both.
  destination.merge(source);
  for (int key = 0; key < 8; ++key) {
    expected_destination.sums[key] += expected_source.sums[key];
    expected_destination.maxima[key] =
      std::max(expected_destination.maxima[key], expected_source.maxima[key]);
    expected_destination.valid_counts[key] += expected_source.valid_counts[key];
    expected_destination.all_counts[key] += expected_source.all_counts[key];
  }
  verify(destination.finalize(), expected_destination);
  destination.merge(destination);
  for (int key = 0; key < 8; ++key) {
    expected_destination.sums[key] *= 2;
    expected_destination.valid_counts[key] *= 2;
    expected_destination.all_counts[key] *= 2;
  }
  verify(std::move(destination).finalize_and_release(), expected_destination);

  // A source read must not mutate its partials or prevent a later update/release.
  append(source, expected_source, 5120, 2, 6, 30);
  verify(std::move(source).finalize_and_release(), expected_source);
}

TEST_F(StreamingGroupbyTest, Decimal128PeerSumCarriesNullsScalesAndTails)
{
  using fp128                      = cudf::test::fixed_point_column_wrapper<__int128_t>;
  constexpr cudf::size_type rows   = 270;
  constexpr cudf::size_type offset = 5;
  constexpr int groups             = 5;
  constexpr __int128_t high        = static_cast<__int128_t>(1) << 96;
  constexpr __int128_t word        = static_cast<__int128_t>(1) << 64;
  auto const scale_a               = numeric::scale_type{-3};
  auto const scale_b               = numeric::scale_type{2};
  std::vector<int32_t> key_data(2 * rows + 2 * offset);
  std::vector<__int128_t> first_data(key_data.size()), second_data(key_data.size());
  std::vector<bool> key_validity(key_data.size());
  std::vector<bool> first_validity(key_data.size()), second_validity(key_data.size());
  std::vector<std::vector<__int128_t>> sums(2, std::vector<__int128_t>(groups));
  std::vector<std::vector<__int128_t>> first_sums = sums;
  std::vector<std::vector<bool>> valid(2, std::vector<bool>(groups, false));
  std::vector<std::vector<bool>> first_valid = valid;
  std::vector<__int128_t> const pattern{
    high + word - 1, 3, -high + word, -high - word + 1, -3, high - word};
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(key_data.size()); ++i) {
    auto const group   = i % groups;
    key_data[i]        = group;
    key_validity[i]    = i % 19 != 0;
    first_data[i]      = pattern[(i / groups) % pattern.size()];
    second_data[i]     = -first_data[i] + i % 5;
    auto const active  = group != 4 && (group != 3 || i >= offset + rows);
    first_validity[i]  = active && i % 7 != 0;
    second_validity[i] = active && i % 11 != 0;
    if (i < offset || i >= offset + 2 * rows || !key_validity[i]) { continue; }
    for (int column = 0; column < 2; ++column) {
      if (!(column == 0 ? first_validity[i] : second_validity[i])) { continue; }
      auto const value = column == 0 ? first_data[i] : second_data[i];
      sums[column][group] += value;
      valid[column][group] = true;
      if (i < offset + rows) {
        first_sums[column][group] += value;
        first_valid[column][group] = true;
      }
    }
  }
  // Even the sum of absolute values fits comfortably in 128 bits; low-word carries/borrows
  // and cancellation across the 64-bit boundary are intentional.
  cudf::test::fixed_width_column_wrapper<int32_t> input_keys(
    key_data.begin(), key_data.end(), key_validity.begin());
  fp128 first(first_data.begin(), first_data.end(), first_validity.begin(), scale_a);
  fp128 second(second_data.begin(), second_data.end(), second_validity.begin(), scale_b);
  auto const batches = cudf::slice(cudf::table_view{{input_keys, first, second}},
                                   {offset, offset + rows, offset + rows, offset + 2 * rows});
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(2, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  // R=1 keeps interleaved peers on the same target; each sliced batch has a partial warp.
  cudf::groupby::streaming_groupby streaming(KEY_COL, requests, rows, cudf::null_policy::EXCLUDE);
  streaming.aggregate(batches[0]);
  auto [snapshot_keys, snapshot_results] = streaming.finalize();
  streaming.aggregate(batches[1]);
  auto [keys, results] = streaming.finalize();

  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2, 3, 4};
  fp128 expected_first(first_sums[0].begin(), first_sums[0].end(), first_valid[0].begin(), scale_a);
  fp128 expected_second(
    first_sums[1].begin(), first_sums[1].end(), first_valid[1].begin(), scale_b);
  fp128 expected_final_first(sums[0].begin(), sums[0].end(), valid[0].begin(), scale_a);
  fp128 expected_final_second(sums[1].begin(), sums[1].end(), valid[1].begin(), scale_b);
  ASSERT_EQ(results.size(), 2);
  ASSERT_EQ(results[0].results.size(), 1);
  ASSERT_EQ(results[1].results.size(), 1);
  EXPECT_EQ(results[0].results[0]->type().id(), cudf::type_id::DECIMAL128);
  EXPECT_EQ(results[1].results[0]->type().id(), cudf::type_id::DECIMAL128);
  EXPECT_EQ(results[0].results[0]->type().scale(), static_cast<int32_t>(scale_a));
  EXPECT_EQ(results[1].results[0]->type().scale(), static_cast<int32_t>(scale_b));
  auto verify = [&](auto& actual_keys,
                    auto& actual_results,
                    std::vector<cudf::column_view> const& expected_values) {
    auto const order       = cudf::sorted_order(actual_keys->view());
    auto const sorted_keys = cudf::gather(actual_keys->view(), *order);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, sorted_keys->view().column(0));
    ASSERT_EQ(actual_results.size(), expected_values.size());
    for (size_t i = 0; i < expected_values.size(); ++i) {
      ASSERT_EQ(actual_results[i].results.size(), 1);
      auto const sorted =
        cudf::gather(cudf::table_view{{actual_results[i].results.front()->view()}}, *order);
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_values[i], sorted->view().column(0));
    }
  };
  verify(snapshot_keys, snapshot_results, {expected_first, expected_second});
  verify(keys, results, {expected_final_first, expected_final_second});
}

TEST_F(StreamingGroupbyTest, SourceFusionKindsPromotionsNullsAndTails)
{
  constexpr int rows   = 270;
  constexpr int offset = 5;
  constexpr int groups = 5;
  std::vector<int64_t> sums(groups), squares(groups), products(groups, 1);
  std::vector<int32_t> minima(groups, std::numeric_limits<int32_t>::max());
  std::vector<int32_t> valid_counts(groups), all_counts(groups), double_counts(groups);
  std::vector<int32_t> product_counts(groups);
  std::vector<double> double_sums(groups), maxima(groups, std::numeric_limits<double>::lowest());
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(1, cudf::make_min_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(
    make_req(1, cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE)));
  requests.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE)));
  requests.push_back(make_req(2, cudf::make_max_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(2, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(
    2, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE)));
  // This I32 column is separated from the first I32 column by a different source type.
  requests.push_back(make_req(3, cudf::make_product_aggregation<cudf::groupby_aggregation>()));
  cudf::groupby::streaming_groupby streaming(KEY_COL, requests, 1024);
  for (int batch = 0; batch < 2; ++batch) {
    std::vector<int32_t> keys(rows + offset), integers(rows + offset), factors(rows + offset);
    std::vector<double> doubles(rows + offset);
    std::vector<bool> key_valid(rows + offset), integer_valid(rows + offset);
    std::vector<bool> double_valid(rows + offset), factor_valid(rows + offset);
    for (int i = 0; i < rows + offset; ++i) {
      auto const group = i % groups;
      keys[i]          = group;
      integers[i]      = (1 << 26) + (i % 7) + batch;
      doubles[i]       = (i % 17 - 8 + batch) * 0.25;
      factors[i]       = i % 3 == 0 ? -1 : 1;
      key_valid[i]     = i % 19 != 0;
      integer_valid[i] = group != 4 && i % 7 != 0;
      double_valid[i]  = group != 4 && i % 11 != 0;
      factor_valid[i]  = group != 4 && i % 13 != 0;
      if (i < offset || !key_valid[i]) { continue; }
      ++all_counts[group];
      if (integer_valid[i]) {
        sums[group] += integers[i];
        squares[group] += static_cast<int64_t>(integers[i]) * integers[i];
        minima[group] = std::min(minima[group], integers[i]);
        ++valid_counts[group];
      }
      if (double_valid[i]) {
        double_sums[group] += doubles[i];
        maxima[group] = std::max(maxima[group], doubles[i]);
        ++double_counts[group];
      }
      if (factor_valid[i]) {
        products[group] *= factors[i];
        ++product_counts[group];
      }
    }
    cudf::test::fixed_width_column_wrapper<int32_t> key_column{
      keys.begin(), keys.end(), key_valid.begin()};
    cudf::test::fixed_width_column_wrapper<int32_t> integer_column{
      integers.begin(), integers.end(), integer_valid.begin()};
    cudf::test::fixed_width_column_wrapper<double> double_column{
      doubles.begin(), doubles.end(), double_valid.begin()};
    cudf::test::fixed_width_column_wrapper<int32_t> factor_column{
      factors.begin(), factors.end(), factor_valid.begin()};
    auto const input =
      cudf::slice(cudf::table_view{{key_column, integer_column, double_column, factor_column}},
                  {offset, offset + rows})
        .front();
    streaming.aggregate(input);
  }
  auto result           = std::move(streaming).finalize_and_release();
  auto& [keys, results] = result;
  // Explicit type assertions catch accidental reuse of the source type for a promoted result.
  ASSERT_EQ(results.size(), requests.size());
  EXPECT_EQ(results[0].results.front()->type().id(), cudf::type_id::INT64);
  EXPECT_EQ(results[1].results.front()->type().id(), cudf::type_id::INT32);
  EXPECT_EQ(results[2].results.front()->type().id(), cudf::type_id::INT64);
  EXPECT_EQ(results[3].results.front()->type().id(), cudf::type_id::INT32);
  EXPECT_EQ(results[5].results.front()->type().id(), cudf::type_id::FLOAT64);
  EXPECT_EQ(results[8].results.front()->type().id(), cudf::type_id::INT64);
  std::vector<bool> valid(groups), double_valid(groups), product_valid(groups);
  for (int group = 0; group < groups; ++group) {
    valid[group]         = valid_counts[group] != 0;
    double_valid[group]  = double_counts[group] != 0;
    product_valid[group] = product_counts[group] != 0;
  }
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<int64_t> expected_sums{
    sums.begin(), sums.end(), valid.begin()};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_minima{
    minima.begin(), minima.end(), valid.begin()};
  cudf::test::fixed_width_column_wrapper<int64_t> expected_squares{
    squares.begin(), squares.end(), valid.begin()};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_valid(valid_counts.begin(),
                                                                 valid_counts.end());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_all(all_counts.begin(),
                                                               all_counts.end());
  cudf::test::fixed_width_column_wrapper<double> expected_maxima{
    maxima.begin(), maxima.end(), double_valid.begin()};
  cudf::test::fixed_width_column_wrapper<double> expected_double_sums{
    double_sums.begin(), double_sums.end(), double_valid.begin()};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_double_counts(double_counts.begin(),
                                                                         double_counts.end());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_products{
    products.begin(), products.end(), product_valid.begin()};
  std::vector<cudf::column_view> const expected_values{expected_sums,
                                                       expected_minima,
                                                       expected_squares,
                                                       expected_valid,
                                                       expected_all,
                                                       expected_maxima,
                                                       expected_double_sums,
                                                       expected_double_counts,
                                                       expected_products};
  auto const order       = cudf::sorted_order(keys->view());
  auto const sorted_keys = cudf::gather(keys->view(), *order);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, sorted_keys->view().column(0));
  for (size_t i = 0; i < expected_values.size(); ++i) {
    ASSERT_EQ(results[i].results.size(), 1);
    auto const sorted =
      cudf::gather(cudf::table_view{{results[i].results.front()->view()}}, *order);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_values[i], sorted->view().column(0));
  }
}

TEST_F(StreamingGroupbyTest, RepresentativeTargetsNormalizedKeysAndNulls)
{
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  cudf::test::fixed_width_column_wrapper<double> keys1{{nan, 0.0, -0.0, 1.0, 0.0, nan, 2.0},
                                                       {true, true, true, true, false, true, true}};
  cudf::test::fixed_width_column_wrapper<double> keys2{{3.0, -0.0, nan, 3.0, 2.0, 4.0, 1.0},
                                                       {true, true, true, true, true, false, true}};
  cudf::test::fixed_width_column_wrapper<double> keys3{{nan, 3.0, 0.0, 1.0, 2.0, 0.0},
                                                       {true, true, true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> values1{1, 2, 3, 4, 100, 5, 6};
  cudf::test::fixed_width_column_wrapper<int32_t> values2{7, 8, 9, 10, 11, 200, 12};
  cudf::test::fixed_width_column_wrapper<int32_t> values3{13, 14, 15, 16, 17, 300};
  cudf::test::structs_column_wrapper nested1{{keys1}, {true, true, true, true, false, true, true}};
  cudf::test::structs_column_wrapper nested2{{keys2}, {true, true, true, true, true, false, true}};
  cudf::test::structs_column_wrapper nested3{{keys3}, {true, true, true, true, true, false}};
  auto requests = single_agg_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  for (auto const include_nulls : {false, true}) {
    SCOPED_TRACE(include_nulls ? "include null keys" : "exclude null keys");
    std::vector<double> expected_key_data{0.0, 1.0, 2.0, 3.0, nan};
    std::vector<int64_t> expected_sum_data{28, 32, 34, 31, 28};
    std::vector<bool> expected_key_validity(expected_key_data.size(), true);
    if (include_nulls) {
      expected_key_data.insert(expected_key_data.begin(), 0.0);
      expected_key_validity.insert(expected_key_validity.begin(), false);
      expected_sum_data.insert(expected_sum_data.begin(), 600);
    }
    auto expected_keys =
      include_nulls
        ? cudf::test::fixed_width_column_wrapper<double>(
            expected_key_data.begin(), expected_key_data.end(), expected_key_validity.begin())
        : cudf::test::fixed_width_column_wrapper<double>(expected_key_data.begin(),
                                                         expected_key_data.end());
    cudf::test::structs_column_wrapper expected_nested{
      {expected_keys}, include_nulls ? expected_key_validity : std::vector<bool>{}};
    cudf::test::fixed_width_column_wrapper<int64_t> expected_sum(expected_sum_data.begin(),
                                                                 expected_sum_data.end());

    for (auto const nested : {false, true}) {
      SCOPED_TRACE(nested ? "nested keys" : "flat keys");
      cudf::groupby::streaming_groupby streaming(
        KEY_COL,
        requests,
        DEFAULT_MAX_DISTINCT_KEYS,
        include_nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE);
      auto const first_keys  = nested ? cudf::column_view{nested1} : cudf::column_view{keys1};
      auto const second_keys = nested ? cudf::column_view{nested2} : cudf::column_view{keys2};
      auto const third_keys  = nested ? cudf::column_view{nested3} : cudf::column_view{keys3};
      auto const expected =
        nested ? cudf::column_view{expected_nested} : cudf::column_view{expected_keys};
      streaming.aggregate(cudf::table_view{{first_keys, values1}});
      // Mix old and new representatives, then send a batch with no new keys. NaNs
      // and signed zeros must preserve their equality under representative rehashing.
      streaming.aggregate(cudf::table_view{{second_keys, values2}});
      streaming.aggregate(cudf::table_view{{third_keys, values3}});
      auto [keys, results] = streaming.finalize();
      check(keys, results, cudf::table_view{{expected}}, {expected_sum});
    }
  }
}

TEST_F(StreamingGroupbyTest, CombinedDecimal128SumAndCountsFusion)
{
  using fp128                      = cudf::test::fixed_point_column_wrapper<__int128_t>;
  constexpr cudf::size_type rows   = 2051;
  constexpr cudf::size_type offset = 5;
  constexpr int groups             = 5;
  constexpr __int128_t high        = static_cast<__int128_t>(1) << 80;
  constexpr __int128_t word        = static_cast<__int128_t>(1) << 64;
  auto const scale                 = numeric::scale_type{-4};
  std::vector<int32_t> key_data(2 * rows + 2 * offset);
  std::vector<__int128_t> data(key_data.size());
  std::vector<bool> key_validity(key_data.size()), validity(key_data.size());
  std::vector<__int128_t> sums(groups), first_sums(groups);
  std::vector<int32_t> valid_counts(groups), all_counts(groups);
  std::vector<int32_t> first_valid_counts(groups), first_all_counts(groups);
  std::vector<bool> valid(groups, false), first_valid(groups, false);
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(key_data.size()); ++i) {
    auto const second = i >= offset + rows;
    auto const group  = i % (second ? groups : 3);
    key_data[i]       = group;
    key_validity[i]   = i % 19 != 0;
    data[i]           = i % 2 ? high + word - 1 : -high + word + 3;
    validity[i]       = group != 4 && (group != 2 || second) && i % 7 != 0;
    if (i < offset || i >= offset + 2 * rows || !key_validity[i]) { continue; }
    ++all_counts[group];
    if (!second) { ++first_all_counts[group]; }
    if (validity[i]) {
      sums[group] += data[i];
      ++valid_counts[group];
      valid[group] = true;
      if (!second) {
        first_sums[group] += data[i];
        ++first_valid_counts[group];
        first_valid[group] = true;
      }
    }
  }
  cudf::test::fixed_width_column_wrapper<int32_t> input_keys(
    key_data.begin(), key_data.end(), key_validity.begin());
  fp128 input_values(data.begin(), data.end(), validity.begin(), scale);
  auto const batches = cudf::slice(cudf::table_view{{input_keys, input_values}},
                                   {offset, offset + rows, offset + rows, offset + 2 * rows});
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.push_back(make_req(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>()));
  requests.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE)));
  requests.push_back(make_req(
    1, cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE)));
  // The same decoded decimal128 source type must fuse SUM with both promoted count outputs.
  // The first batch has C=3/R=2; new groups force C=6/R=1 in the second batch.
  // Both slices end in partial warps, and the sum of absolute inputs fits signed 128 bits.
  cudf::groupby::streaming_groupby streaming(KEY_COL, requests, rows, cudf::null_policy::EXCLUDE);
  streaming.aggregate(batches[0]);
  auto [snapshot_keys, snapshot_results] = streaming.finalize();
  streaming.aggregate(batches[1]);
  auto [keys, results] = std::move(streaming).finalize_and_release();
  auto verify          = [&](auto& actual_keys,
                    auto& actual_results,
                    int count,
                    std::vector<__int128_t> const& expected_sums,
                    std::vector<bool> const& expected_valid,
                    std::vector<int32_t> const& expected_valid_counts,
                    std::vector<int32_t> const& expected_all_counts) {
    std::vector<int32_t> key_values(count);
    std::iota(key_values.begin(), key_values.end(), 0);
    cudf::test::fixed_width_column_wrapper<int32_t> expected_keys(key_values.begin(),
                                                                  key_values.end());
    fp128 expected_sum(
      expected_sums.begin(), expected_sums.begin() + count, expected_valid.begin(), scale);
    cudf::test::fixed_width_column_wrapper<int32_t> expected_valid_count(
      expected_valid_counts.begin(), expected_valid_counts.begin() + count);
    cudf::test::fixed_width_column_wrapper<int32_t> expected_all_count(
      expected_all_counts.begin(), expected_all_counts.begin() + count);
    auto const order       = cudf::sorted_order(actual_keys->view());
    auto const sorted_keys = cudf::gather(actual_keys->view(), *order);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, sorted_keys->view().column(0));
    ASSERT_EQ(actual_results.size(), 3);
    std::vector<cudf::column_view> const expected_values{
      expected_sum, expected_valid_count, expected_all_count};
    for (size_t i = 0; i < expected_values.size(); ++i) {
      ASSERT_EQ(actual_results[i].results.size(), 1);
      EXPECT_EQ(actual_results[i].results[0]->type(), expected_values[i].type());
      auto const sorted =
        cudf::gather(cudf::table_view{{actual_results[i].results.front()->view()}}, *order);
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_values[i], sorted->view().column(0));
    }
  };
  verify(snapshot_keys,
         snapshot_results,
         3,
         first_sums,
         first_valid,
         first_valid_counts,
         first_all_counts);
  verify(keys, results, groups, sums, valid, valid_counts, all_counts);
}
