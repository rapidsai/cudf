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

#include "arrow/util/tdigest.h"

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <rmm/exec_policy.hpp>

#include <tests/groupby/groupby_test_util.hpp>

#include <thrust/fill.h>

namespace cudf {
namespace test {

using namespace cudf;

typedef thrust::tuple<size_type, double, double> expected_value;

template <typename T>
struct TDigestAllTypes : public cudf::test::BaseFixture {
};
TYPED_TEST_CASE(TDigestAllTypes, cudf::test::NumericTypes);

struct tdigest_gen {
  template <
    typename T,
    typename std::enable_if_t<cudf::is_numeric<T>() || cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& keys, column_view const& values, int delta)
  {
    cudf::table_view t({keys});
    cudf::groupby::groupby gb(t);
    std::vector<cudf::groupby::aggregation_request> requests;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
    requests.push_back({values, std::move(aggregations)});
    auto result = gb.aggregate(requests);
    return std::move(result.second[0].results[0]);
  }

  template <
    typename T,
    typename std::enable_if_t<!cudf::is_numeric<T>() && !cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& keys, column_view const& values, int delta)
  {
    CUDF_FAIL("Invalid tdigest test type");
  }
};

void tdigest_sample_compare(column_view const& result,
                            std::vector<expected_value> const& h_expected)
{
  cudf::detail::check_is_valid_tdigest_column(result);
  cudf::structs_column_view scv(result);
  cudf::lists_column_view lcv(scv.child(cudf::detail::tdigest_centroid_column_index));
  cudf::structs_column_view tdigests(lcv.child());
  column_view result_mean   = tdigests.child(cudf::detail::tdigest_mean_column_index);
  column_view result_weight = tdigests.child(cudf::detail::tdigest_weight_column_index);

  auto expected_mean = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, h_expected.size(), mask_state::UNALLOCATED);
  auto expected_weight = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, h_expected.size(), mask_state::UNALLOCATED);
  auto sampled_result_mean = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, h_expected.size(), mask_state::UNALLOCATED);
  auto sampled_result_weight = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, h_expected.size(), mask_state::UNALLOCATED);

  rmm::device_vector<expected_value> expected(h_expected.begin(), h_expected.end());
  auto iter = thrust::make_counting_iterator(0);
  thrust::for_each(
    rmm::exec_policy(rmm::cuda_stream_default),
    iter,
    iter + expected.size(),
    [expected            = expected.data().get(),
     expected_mean       = expected_mean->mutable_view().begin<double>(),
     expected_weight     = expected_weight->mutable_view().begin<double>(),
     result_mean         = result_mean.begin<double>(),
     result_weight       = result_weight.begin<double>(),
     sampled_result_mean = sampled_result_mean->mutable_view().begin<double>(),
     sampled_result_weight =
       sampled_result_weight->mutable_view().begin<double>()] __device__(size_type index) {
      expected_mean[index]         = thrust::get<1>(expected[index]);
      expected_weight[index]       = thrust::get<2>(expected[index]);
      auto const src_index         = thrust::get<0>(expected[index]);
      sampled_result_mean[index]   = result_mean[src_index];
      sampled_result_weight[index] = result_weight[src_index];
    });

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected_mean, *sampled_result_mean);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_weight, *sampled_result_weight);
}

template <typename T>
std::unique_ptr<column> make_expected_tdigest(column_view const& mean,
                                              column_view const& weight,
                                              T min,
                                              T max)
{
  std::vector<std::unique_ptr<column>> inner_children;
  inner_children.push_back(std::make_unique<cudf::column>(mean));
  inner_children.push_back(std::make_unique<cudf::column>(weight));
  // tdigest struct
  auto tdigests = cudf::make_structs_column(mean.size(), std::move(inner_children), 0, {});

  std::vector<offset_type> h_offsets{0, mean.size()};
  auto offsets =
    cudf::make_fixed_width_column(data_type{type_id::INT32}, 2, mask_state::UNALLOCATED);
  cudaMemcpy(offsets->mutable_view().begin<offset_type>(),
             h_offsets.data(),
             sizeof(offset_type) * 2,
             cudaMemcpyHostToDevice);

  auto list = cudf::make_lists_column(1, std::move(offsets), std::move(tdigests), 0, {});

  auto min_col =
    cudf::make_fixed_width_column(data_type{type_id::FLOAT64}, 1, mask_state::UNALLOCATED);
  thrust::fill(rmm::exec_policy(rmm::cuda_stream_default),
               min_col->mutable_view().begin<double>(),
               min_col->mutable_view().end<double>(),
               static_cast<double>(min));
  auto max_col =
    cudf::make_fixed_width_column(data_type{type_id::FLOAT64}, 1, mask_state::UNALLOCATED);
  thrust::fill(rmm::exec_policy(rmm::cuda_stream_default),
               max_col->mutable_view().begin<double>(),
               max_col->mutable_view().end<double>(),
               static_cast<double>(max));

  std::vector<std::unique_ptr<column>> children;
  children.push_back(std::move(list));
  children.push_back(std::move(min_col));
  children.push_back(std::move(max_col));
  return make_structs_column(1, std::move(children), 0, {});
}

TYPED_TEST(TDigestAllTypes, Simple)
{
  using T = TypeParam;

  // create a tdigest that has far fewer values in it than the delta value. this should result
  // in every value remaining uncompressed
  cudf::test::fixed_width_column_wrapper<T> values{126, 15, 1, 99, 67};
  cudf::test::fixed_width_column_wrapper<int> keys{0, 0, 0, 0, 0};
  int const delta = 1000;
  auto result     = cudf::type_dispatcher(
    static_cast<column_view>(values).type(), tdigest_gen{}, keys, values, delta);

  cudf::test::fixed_width_column_wrapper<T> raw_mean({1, 15, 67, 99, 126});
  cudf::test::fixed_width_column_wrapper<double> weight{1, 1, 1, 1, 1};
  auto mean        = cudf::cast(raw_mean, data_type{type_id::FLOAT64});
  double const min = 1;
  double const max = 126;
  auto expected = make_expected_tdigest<T>(*mean, weight, static_cast<T>(min), static_cast<T>(max));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TYPED_TEST(TDigestAllTypes, SimpleWithNulls)
{
  using T = TypeParam;

  // create a tdigest that has far fewer values in it than the delta value. this should result
  // in every value remaining uncompressed
  cudf::test::fixed_width_column_wrapper<T> values{{122, 15, 1, 99, 67, 101, 100, 84, 44, 2},
                                                   {1, 0, 1, 0, 1, 0, 1, 0, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int> keys{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int const delta = 1000;
  auto result     = cudf::type_dispatcher(
    static_cast<column_view>(values).type(), tdigest_gen{}, keys, values, delta);

  cudf::test::fixed_width_column_wrapper<T> raw_mean({1, 44, 67, 100, 122});
  cudf::test::fixed_width_column_wrapper<double> weight{1, 1, 1, 1, 1};
  auto mean        = cudf::cast(raw_mean, data_type{type_id::FLOAT64});
  double const min = 1;
  double const max = 122;
  auto expected = make_expected_tdigest<T>(*mean, weight, static_cast<T>(min), static_cast<T>(max));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TYPED_TEST(TDigestAllTypes, AllNull)
{
  using T = TypeParam;

  // create a tdigest that has far fewer values in it than the delta value. this should result
  // in every value remaining uncompressed
  cudf::test::fixed_width_column_wrapper<T> values{{122, 15, 1, 99, 67, 101, 100, 84, 44, 2},
                                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  cudf::test::fixed_width_column_wrapper<int> keys{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int const delta = 1000;
  auto result     = cudf::type_dispatcher(
    static_cast<column_view>(values).type(), tdigest_gen{}, keys, values, delta);

  // NOTE: an empty tdigest column still has 1 row.
  auto expected = cudf::detail::make_empty_tdigest_column();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TYPED_TEST(TDigestAllTypes, LargeGroups)
{
  auto _values    = generate_standardized_percentile_distribution(data_type{type_id::FLOAT64});
  int const delta = 1000;

  // generate a random set of keys
  std::vector<int> h_keys;
  h_keys.reserve(_values->size());
  auto iter = thrust::make_counting_iterator(0);
  std::transform(iter, iter + _values->size(), std::back_inserter(h_keys), [](int i) {
    return static_cast<int>(round(rand_range(0, 8)));
  });
  cudf::test::fixed_width_column_wrapper<int> _keys(h_keys.begin(), h_keys.end());

  // group the input values together
  cudf::table_view k({_keys});
  cudf::groupby::groupby setup_gb(k);
  cudf::table_view v({*_values});
  auto groups = setup_gb.get_groups(v);

  // slice it all up so we have keys/columns for everything.
  std::vector<column_view> keys;
  std::vector<column_view> values;
  for (size_t idx = 0; idx < groups.offsets.size() - 1; idx++) {
    auto k =
      cudf::slice(groups.keys->get_column(0), {groups.offsets[idx], groups.offsets[idx + 1]});
    keys.push_back(k[0]);

    auto v =
      cudf::slice(groups.values->get_column(0), {groups.offsets[idx], groups.offsets[idx + 1]});
    values.push_back(v[0]);
  }

  // generate a seperate tdigest for each group
  std::vector<std::unique_ptr<column>> parts;
  std::transform(
    iter, iter + values.size(), std::back_inserter(parts), [&keys, &values, delta](int i) {
      cudf::table_view t({keys[i]});
      cudf::groupby::groupby gb(t);
      std::vector<cudf::groupby::aggregation_request> requests;
      std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
      aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
      requests.push_back({values[i], std::move(aggregations)});
      auto result = gb.aggregate(requests);
      return std::move(result.second[0].results[0]);
    });
  std::vector<column_view> part_views;
  std::transform(parts.begin(),
                 parts.end(),
                 std::back_inserter(part_views),
                 [](std::unique_ptr<column> const& col) { return col->view(); });
  auto merged_parts = cudf::concatenate(part_views);

  // generate a tdigest on the whole input set
  cudf::table_view t({_keys});
  cudf::groupby::groupby gb(t);
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
  requests.push_back({*_values, std::move(aggregations)});
  auto result = gb.aggregate(requests);

  // verify that they end up the same.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result.second[0].results[0], *merged_parts);
}

struct TDigestTest : public cudf::test::BaseFixture {
};

TEST_F(TDigestTest, LargeInputDouble)
{
  // these tests are being done explicitly because of the way we have to precompute the correct
  // answers. since the input values generated by the generate_distribution() function below are
  // cast to specific types -before- being sent into the aggregation, I can't (safely) just use the
  // expected values that you get when using doubles all the way through.  so I have to pregenerate
  // the correct answers for each type by hand. so, we'll choose a reasonable subset (double,
  // decimal, int, bool)

  auto values = generate_standardized_percentile_distribution(data_type{type_id::FLOAT64});
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, values->size(), mask_state::UNALLOCATED);
  thrust::fill(rmm::exec_policy(rmm::cuda_stream_default),
               keys->mutable_view().template begin<int>(),
               keys->mutable_view().template end<int>(),
               0);

  // compare against a sample of known/expected values (which themselves were verified against the
  // Arrow implementation)

  // delta 1000
  {
    int const delta = 1000;
    auto result =
      cudf::type_dispatcher(values->view().type(), tdigest_gen{}, *keys, *values, delta);
    std::vector<expected_value> expected{{0, 0.00040692343794663995, 7},
                                         {10, 0.16234555627091204477, 153},
                                         {59, 5.12764811246045937310, 858},
                                         {250, 62.54581814492237157310, 2356},
                                         {368, 87.85834376680742252574, 1735},
                                         {409, 94.07685720279611985006, 1272},
                                         {491, 99.94197663121231300920, 130},
                                         {500, 99.99969880795092080916, 2}};

    tdigest_sample_compare(*result, expected);
  }

  // delta 100
  {
    int const delta = 100;
    auto result =
      cudf::type_dispatcher(values->view().type(), tdigest_gen{}, *keys, *values, delta);
    std::vector<expected_value> expected{{0, 0.07265722021410986331, 739},
                                         {7, 8.19766194442652640362, 10693},
                                         {16, 36.82277869518204482802, 20276},
                                         {29, 72.95424834129075009059, 22623},
                                         {38, 90.61229683516096145013, 15581},
                                         {46, 99.07283498858802772702, 5142},
                                         {50, 99.99970905482754801596, 1}};

    tdigest_sample_compare(*result, expected);
  }

  // delta 10
  {
    int const delta = 10;
    auto result =
      cudf::type_dispatcher(values->view().type(), tdigest_gen{}, *keys, *values, delta);
    std::vector<expected_value> expected{{0, 7.15508346777729631327, 71618},
                                         {1, 33.04971680740474226923, 187499},
                                         {2, 62.50566666553867634093, 231762},
                                         {3, 83.46216572053654658703, 187500},
                                         {4, 96.42204425201593664951, 71620},
                                         {5, 99.99970905482754801596, 1}};

    tdigest_sample_compare(*result, expected);
  }
}

TEST_F(TDigestTest, LargeInputInt)
{
  // these tests are being done explicitly because of the way we have to precompute the correct
  // answers. since the input values generated by the generate_distribution() function below are
  // cast to specific types -before- being sent into the aggregation, I can't (safely) just use the
  // expected values that you get when using doubles all the way through.  so I have to pregenerate
  // the correct answers for each type by hand. so, we'll choose a reasonable subset (double,
  // decimal, int, bool)

  auto values = generate_standardized_percentile_distribution(data_type{type_id::INT32});
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, values->size(), mask_state::UNALLOCATED);
  thrust::fill(rmm::exec_policy(rmm::cuda_stream_default),
               keys->mutable_view().template begin<int>(),
               keys->mutable_view().template end<int>(),
               0);

  // compare against a sample of known/expected values (which themselves were verified against the
  // Arrow implementation)

  // delta 1000
  {
    int const delta = 1000;
    auto result =
      cudf::type_dispatcher(values->view().type(), tdigest_gen{}, *keys, *values, delta);
    std::vector<expected_value> expected{{0, 0, 7},
                                         {14, 0, 212},
                                         {26, 0.83247422680412408447, 388},
                                         {44, 2, 648},
                                         {45, 2.42598187311178170589, 662},
                                         {342, 82.75190258751908345403, 1971},
                                         {383, 90, 1577},
                                         {417, 94.88376068376066996279, 1170},
                                         {418, 95, 1157},
                                         {479, 99, 307},
                                         {500, 99, 2}};

    tdigest_sample_compare(*result, expected);
  }

  // delta 100
  {
    int const delta = 100;
    auto result =
      cudf::type_dispatcher(values->view().type(), tdigest_gen{}, *keys, *values, delta);
    std::vector<expected_value> expected{{0, 0, 739},
                                         {7, 7.71486018890863167741, 10693},
                                         {16, 36.32491615703294485229, 20276},
                                         {29, 72.44392874508245938614, 22623},
                                         {38, 90.14209614273795523332, 15581},
                                         {46, 98.64041229093737683797, 5142},
                                         {50, 99, 1}};

    tdigest_sample_compare(*result, expected);
  }

  // delta 10
  {
    int const delta = 10;
    auto result =
      cudf::type_dispatcher(values->view().type(), tdigest_gen{}, *keys, *values, delta);
    std::vector<expected_value> expected{{0, 6.66025300902007799664, 71618},
                                         {1, 32.54912826201739051157, 187499},
                                         {2, 62.00734805533262772315, 231762},
                                         {3, 82.96355733333332693746, 187500},
                                         {4, 95.91280368612116546956, 71620},
                                         {5, 99, 1}};

    tdigest_sample_compare(*result, expected);
  }
}

TEST_F(TDigestTest, LargeInputDecimal)
{
  // these tests are being done explicitly because of the way we have to precompute the correct
  // answers. since the input values generated by the generate_distribution() function below are
  // cast to specific types -before- being sent into the aggregation, I can't (safely) just use the
  // expected values that you get when using doubles all the way through.  so I have to pregenerate
  // the correct answers for each type by hand. so, we'll choose a reasonable subset (double,
  // decimal, int, bool)

  auto values = generate_standardized_percentile_distribution(data_type{type_id::DECIMAL32, -4});
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, values->size(), mask_state::UNALLOCATED);
  thrust::fill(rmm::exec_policy(rmm::cuda_stream_default),
               keys->mutable_view().template begin<int>(),
               keys->mutable_view().template end<int>(),
               0);

  // compare against a sample of known/expected values (which themselves were verified against the
  // Arrow implementation)

  // delta 1000
  {
    int const delta = 1000;
    auto result =
      cudf::type_dispatcher(values->view().type(), tdigest_gen{}, *keys, *values, delta);
    std::vector<expected_value> expected{{0, 0.00035714285714285709, 7},
                                         {10, 0.16229738562091505782, 153},
                                         {59, 5.12759696969697031932, 858},
                                         {250, 62.54576854838715860296, 2356},
                                         {368, 87.85829446685879418055, 1735},
                                         {409, 94.07680636792450457051, 1272},
                                         {491, 99.94192461538463589932, 130},
                                         {500, 99.99965000000000259206, 2}};

    tdigest_sample_compare(*result, expected);
  }

  // delta 100
  {
    int const delta = 100;
    auto result =
      cudf::type_dispatcher(values->view().type(), tdigest_gen{}, *keys, *values, delta);
    std::vector<expected_value> expected{{0, 0.07260811907983763525, 739},
                                         {7, 8.19761183016926864298, 10693},
                                         {16, 36.82272891595975750079, 20276},
                                         {29, 72.95419827167043536065, 22623},
                                         {38, 90.61224673640975879607, 15581},
                                         {46, 99.07278498638662256326, 5142},
                                         {50, 99.99970000000000425189, 1}};

    tdigest_sample_compare(*result, expected);
  }

  // delta 10
  {
    int const delta = 10;
    auto result =
      cudf::type_dispatcher(values->view().type(), tdigest_gen{}, *keys, *values, delta);
    std::vector<expected_value> expected{{0, 7.15503361864335740705, 71618},
                                         {1, 33.04966679715625588187, 187499},
                                         {2, 62.50561666407782013266, 231762},
                                         {3, 83.46211575573336460820, 187500},
                                         {4, 96.42199425300195514410, 71620},
                                         {5, 99.99970000000000425189, 1}};

    tdigest_sample_compare(*result, expected);
  }
}

struct TDigestMergeTest : public cudf::test::BaseFixture {
};

// Note: there is no need to test different types here as the internals of a tdigest are always
// the same regardless of input.
TEST_F(TDigestMergeTest, Simple)
{
  auto values = generate_standardized_percentile_distribution(data_type{type_id::FLOAT64});
  CUDF_EXPECTS(values->size() == 750000, "Unexpected distribution size");
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, values->size(), mask_state::UNALLOCATED);
  thrust::fill(rmm::exec_policy(rmm::cuda_stream_default),
               keys->mutable_view().template begin<int>(),
               keys->mutable_view().template end<int>(),
               0);

  auto split_values = cudf::split(*values, {250000, 500000});
  auto split_keys   = cudf::split(*keys, {250000, 500000});

  int const delta = 1000;

  // generate seperate digests
  std::vector<std::unique_ptr<column>> parts;
  auto iter = thrust::make_counting_iterator(0);
  std::transform(
    iter,
    iter + split_values.size(),
    std::back_inserter(parts),
    [&split_keys, &split_values, delta](int i) {
      cudf::table_view t({split_keys[i]});
      cudf::groupby::groupby gb(t);
      std::vector<cudf::groupby::aggregation_request> requests;
      std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
      aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
      requests.push_back({split_values[i], std::move(aggregations)});
      auto result = gb.aggregate(requests);
      return std::move(result.second[0].results[0]);
    });
  std::vector<column_view> part_views;
  std::transform(parts.begin(),
                 parts.end(),
                 std::back_inserter(part_views),
                 [](std::unique_ptr<column> const& col) { return col->view(); });

  // merge delta = 1000
  {
    int const merge_delta = 1000;

    // merge them
    auto merge_input = cudf::concatenate(part_views);
    cudf::test::fixed_width_column_wrapper<int> merge_keys{0, 0, 0};
    cudf::table_view key_table({merge_keys});
    cudf::groupby::groupby gb(key_table);
    std::vector<cudf::groupby::aggregation_request> requests;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.push_back(
      cudf::make_merge_tdigest_aggregation<cudf::groupby_aggregation>(merge_delta));
    requests.push_back({*merge_input, std::move(aggregations)});
    auto result = gb.aggregate(requests);

    std::vector<expected_value> expected{{0, 0.00013945158577498588, 2},
                                         {10, 0.04804393446447510763, 50},
                                         {59, 1.68846964439246893797, 284},
                                         {250, 33.36323141295877547918, 1479},
                                         {368, 65.36307727957283475462, 2292},
                                         {409, 73.95399208218296394080, 1784},
                                         {490, 87.67566167909056673579, 1570},
                                         {491, 87.83119717763385381204, 1570},
                                         {500, 89.24891838334393412424, 1555},
                                         {578, 95.87182997389099625707, 583},
                                         {625, 98.20470345147104751504, 405},
                                         {700, 99.96818381983835877236, 56},
                                         {711, 99.99970905482754801596, 1}};

    tdigest_sample_compare(*result.second[0].results[0], expected);
  }
}

}  // namespace test
}  // namespace cudf
