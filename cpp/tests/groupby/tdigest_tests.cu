/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/tdigest_utilities.cuh>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>

namespace {
/**
 * @brief Functor to generate a tdigest by key.
 *
 */
struct tdigest_gen_grouped {
  template <
    typename T,
    typename std::enable_if_t<cudf::is_numeric<T>() || cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& keys,
                                           cudf::column_view const& values,
                                           int delta)
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
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& keys,
                                           cudf::column_view const& values,
                                           int delta)
  {
    CUDF_FAIL("Invalid tdigest test type");
  }
};

/**
 * @brief Functor for generating a tdigest using groupby with a constant key.
 *
 */
struct tdigest_groupby_simple_op {
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& values, int delta) const
  {
    // make a simple set of matching keys.
    auto keys = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32}, values.size(), cudf::mask_state::UNALLOCATED);
    thrust::fill(rmm::exec_policy(cudf::get_default_stream()),
                 keys->mutable_view().template begin<int>(),
                 keys->mutable_view().template end<int>(),
                 0);

    cudf::table_view t({*keys});
    cudf::groupby::groupby gb(t);
    std::vector<cudf::groupby::aggregation_request> requests;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
    requests.push_back({values, std::move(aggregations)});
    auto result = gb.aggregate(requests);
    return std::move(result.second[0].results[0]);
  }
};

/**
 * @brief Functor for merging tdigests using groupby with a constant key.
 *
 */
struct tdigest_groupby_simple_merge_op {
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& merge_values,
                                           int merge_delta) const
  {
    // make a simple set of matching keys.
    auto merge_keys = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32}, merge_values.size(), cudf::mask_state::UNALLOCATED);
    thrust::fill(rmm::exec_policy(cudf::get_default_stream()),
                 merge_keys->mutable_view().template begin<int>(),
                 merge_keys->mutable_view().template end<int>(),
                 0);

    cudf::table_view key_table({*merge_keys});
    cudf::groupby::groupby gb(key_table);
    std::vector<cudf::groupby::aggregation_request> requests;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.push_back(
      cudf::make_merge_tdigest_aggregation<cudf::groupby_aggregation>(merge_delta));
    requests.push_back({merge_values, std::move(aggregations)});
    auto result = gb.aggregate(requests);
    return std::move(result.second[0].results[0]);
  }
};
}  // namespace

template <typename T>
struct TDigestAllTypes : public cudf::test::BaseFixture {};
TYPED_TEST_SUITE(TDigestAllTypes, cudf::test::NumericTypes);

TYPED_TEST(TDigestAllTypes, Simple)
{
  using T = TypeParam;
  cudf::test::tdigest_simple_aggregation<T>(tdigest_groupby_simple_op{});
}

TYPED_TEST(TDigestAllTypes, SimpleWithNulls)
{
  using T = TypeParam;
  cudf::test::tdigest_simple_with_nulls_aggregation<T>(tdigest_groupby_simple_op{});
}

TYPED_TEST(TDigestAllTypes, AllNull)
{
  using T = TypeParam;
  cudf::test::tdigest_simple_all_nulls_aggregation<T>(tdigest_groupby_simple_op{});
}

TYPED_TEST(TDigestAllTypes, LargeGroups)
{
  auto _values = cudf::test::generate_standardized_percentile_distribution(
    cudf::data_type{cudf::type_id::FLOAT64});
  int const delta = 1000;

  // generate a random set of keys
  std::vector<int> h_keys;
  h_keys.reserve(_values->size());
  auto iter = thrust::make_counting_iterator(0);
  std::transform(iter, iter + _values->size(), std::back_inserter(h_keys), [](int i) {
    return static_cast<int>(round(cudf::test::rand_range(0, 8)));
  });
  cudf::test::fixed_width_column_wrapper<int> _keys(h_keys.begin(), h_keys.end());

  // group the input values together
  cudf::table_view k({_keys});
  cudf::groupby::groupby setup_gb(k);
  cudf::table_view v({*_values});
  auto groups = setup_gb.get_groups(v);

  // slice it all up so we have keys/columns for everything.
  std::vector<cudf::column_view> keys;
  std::vector<cudf::column_view> values;
  for (size_t idx = 0; idx < groups.offsets.size() - 1; idx++) {
    auto k =
      cudf::slice(groups.keys->get_column(0), {groups.offsets[idx], groups.offsets[idx + 1]});
    keys.push_back(k[0]);

    auto v =
      cudf::slice(groups.values->get_column(0), {groups.offsets[idx], groups.offsets[idx + 1]});
    values.push_back(v[0]);
  }

  // generate a separate tdigest for each group
  std::vector<std::unique_ptr<cudf::column>> parts;
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
  std::vector<cudf::column_view> part_views;
  std::transform(parts.begin(),
                 parts.end(),
                 std::back_inserter(part_views),
                 [](std::unique_ptr<cudf::column> const& col) { return col->view(); });
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

struct TDigestTest : public cudf::test::BaseFixture {};

TEST_F(TDigestTest, EmptyMixed)
{
  cudf::test::fixed_width_column_wrapper<double> values{
    {123456.78, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0}, {1, 0, 0, 1, 0, 0, 1, 1, 0}};
  cudf::test::strings_column_wrapper keys{"b", "a", "c", "c", "d", "d", "e", "e", "f"};

  auto const delta = 1000;
  cudf::table_view t({keys});
  cudf::groupby::groupby gb(t);
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
  requests.push_back({values, std::move(aggregations)});
  auto result = gb.aggregate(requests);

  using FCW = cudf::test::fixed_width_column_wrapper<double>;
  auto expected =
    cudf::test::make_expected_tdigest_column({{FCW{}, FCW{}, 0, 0},
                                              {FCW{123456.78}, FCW{1.0}, 123456.78, 123456.78},
                                              {FCW{25.0}, FCW{1.0}, 25.0, 25.0},
                                              {FCW{}, FCW{}, 0, 0},
                                              {FCW{50.0, 60.0}, FCW{1.0, 1.0}, 50.0, 60.0},
                                              {FCW{}, FCW{}, 0, 0}});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result.second[0].results[0], *expected);
}

TEST_F(TDigestTest, LargeInputDouble)
{
  cudf::test::tdigest_simple_large_input_double_aggregation(tdigest_groupby_simple_op{});
}

TEST_F(TDigestTest, LargeInputInt)
{
  cudf::test::tdigest_simple_large_input_int_aggregation(tdigest_groupby_simple_op{});
}

TEST_F(TDigestTest, LargeInputDecimal)
{
  cudf::test::tdigest_simple_large_input_decimal_aggregation(tdigest_groupby_simple_op{});
}

struct TDigestMergeTest : public cudf::test::BaseFixture {};

// Note: there is no need to test different types here as the internals of a tdigest are always
// the same regardless of input.
TEST_F(TDigestMergeTest, Simple)
{
  cudf::test::tdigest_merge_simple(tdigest_groupby_simple_op{}, tdigest_groupby_simple_merge_op{});
}

struct key_groups {
  __device__ cudf::size_type operator()(cudf::size_type i) { return i < 250000 ? 0 : 1; }
};
TEST_F(TDigestMergeTest, Grouped)
{
  auto values = cudf::test::generate_standardized_percentile_distribution(
    cudf::data_type{cudf::type_id::FLOAT64});
  ASSERT_EQ(values->size(), 750000);
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, values->size(), cudf::mask_state::UNALLOCATED);
  // 3 groups. 0-250000 in group 0.  250000-500000 in group 1 and 500000-750000 in group 1
  auto key_iter = cudf::detail::make_counting_transform_iterator(0, key_groups{});
  thrust::copy(rmm::exec_policy(cudf::get_default_stream()),
               key_iter,
               key_iter + keys->size(),
               keys->mutable_view().template begin<int>());

  auto split_values         = cudf::split(*values, {250000, 500000});
  auto grouped_split_values = cudf::split(*values, {250000});
  auto split_keys           = cudf::split(*keys, {250000, 500000});

  int const delta = 1000;

  // generate separate digests
  std::vector<std::unique_ptr<cudf::column>> parts;
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
  std::vector<cudf::column_view> part_views;
  std::transform(parts.begin(),
                 parts.end(),
                 std::back_inserter(part_views),
                 [](std::unique_ptr<cudf::column> const& col) { return col->view(); });

  // merge delta = 1000
  {
    int const merge_delta = 1000;

    // merge them
    auto merge_input = cudf::concatenate(part_views);
    cudf::test::fixed_width_column_wrapper<int> merge_keys{0, 1, 1};
    cudf::table_view key_table({merge_keys});
    cudf::groupby::groupby gb(key_table);
    std::vector<cudf::groupby::aggregation_request> requests;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.push_back(
      cudf::make_merge_tdigest_aggregation<cudf::groupby_aggregation>(merge_delta));
    requests.push_back({*merge_input, std::move(aggregations)});
    auto result = gb.aggregate(requests);

    ASSERT_EQ(result.second[0].results[0]->size(), 2);
    cudf::tdigest::tdigest_column_view tdv(*result.second[0].results[0]);

    // verify centroids
    std::vector<cudf::test::expected_value> expected{// group 0
                                                     {0, 0.00013945158577498588, 2},
                                                     {10, 0.04804393446447509375, 50},
                                                     {66, 2.10089484962640948851, 316},
                                                     {139, 8.92977366346101852912, 601},
                                                     {243, 23.89152910016953867967, 784},
                                                     {366, 41.62636569363655780762, 586},
                                                     {432, 47.73085102980330418632, 326},
                                                     {460, 49.20637897385523018556, 196},
                                                     {501, 49.99998311512171511595, 1},
                                                     // group 1
                                                     {502 + 0, 50.00022508669655252334, 2},
                                                     {502 + 15, 50.05415694538910287292, 74},
                                                     {502 + 70, 51.21421484112906341579, 334},
                                                     {502 + 150, 55.19367617848146778670, 635},
                                                     {502 + 260, 63.24605285552920008740, 783},
                                                     {502 + 380, 76.99522005804017510400, 1289},
                                                     {502 + 440, 84.22673817294192133431, 758},
                                                     {502 + 490, 88.11787981529532487457, 784},
                                                     {502 + 555, 93.02766411136053648079, 704},
                                                     {502 + 618, 96.91486035315536184953, 516},
                                                     {502 + 710, 99.87755861436669135855, 110},
                                                     {502 + 733, 99.99970905482754801596, 1}};
    cudf::test::tdigest_sample_compare(tdv, expected);

    // verify min/max
    auto split_results = cudf::split(*result.second[0].results[0], {1});
    auto iter          = thrust::make_counting_iterator(0);
    std::for_each(iter, iter + split_results.size(), [&](cudf::size_type i) {
      auto copied = std::make_unique<cudf::column>(split_results[i]);
      cudf::test::tdigest_minmax_compare<double>(cudf::tdigest::tdigest_column_view(*copied),
                                                 grouped_split_values[i]);
    });
  }

  // merge delta = 100
  {
    int const merge_delta = 100;

    // merge them
    auto merge_input = cudf::concatenate(part_views);
    cudf::test::fixed_width_column_wrapper<int> merge_keys{0, 1, 1};
    cudf::table_view key_table({merge_keys});
    cudf::groupby::groupby gb(key_table);
    std::vector<cudf::groupby::aggregation_request> requests;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.push_back(
      cudf::make_merge_tdigest_aggregation<cudf::groupby_aggregation>(merge_delta));
    requests.push_back({*merge_input, std::move(aggregations)});
    auto result = gb.aggregate(requests);

    ASSERT_EQ(result.second[0].results[0]->size(), 2);
    cudf::tdigest::tdigest_column_view tdv(*result.second[0].results[0]);

    // verify centroids
    std::vector<cudf::test::expected_value> expected{// group 0
                                                     {0, 0.02182479870203561656, 231},
                                                     {3, 0.60625795002234528219, 1688},
                                                     {13, 8.40462931740497687372, 5867},
                                                     {27, 28.79997783486397722186, 7757},
                                                     {35, 40.22391421196020644402, 6224},
                                                     {45, 48.96506331299028857984, 2225},
                                                     {50, 49.99979491345574444949, 4},
                                                     // group 1
                                                     {51 + 0, 50.02171921312970681583, 460},
                                                     {51 + 5, 51.45308398121498072442, 5074},
                                                     {51 + 11, 55.96880716301625113829, 10011},
                                                     {51 + 22, 70.18029861315150697010, 15351},
                                                     {51 + 38, 92.65943436519887654867, 10718},
                                                     {51 + 47, 99.27745505225347244505, 3639}};
    cudf::test::tdigest_sample_compare(tdv, expected);

    // verify min/max
    auto split_results = cudf::split(*result.second[0].results[0], {1});
    auto iter          = thrust::make_counting_iterator(0);
    std::for_each(iter, iter + split_results.size(), [&](cudf::size_type i) {
      auto copied = std::make_unique<cudf::column>(split_results[i]);
      cudf::test::tdigest_minmax_compare<double>(cudf::tdigest::tdigest_column_view(*copied),
                                                 grouped_split_values[i]);
    });
  }

  // merge delta = 10
  {
    int const merge_delta = 10;

    // merge them
    auto merge_input = cudf::concatenate(part_views);
    cudf::test::fixed_width_column_wrapper<int> merge_keys{0, 1, 1};
    cudf::table_view key_table({merge_keys});
    cudf::groupby::groupby gb(key_table);
    std::vector<cudf::groupby::aggregation_request> requests;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.push_back(
      cudf::make_merge_tdigest_aggregation<cudf::groupby_aggregation>(merge_delta));
    requests.push_back({*merge_input, std::move(aggregations)});
    auto result = gb.aggregate(requests);

    ASSERT_EQ(result.second[0].results[0]->size(), 2);
    cudf::tdigest::tdigest_column_view tdv(*result.second[0].results[0]);

    // verify centroids
    std::vector<cudf::test::expected_value> expected{// group 0
                                                     {0, 2.34644806683495144028, 23623},
                                                     {1, 10.95523693698660672169, 62290},
                                                     {2, 24.90731657803452847588, 77208},
                                                     {3, 38.88062495289155862110, 62658},
                                                     {4, 47.56288303840698006297, 24217},
                                                     {5, 49.99979491345574444949, 4},
                                                     // group 1
                                                     {6 + 0, 52.40174463129091719793, 47410},
                                                     {6 + 1, 60.97025126481504031517, 124564},
                                                     {6 + 2, 74.91722742839780835311, 154387},
                                                     {6 + 3, 88.87559489177009197647, 124810},
                                                     {6 + 4, 97.55823307073454486726, 48817},
                                                     {6 + 5, 99.99901807905750672489, 12}};
    cudf::test::tdigest_sample_compare(tdv, expected);

    // verify min/max
    auto split_results = cudf::split(*result.second[0].results[0], {1});
    auto iter          = thrust::make_counting_iterator(0);
    std::for_each(iter, iter + split_results.size(), [&](cudf::size_type i) {
      auto copied = std::make_unique<cudf::column>(split_results[i]);
      cudf::test::tdigest_minmax_compare<double>(cudf::tdigest::tdigest_column_view(*copied),
                                                 grouped_split_values[i]);
    });
  }
}

TEST_F(TDigestMergeTest, Empty)
{
  cudf::test::tdigest_merge_empty(tdigest_groupby_simple_merge_op{});
}

TEST_F(TDigestMergeTest, EmptyGroups)
{
  cudf::test::fixed_width_column_wrapper<double> values_b{{126, 15, 1, 99, 67, 55, 2},
                                                          {1, 0, 0, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> values_d{{100, 200, 300, 400, 500, 600, 700},
                                                          {1, 1, 1, 1, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int> keys{0, 0, 0, 0, 0, 0, 0};
  int const delta = 1000;

  auto a = cudf::tdigest::detail::make_empty_tdigests_column(
    1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto b = cudf::type_dispatcher(
    static_cast<cudf::column_view>(values_b).type(), tdigest_gen_grouped{}, keys, values_b, delta);
  auto c = cudf::tdigest::detail::make_empty_tdigests_column(
    1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto d = cudf::type_dispatcher(
    static_cast<cudf::column_view>(values_d).type(), tdigest_gen_grouped{}, keys, values_d, delta);
  auto e = cudf::tdigest::detail::make_empty_tdigests_column(
    1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  std::vector<cudf::column_view> cols;
  cols.push_back(*a);
  cols.push_back(*b);
  cols.push_back(*c);
  cols.push_back(*d);
  cols.push_back(*e);
  auto values = cudf::concatenate(cols);

  cudf::test::fixed_width_column_wrapper<int> merge_keys{0, 0, 1, 0, 2};

  cudf::table_view t({merge_keys});
  cudf::groupby::groupby gb(t);
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.push_back(cudf::make_merge_tdigest_aggregation<cudf::groupby_aggregation>(delta));
  requests.push_back({*values, std::move(aggregations)});
  auto result = gb.aggregate(requests);

  using FCW = cudf::test::fixed_width_column_wrapper<double>;
  cudf::test::fixed_width_column_wrapper<double> expected_means{
    2, 55, 67, 99, 100, 126, 200, 300, 400, 500, 600};
  cudf::test::fixed_width_column_wrapper<double> expected_weights{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto expected = cudf::test::make_expected_tdigest_column(
    {{expected_means, expected_weights, 2, 600}, {FCW{}, FCW{}, 0, 0}, {FCW{}, FCW{}, 0, 0}});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *result.second[0].results[0]);
}

namespace {
std::unique_ptr<cudf::table> do_agg(
  cudf::column_view key,
  cudf::column_view val,
  std::function<std::unique_ptr<cudf::groupby_aggregation>()> make_agg)
{
  std::vector<cudf::column_view> keys;
  keys.push_back(key);
  cudf::table_view const key_table(keys);

  cudf::groupby::groupby gb(key_table);
  std::vector<cudf::groupby::aggregation_request> requests;
  cudf::groupby::aggregation_request req;
  req.values = val;
  req.aggregations.push_back(make_agg());
  requests.push_back(std::move(req));

  auto result = gb.aggregate(std::move(requests));

  std::vector<std::unique_ptr<cudf::column>> result_columns;
  for (auto&& c : result.first->release()) {
    result_columns.push_back(std::move(c));
  }

  EXPECT_EQ(result.second.size(), 1);
  EXPECT_EQ(result.second[0].results.size(), 1);
  result_columns.push_back(std::move(result.second[0].results[0]));

  return std::make_unique<cudf::table>(std::move(result_columns));
}
}  // namespace

TEST_F(TDigestMergeTest, AllValuesAreNull)
{
  // The input must be sorted by the key.
  // See `aggregate_result_functor::operator()<aggregation::TDIGEST>` for details.
  auto const keys      = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 0, 1, 1, 2}};
  auto const keys_view = cudf::column_view(keys);
  auto val_elems  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto val_valids = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    // All values are null
    return false;
  });
  auto const vals = cudf::test::fixed_width_column_wrapper<int32_t>{
    val_elems, val_elems + keys_view.size(), val_valids};

  auto const delta = 1000;

  // Compute tdigest. The result should have 3 empty clusters, one per group.
  auto const compute_result = do_agg(keys_view, cudf::column_view(vals), [&delta]() {
    return cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta);
  });

  auto const expected_computed_keys = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 1, 2}};
  cudf::column_view const expected_computed_keys_view{expected_computed_keys};
  auto const expected_computed_vals =
    cudf::tdigest::detail::make_empty_tdigests_column(expected_computed_keys_view.size(),
                                                      cudf::get_default_stream(),
                                                      rmm::mr::get_current_device_resource());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_computed_keys_view, compute_result->get_column(0).view());
  // The computed values are nullable even though the input values are not.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_computed_vals->view(),
                                 compute_result->get_column(1).view());

  // Merge tdigest. The result should have 3 empty clusters, one per group.
  auto const merge_result =
    do_agg(compute_result->get_column(0).view(), compute_result->get_column(1).view(), [&delta]() {
      return cudf::make_merge_tdigest_aggregation<cudf::groupby_aggregation>(delta);
    });

  auto const expected_merged_keys = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 1, 2}};
  cudf::column_view const expected_merged_keys_view{expected_merged_keys};
  auto const expected_merged_vals =
    cudf::tdigest::detail::make_empty_tdigests_column(expected_merged_keys_view.size(),
                                                      cudf::get_default_stream(),
                                                      rmm::mr::get_current_device_resource());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_merged_keys_view, merge_result->get_column(0).view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_merged_vals->view(), merge_result->get_column(1).view());
}

TEST_F(TDigestMergeTest, AllValuesInOneGroupIsNull)
{
  cudf::test::fixed_width_column_wrapper<int> keys{0, 1, 2, 2, 3};
  cudf::test::fixed_width_column_wrapper<double> vals{{10.0, 20.0, {}, {}, 30.0},
                                                      {true, true, false, false, true}};

  auto const delta = 1000;

  // Compute tdigest. The result should have 3 empty clusters, one per group.
  auto const compute_result = do_agg(cudf::column_view(keys), cudf::column_view(vals), [&delta]() {
    return cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta);
  });

  auto const expected_keys = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 1, 2, 3}};

  cudf::test::fixed_width_column_wrapper<double> expected_means{10, 20, 30};
  cudf::test::fixed_width_column_wrapper<double> expected_weights{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_offsets{0, 1, 2, 2, 3};
  cudf::test::fixed_width_column_wrapper<double> expected_mins{10.0, 20.0, 0.0, 30.0};
  cudf::test::fixed_width_column_wrapper<double> expected_maxes{10.0, 20.0, 0.0, 30.0};
  auto const expected_values =
    cudf::tdigest::detail::make_tdigest_column(4,
                                               std::make_unique<cudf::column>(expected_means),
                                               std::make_unique<cudf::column>(expected_weights),
                                               std::make_unique<cudf::column>(expected_offsets),
                                               std::make_unique<cudf::column>(expected_mins),
                                               std::make_unique<cudf::column>(expected_maxes),
                                               cudf::get_default_stream(),
                                               rmm::mr::get_current_device_resource());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::column_view{expected_keys},
                                 compute_result->get_column(0).view());
  // The computed values are nullable even though the input values are not.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_values->view(), compute_result->get_column(1).view());

  // Merge tdigest. The result should have 3 empty clusters, one per group.
  auto const merge_result =
    do_agg(compute_result->get_column(0).view(), compute_result->get_column(1).view(), [&delta]() {
      return cudf::make_merge_tdigest_aggregation<cudf::groupby_aggregation>(delta);
    });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::column_view{expected_keys},
                                 merge_result->get_column(0).view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_values->view(), merge_result->get_column(1).view());
}
