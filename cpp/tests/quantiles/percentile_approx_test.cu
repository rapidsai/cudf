/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <arrow/util/tdigest.h>

#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/groupby.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/reduction.hpp>
#include <cudf/tdigest/tdigest_column_view.cuh>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <rmm/exec_policy.hpp>

#include <tests/groupby/groupby_test_util.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

using namespace cudf;
using namespace cudf::tdigest;

std::unique_ptr<column> arrow_percentile_approx(column_view const& _values,
                                                int delta,
                                                std::vector<double> const& percentages)
{
  // sort the incoming values using the same settings that groupby does.
  // this is a little weak because null_order::AFTER is hardcoded internally to groupby.
  table_view t({_values});
  auto sorted_t      = cudf::sort(t, {}, {null_order::AFTER});
  auto sorted_values = sorted_t->get_column(0).view();

  std::vector<double> h_values(sorted_values.size());
  cudaMemcpy(h_values.data(),
             sorted_values.data<double>(),
             sizeof(double) * sorted_values.size(),
             cudaMemcpyDeviceToHost);
  std::vector<char> h_validity(sorted_values.size());
  if (sorted_values.null_mask() != nullptr) {
    auto validity = cudf::mask_to_bools(sorted_values.null_mask(), 0, sorted_values.size());
    cudaMemcpy(h_validity.data(),
               (validity->view().data<char>()),
               sizeof(char) * sorted_values.size(),
               cudaMemcpyDeviceToHost);
  }

  // generate the tdigest
  arrow::internal::TDigest atd(delta, sorted_values.size() * 2);
  for (size_t idx = 0; idx < h_values.size(); idx++) {
    if (sorted_values.null_mask() == nullptr || h_validity[idx]) { atd.Add(h_values[idx]); }
  }

  // generate the percentiles and stuff them into a list column
  std::vector<double> h_result;
  h_result.reserve(percentages.size());
  std::transform(
    percentages.begin(), percentages.end(), std::back_inserter(h_result), [&atd](double p) {
      return atd.Quantile(p);
    });
  cudf::test::fixed_width_column_wrapper<double> result(h_result.begin(), h_result.end());
  cudf::test::fixed_width_column_wrapper<size_type> offsets{
    0, static_cast<size_type>(percentages.size())};
  return cudf::make_lists_column(1, offsets.release(), result.release(), 0, {});
}

struct percentile_approx_dispatch {
  template <
    typename T,
    typename Func,
    typename std::enable_if_t<cudf::is_numeric<T>() || cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(Func op,
                                     column_view const& values,
                                     int delta,
                                     std::vector<double> const& percentages,
                                     size_type ulps)
  {
    // arrow implementation.
    auto expected = [&]() {
      // we're explicitly casting back to doubles here but this is ok because that is
      // exactly what happens inside of the cudf implementation as values are processed as well. so
      // this should not affect results.
      auto as_doubles = cudf::cast(values, data_type{type_id::FLOAT64});
      return arrow_percentile_approx(*as_doubles, delta, percentages);
    }();

    // gpu implementation
    auto agg_result = op(values, delta);

    cudf::test::fixed_width_column_wrapper<double> g_percentages(percentages.begin(),
                                                                 percentages.end());
    tdigest_column_view tdv(*agg_result);
    auto result = cudf::percentile_approx(tdv, g_percentages);

    cudf::test::expect_columns_equivalent(
      *expected, *result, cudf::test::debug_output_level::FIRST_ERROR, ulps);

    return result;
  }

  template <
    typename T,
    typename Func,
    typename std::enable_if_t<!cudf::is_numeric<T>() && !cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(Func op,
                                     column_view const& values,
                                     int delta,
                                     std::vector<double> const& percentages,
                                     size_type ulps)
  {
    CUDF_FAIL("Invalid input type for percentile_approx test");
  }
};

void percentile_approx_test(column_view const& _keys,
                            column_view const& _values,
                            int delta,
                            std::vector<double> const& percentages,
                            size_type ulps)
{
  // first pass:  validate the actual percentages we get per group.

  // produce the groups.
  cudf::table_view k({_keys});
  cudf::groupby::groupby pass1_gb(k);
  cudf::table_view v({_values});
  auto groups = pass1_gb.get_groups(v);
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

  std::vector<std::unique_ptr<column>> groupby_parts;
  std::vector<std::unique_ptr<column>> reduce_parts;
  for (size_t idx = 0; idx < values.size(); idx++) {
    // via groupby
    auto groupby = [&](column_view const& values, int delta) {
      cudf::table_view t({keys[idx]});
      cudf::groupby::groupby gb(t);
      std::vector<cudf::groupby::aggregation_request> requests;
      std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
      aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
      requests.push_back({values, std::move(aggregations)});
      return std::move(gb.aggregate(requests).second[0].results[0]);
    };
    groupby_parts.push_back(cudf::type_dispatcher(values[idx].type(),
                                                  percentile_approx_dispatch{},
                                                  groupby,
                                                  values[idx],
                                                  delta,
                                                  percentages,
                                                  ulps));

    // via reduce
    auto reduce = [](column_view const& values, int delta) {
      // result is a scalar, but we want to extract out the underlying column
      auto scalar_result =
        cudf::reduce(values,
                     *cudf::make_tdigest_aggregation<cudf::reduce_aggregation>(delta),
                     data_type{type_id::STRUCT});
      auto tbl = static_cast<cudf::struct_scalar const*>(scalar_result.get())->view();
      std::vector<std::unique_ptr<cudf::column>> cols;
      std::transform(
        tbl.begin(), tbl.end(), std::back_inserter(cols), [](cudf::column_view const& col) {
          return std::make_unique<cudf::column>(col);
        });
      return cudf::make_structs_column(tbl.num_rows(), std::move(cols), 0, rmm::device_buffer());
    };
    // groupby path
    reduce_parts.push_back(cudf::type_dispatcher(values[idx].type(),
                                                 percentile_approx_dispatch{},
                                                 reduce,
                                                 values[idx],
                                                 delta,
                                                 percentages,
                                                 ulps));
  }

  // second pass. run the percentile_approx with all the keys in one pass and make sure we get the
  // same results as the concatenated by-key results.
  std::vector<column_view> part_views;
  std::transform(groupby_parts.begin(),
                 groupby_parts.end(),
                 std::back_inserter(part_views),
                 [](std::unique_ptr<column> const& c) { return c->view(); });
  auto expected = cudf::concatenate(part_views);

  cudf::groupby::groupby gb(k);
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
  requests.push_back({_values, std::move(aggregations)});
  auto gb_result = gb.aggregate(requests);

  cudf::test::fixed_width_column_wrapper<double> g_percentages(percentages.begin(),
                                                               percentages.end());
  tdigest_column_view tdv(*(gb_result.second[0].results[0]));
  auto result = cudf::percentile_approx(tdv, g_percentages);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *result);
}

void simple_test(data_type input_type, std::vector<std::pair<int, int>> params)
{
  auto values = cudf::test::generate_standardized_percentile_distribution(input_type);
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, values->size(), mask_state::UNALLOCATED);
  thrust::fill(rmm::exec_policy(cudf::get_default_stream()),
               keys->mutable_view().template begin<int>(),
               keys->mutable_view().template end<int>(),
               0);

  // runs both groupby and reduce paths
  std::for_each(params.begin(), params.end(), [&](std::pair<int, int> const& params) {
    percentile_approx_test(
      *keys, *values, params.first, {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0}, params.second);
  });
}

struct group_index {
  __device__ int operator()(int i) { return i / 150000; }
};

void grouped_test(data_type input_type, std::vector<std::pair<int, int>> params)
{
  auto values = cudf::test::generate_standardized_percentile_distribution(input_type);
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, values->size(), mask_state::UNALLOCATED);
  auto i = thrust::make_counting_iterator(0);
  thrust::transform(rmm::exec_policy(cudf::get_default_stream()),
                    i,
                    i + values->size(),
                    keys->mutable_view().template begin<int>(),
                    group_index{});

  std::for_each(params.begin(), params.end(), [&](std::pair<int, int> const& params) {
    percentile_approx_test(
      *keys, *values, params.first, {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0}, params.second);
  });
}

std::pair<rmm::device_buffer, size_type> make_null_mask(column_view const& col)
{
  return cudf::detail::valid_if(thrust::make_counting_iterator<size_type>(0),
                                thrust::make_counting_iterator<size_type>(col.size()),
                                [] __device__(size_type i) { return i % 2 == 0; });
}

void simple_with_nulls_test(data_type input_type, std::vector<std::pair<int, int>> params)
{
  auto values = cudf::test::generate_standardized_percentile_distribution(input_type);
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, values->size(), mask_state::UNALLOCATED);
  thrust::fill(rmm::exec_policy(cudf::get_default_stream()),
               keys->mutable_view().template begin<int>(),
               keys->mutable_view().template end<int>(),
               0);

  // add a null mask
  auto mask = make_null_mask(*values);
  values->set_null_mask(mask.first, mask.second);

  std::for_each(params.begin(), params.end(), [&](std::pair<int, int> const& params) {
    percentile_approx_test(
      *keys, *values, params.first, {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0}, params.second);
  });
}

void grouped_with_nulls_test(data_type input_type, std::vector<std::pair<int, int>> params)
{
  auto values = cudf::test::generate_standardized_percentile_distribution(input_type);
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, values->size(), mask_state::UNALLOCATED);
  auto i = thrust::make_counting_iterator(0);
  thrust::transform(rmm::exec_policy(cudf::get_default_stream()),
                    i,
                    i + values->size(),
                    keys->mutable_view().template begin<int>(),
                    group_index{});

  // add a null mask
  auto mask = make_null_mask(*values);
  values->set_null_mask(mask.first, mask.second);

  std::for_each(params.begin(), params.end(), [&](std::pair<int, int> const& params) {
    percentile_approx_test(
      *keys, *values, params.first, {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0}, params.second);
  });
}

template <typename T>
data_type get_appropriate_type()
{
  if constexpr (cudf::is_fixed_point<T>()) { return data_type{cudf::type_to_id<T>(), -7}; }
  return data_type{cudf::type_to_id<T>()};
}

using PercentileApproxTypes =
  cudf::test::Concat<cudf::test::NumericTypes, cudf::test::FixedPointTypes>;

template <typename T>
struct PercentileApproxInputTypesTest : public cudf::test::BaseFixture {
};
TYPED_TEST_SUITE(PercentileApproxInputTypesTest, PercentileApproxTypes);

TYPED_TEST(PercentileApproxInputTypesTest, Simple)
{
  using T               = TypeParam;
  auto const input_type = get_appropriate_type<T>();

  simple_test(input_type,
              {{1000, cudf::test::default_ulp},
               {100, cudf::test::default_ulp * 4},
               {10, cudf::test::default_ulp * 11}});
}

TYPED_TEST(PercentileApproxInputTypesTest, Grouped)
{
  using T               = TypeParam;
  auto const input_type = get_appropriate_type<T>();

  grouped_test(input_type,
               {{1000, cudf::test::default_ulp},
                {100, cudf::test::default_ulp * 2},
                {10, cudf::test::default_ulp * 10}});
}

TYPED_TEST(PercentileApproxInputTypesTest, SimpleWithNulls)
{
  using T               = TypeParam;
  auto const input_type = get_appropriate_type<T>();

  simple_with_nulls_test(input_type,
                         {{1000, cudf::test::default_ulp},
                          {100, cudf::test::default_ulp * 2},
                          {10, cudf::test::default_ulp * 11}});
}

TYPED_TEST(PercentileApproxInputTypesTest, GroupedWithNulls)
{
  using T               = TypeParam;
  auto const input_type = get_appropriate_type<T>();

  grouped_with_nulls_test(input_type,
                          {{1000, cudf::test::default_ulp},
                           {100, cudf::test::default_ulp * 2},
                           {10, cudf::test::default_ulp * 6}});
}

struct PercentileApproxTest : public cudf::test::BaseFixture {
};

TEST_F(PercentileApproxTest, EmptyInput)
{
  auto empty_ = cudf::detail::tdigest::make_empty_tdigest_column();
  cudf::test::fixed_width_column_wrapper<double> percentiles{0.0, 0.25, 0.3};

  std::vector<column_view> input;
  input.push_back(*empty_);
  input.push_back(*empty_);
  input.push_back(*empty_);
  auto empty = cudf::concatenate(input);

  tdigest_column_view tdv(*empty);
  auto result = cudf::percentile_approx(tdv, percentiles);

  cudf::test::fixed_width_column_wrapper<offset_type> offsets{0, 0, 0, 0};
  std::vector<bool> nulls{0, 0, 0};
  auto expected =
    cudf::make_lists_column(3,
                            offsets.release(),
                            cudf::make_empty_column(type_id::FLOAT64),
                            3,
                            cudf::test::detail::make_null_mask(nulls.begin(), nulls.end()));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(PercentileApproxTest, EmptyPercentiles)
{
  auto const delta = 1000;

  cudf::test::fixed_width_column_wrapper<double> values{0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<int> keys{0, 0, 0, 1, 1, 1};
  cudf::table_view t({keys});
  cudf::groupby::groupby gb(t);
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
  requests.push_back({values, std::move(aggregations)});
  auto tdigest_column = gb.aggregate(requests);

  cudf::test::fixed_width_column_wrapper<double> percentiles{};

  tdigest_column_view tdv(*tdigest_column.second[0].results[0]);
  auto result = cudf::percentile_approx(tdv, percentiles);

  cudf::test::fixed_width_column_wrapper<offset_type> offsets{0, 0, 0};
  auto expected = cudf::make_lists_column(2,
                                          offsets.release(),
                                          cudf::make_empty_column(type_id::FLOAT64),
                                          2,
                                          cudf::detail::create_null_mask(2, mask_state::ALL_NULL));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(PercentileApproxTest, NullPercentiles)
{
  auto const delta = 1000;

  cudf::test::fixed_width_column_wrapper<double> values{1, 1, 2, 3, 4, 5, 6, 7, 8};
  cudf::test::fixed_width_column_wrapper<int> keys{0, 0, 0, 0, 0, 1, 1, 1, 1};
  cudf::table_view t({keys});
  cudf::groupby::groupby gb(t);
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
  requests.push_back({values, std::move(aggregations)});
  auto tdigest_column = gb.aggregate(requests);

  tdigest_column_view tdv(*tdigest_column.second[0].results[0]);

  cudf::test::fixed_width_column_wrapper<double> npercentiles{{0.5, 0.5, 1.0, 1.0}, {0, 0, 1, 1}};
  auto result = cudf::percentile_approx(tdv, npercentiles);

  std::vector<bool> valids{0, 0, 1, 1};
  cudf::test::lists_column_wrapper<double> expected{{{99, 99, 4, 4}, valids.begin()},
                                                    {{99, 99, 8, 8}, valids.begin()}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}
