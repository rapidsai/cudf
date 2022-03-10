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

#pragma once

#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/groupby.hpp>
#include <cudf/tdigest/tdigest_column_view.cuh>

#include <cudf_test/column_wrapper.hpp>

#include <tests/groupby/groupby_test_util.hpp>

#include <thrust/extrema.h>

#include <rmm/exec_policy.hpp>

// for use with groupby and reduction aggregation tests.

namespace cudf {
namespace test {

using expected_value = thrust::tuple<size_type, double, double>;

template <typename T>
struct column_min {
  __device__ double operator()(device_span<T const> vals)
  {
    return static_cast<double>(*thrust::min_element(thrust::seq, vals.begin(), vals.end()));
  }
};

template <typename T>
struct column_max {
  __device__ double operator()(device_span<T const> vals)
  {
    return static_cast<double>(*thrust::max_element(thrust::seq, vals.begin(), vals.end()));
  }
};

struct tdigest_gen_grouped {
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

struct tdigest_gen {
  template <
    typename T,
    typename Func,
    typename std::enable_if_t<cudf::is_numeric<T>() || cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(Func op, column_view const& values, int delta)
  {
    return op(values, delta);
  }

  template <
    typename T,
    typename Func,
    typename std::enable_if_t<!cudf::is_numeric<T>() && !cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(Func op, column_view const& values, int delta)
  {
    CUDF_FAIL("Invalid tdigest test type");
  }
};

void tdigest_sample_compare(cudf::tdigest::tdigest_column_view const& tdv,
                            std::vector<expected_value> const& h_expected);

template <typename T>
void tdigest_minmax_compare(cudf::tdigest::tdigest_column_view const& tdv,
                            column_view const& input_values)
{
  // verify min/max
  thrust::host_vector<device_span<T const>> h_spans;
  h_spans.push_back({input_values.begin<T>(), static_cast<size_t>(input_values.size())});
  thrust::device_vector<device_span<T const>> spans(h_spans);

  auto expected_min = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, spans.size(), mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy(rmm::cuda_stream_default),
                    spans.begin(),
                    spans.end(),
                    expected_min->mutable_view().template begin<double>(),
                    column_min<T>{});
  column_view result_min(data_type{type_id::FLOAT64}, tdv.size(), tdv.min_begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_min, *expected_min);

  auto expected_max = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, spans.size(), mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy(rmm::cuda_stream_default),
                    spans.begin(),
                    spans.end(),
                    expected_max->mutable_view().template begin<double>(),
                    column_max<T>{});
  column_view result_max(data_type{type_id::FLOAT64}, tdv.size(), tdv.max_begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_max, *expected_max);
}

struct expected_tdigest {
  column_view mean;
  column_view weight;
  double min, max;
};

std::unique_ptr<column> make_expected_tdigest_column(std::vector<expected_tdigest> const& groups);

// shared tests for groupby/reduction.
template <typename T, typename Func>
void simple_tdigest_aggregation(Func op)
{
  // create a tdigest that has far fewer values in it than the delta value. this should result
  // in every value remaining uncompressed
  cudf::test::fixed_width_column_wrapper<T> values{126, 15, 1, 99, 67};
  int const delta = 1000;
  auto result     = cudf::type_dispatcher(
    static_cast<column_view>(values).type(), tdigest_gen{}, op, values, delta);

  cudf::test::fixed_width_column_wrapper<T> raw_mean({1, 15, 67, 99, 126});
  cudf::test::fixed_width_column_wrapper<double> weight{1, 1, 1, 1, 1};
  auto mean        = cudf::cast(raw_mean, data_type{type_id::FLOAT64});
  double const min = 1;
  double const max = 126;
  auto expected    = make_expected_tdigest_column({{*mean,
                                                 weight,
                                                 static_cast<double>(static_cast<T>(min)),
                                                 static_cast<double>(static_cast<T>(max))}});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

template <typename T, typename Func>
void simple_with_null_tdigest_aggregation(Func op)
{
  // create a tdigest that has far fewer values in it than the delta value. this should result
  // in every value remaining uncompressed
  cudf::test::fixed_width_column_wrapper<T> values{{122, 15, 1, 99, 67, 101, 100, 84, 44, 2},
                                                   {1, 0, 1, 0, 1, 0, 1, 0, 1, 0}};
  int const delta = 1000;
  auto result     = cudf::type_dispatcher(
    static_cast<column_view>(values).type(), tdigest_gen{}, op, values, delta);

  cudf::test::fixed_width_column_wrapper<T> raw_mean({1, 44, 67, 100, 122});
  cudf::test::fixed_width_column_wrapper<double> weight{1, 1, 1, 1, 1};
  auto mean        = cudf::cast(raw_mean, data_type{type_id::FLOAT64});
  double const min = 1;
  double const max = 122;
  auto expected    = make_expected_tdigest_column({{*mean,
                                                 weight,
                                                 static_cast<double>(static_cast<T>(min)),
                                                 static_cast<double>(static_cast<T>(max))}});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

template <typename T, typename Func>
void simple_all_null_tdigest_aggregation(Func op)
{
  // create a tdigest that has far fewer values in it than the delta value. this should result
  // in every value remaining uncompressed
  cudf::test::fixed_width_column_wrapper<T> values{{122, 15, 1, 99, 67, 101, 100, 84, 44, 2},
                                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  int const delta = 1000;
  auto result     = cudf::type_dispatcher(
    static_cast<column_view>(values).type(), tdigest_gen{}, op, values, delta);

  // NOTE: an empty tdigest column still has 1 row.
  auto expected = cudf::detail::tdigest::make_empty_tdigest_column();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

template <typename Func>
void simple_large_input_double_tdigest_aggregation(Func op)
{
  // these tests are being done explicitly because of the way we have to precompute the correct
  // answers. since the input values generated by the generate_distribution() function below are
  // cast to specific types -before- being sent into the aggregation, I can't (safely) just use the
  // expected values that you get when using doubles all the way through.  so I have to pregenerate
  // the correct answers for each type by hand. so, we'll choose a reasonable subset (double,
  // decimal, int, bool)

  auto values = generate_standardized_percentile_distribution(data_type{type_id::FLOAT64});

  // compare against a sample of known/expected values (which themselves were verified against the
  // Arrow implementation)

  // delta 1000
  {
    int const delta = 1000;
    auto result = cudf::type_dispatcher(values->view().type(), tdigest_gen{}, op, *values, delta);
    std::vector<expected_value> expected{{0, 0.00040692343794663995, 7},
                                         {10, 0.16234555627091204477, 153},
                                         {59, 5.12764811246045937310, 858},
                                         {250, 62.54581814492237157310, 2356},
                                         {368, 87.85834376680742252574, 1735},
                                         {409, 94.07685720279611985006, 1272},
                                         {491, 99.94197663121231300920, 130},
                                         {500, 99.99969880795092080916, 2}};
    cudf::tdigest::tdigest_column_view tdv(*result);

    tdigest_sample_compare(tdv, expected);

    // verify min/max
    tdigest_minmax_compare<double>(tdv, *values);
  }

  // delta 100
  {
    int const delta = 100;
    auto result = cudf::type_dispatcher(values->view().type(), tdigest_gen{}, op, *values, delta);
    std::vector<expected_value> expected{{0, 0.07265722021410986331, 739},
                                         {7, 8.19766194442652640362, 10693},
                                         {16, 36.82277869518204482802, 20276},
                                         {29, 72.95424834129075009059, 22623},
                                         {38, 90.61229683516096145013, 15581},
                                         {46, 99.07283498858802772702, 5142},
                                         {50, 99.99970905482754801596, 1}};
    cudf::tdigest::tdigest_column_view tdv(*result);

    tdigest_sample_compare(tdv, expected);

    // verify min/max
    tdigest_minmax_compare<double>(tdv, *values);
  }

  // delta 10
  {
    int const delta = 10;
    auto result = cudf::type_dispatcher(values->view().type(), tdigest_gen{}, op, *values, delta);
    std::vector<expected_value> expected{{0, 7.15508346777729631327, 71618},
                                         {1, 33.04971680740474226923, 187499},
                                         {2, 62.50566666553867634093, 231762},
                                         {3, 83.46216572053654658703, 187500},
                                         {4, 96.42204425201593664951, 71620},
                                         {5, 99.99970905482754801596, 1}};
    cudf::tdigest::tdigest_column_view tdv(*result);

    tdigest_sample_compare(tdv, expected);

    // verify min/max
    tdigest_minmax_compare<double>(tdv, *values);
  }
}

template <typename Func>
void simple_large_input_int_tdigest_aggregation(Func op)
{
  // these tests are being done explicitly because of the way we have to precompute the correct
  // answers. since the input values generated by the generate_distribution() function below are
  // cast to specific types -before- being sent into the aggregation, I can't (safely) just use the
  // expected values that you get when using doubles all the way through.  so I have to pregenerate
  // the correct answers for each type by hand. so, we'll choose a reasonable subset (double,
  // decimal, int, bool)

  auto values = generate_standardized_percentile_distribution(data_type{type_id::INT32});

  // compare against a sample of known/expected values (which themselves were verified against the
  // Arrow implementation)

  // delta 1000
  {
    int const delta = 1000;
    auto result = cudf::type_dispatcher(values->view().type(), tdigest_gen{}, op, *values, delta);
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
    cudf::tdigest::tdigest_column_view tdv(*result);

    tdigest_sample_compare(tdv, expected);

    // verify min/max
    tdigest_minmax_compare<int>(tdv, *values);
  }

  // delta 100
  {
    int const delta = 100;
    auto result = cudf::type_dispatcher(values->view().type(), tdigest_gen{}, op, *values, delta);
    std::vector<expected_value> expected{{0, 0, 739},
                                         {7, 7.71486018890863167741, 10693},
                                         {16, 36.32491615703294485229, 20276},
                                         {29, 72.44392874508245938614, 22623},
                                         {38, 90.14209614273795523332, 15581},
                                         {46, 98.64041229093737683797, 5142},
                                         {50, 99, 1}};
    cudf::tdigest::tdigest_column_view tdv(*result);

    tdigest_sample_compare(tdv, expected);

    // verify min/max
    tdigest_minmax_compare<int>(tdv, *values);
  }

  // delta 10
  {
    int const delta = 10;
    auto result = cudf::type_dispatcher(values->view().type(), tdigest_gen{}, op, *values, delta);
    std::vector<expected_value> expected{{0, 6.66025300902007799664, 71618},
                                         {1, 32.54912826201739051157, 187499},
                                         {2, 62.00734805533262772315, 231762},
                                         {3, 82.96355733333332693746, 187500},
                                         {4, 95.91280368612116546956, 71620},
                                         {5, 99, 1}};
    cudf::tdigest::tdigest_column_view tdv(*result);

    tdigest_sample_compare(tdv, expected);

    // verify min/max
    tdigest_minmax_compare<int>(tdv, *values);
  }
}

template <typename Func>
void simple_large_input_decimal_tdigest_aggregation(Func op)
{
  // these tests are being done explicitly because of the way we have to precompute the correct
  // answers. since the input values generated by the generate_distribution() function below are
  // cast to specific types -before- being sent into the aggregation, I can't (safely) just use the
  // expected values that you get when using doubles all the way through.  so I have to pregenerate
  // the correct answers for each type by hand. so, we'll choose a reasonable subset (double,
  // decimal, int, bool)

  auto values = generate_standardized_percentile_distribution(data_type{type_id::DECIMAL32, -4});
  auto cast_values = cudf::cast(*values, data_type{type_id::FLOAT64});

  // compare against a sample of known/expected values (which themselves were verified against the
  // Arrow implementation)

  // delta 1000
  {
    int const delta = 1000;
    auto result = cudf::type_dispatcher(values->view().type(), tdigest_gen{}, op, *values, delta);
    std::vector<expected_value> expected{{0, 0.00035714285714285709, 7},
                                         {10, 0.16229738562091505782, 153},
                                         {59, 5.12759696969697031932, 858},
                                         {250, 62.54576854838715860296, 2356},
                                         {368, 87.85829446685879418055, 1735},
                                         {409, 94.07680636792450457051, 1272},
                                         {491, 99.94192461538463589932, 130},
                                         {500, 99.99965000000000259206, 2}};
    cudf::tdigest::tdigest_column_view tdv(*result);

    tdigest_sample_compare(tdv, expected);

    // verify min/max
    tdigest_minmax_compare<double>(tdv, *cast_values);
  }

  // delta 100
  {
    int const delta = 100;
    auto result = cudf::type_dispatcher(values->view().type(), tdigest_gen{}, op, *values, delta);
    std::vector<expected_value> expected{{0, 0.07260811907983763525, 739},
                                         {7, 8.19761183016926864298, 10693},
                                         {16, 36.82272891595975750079, 20276},
                                         {29, 72.95419827167043536065, 22623},
                                         {38, 90.61224673640975879607, 15581},
                                         {46, 99.07278498638662256326, 5142},
                                         {50, 99.99970000000000425189, 1}};
    cudf::tdigest::tdigest_column_view tdv(*result);

    tdigest_sample_compare(tdv, expected);

    // verify min/max
    tdigest_minmax_compare<double>(tdv, *cast_values);
  }

  // delta 10
  {
    int const delta = 10;
    auto result = cudf::type_dispatcher(values->view().type(), tdigest_gen{}, op, *values, delta);
    std::vector<expected_value> expected{{0, 7.15503361864335740705, 71618},
                                         {1, 33.04966679715625588187, 187499},
                                         {2, 62.50561666407782013266, 231762},
                                         {3, 83.46211575573336460820, 187500},
                                         {4, 96.42199425300195514410, 71620},
                                         {5, 99.99970000000000425189, 1}};
    cudf::tdigest::tdigest_column_view tdv(*result);

    tdigest_sample_compare(tdv, expected);

    // verify min/max
    tdigest_minmax_compare<double>(tdv, *cast_values);
  }
}

}  // namespace test
}  // namespace cudf