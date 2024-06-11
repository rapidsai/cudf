/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf_test/column_wrapper.hpp>

#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/groupby.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

// for use with groupby and reduction aggregation tests.

namespace CUDF_EXPORT cudf {
namespace test {

using expected_value = thrust::tuple<size_type, double, double>;

/**
 * @brief Device functor to compute min of a sequence of values serially.
 */
template <typename T>
struct column_min {
  /**
   * @brief Computes the min of a sequence of values serially.
   *
   * @param vals The sequence of values to compute the min of
   * @return The min value
   */
  __device__ double operator()(device_span<T const> vals)
  {
    return static_cast<double>(*thrust::min_element(thrust::seq, vals.begin(), vals.end()));
  }
};

/**
 * @brief Device functor to compute max of a sequence of values serially.
 */
template <typename T>
struct column_max {
  /**
   * @brief Computes the max of a sequence of values serially.
   *
   * @param vals The sequence of values to compute the max of
   * @return The max value
   */
  __device__ double operator()(device_span<T const> vals)
  {
    return static_cast<double>(*thrust::max_element(thrust::seq, vals.begin(), vals.end()));
  }
};

/**
 * @brief Functor to generate a tdigest.
 */
struct tdigest_gen {
  // @cond
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
  // @endcond
};

template <typename T>
inline T frand()
{
  return static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
}

template <typename T>
inline T rand_range(T min, T max)
{
  return min + static_cast<T>(frand<T>() * (max - min));
}

inline std::unique_ptr<column> generate_typed_percentile_distribution(
  std::vector<double> const& buckets,
  std::vector<int> const& sizes,
  data_type t,
  bool sorted = false)
{
  srand(0);

  std::vector<double> values;
  size_t total_size = std::reduce(sizes.begin(), sizes.end(), 0);
  values.reserve(total_size);
  for (size_t idx = 0; idx < sizes.size(); idx++) {
    double min = idx == 0 ? 0.0f : buckets[idx - 1];
    double max = buckets[idx];

    for (int v_idx = 0; v_idx < sizes[idx]; v_idx++) {
      values.push_back(rand_range(min, max));
    }
  }

  if (sorted) { std::sort(values.begin(), values.end()); }

  cudf::test::fixed_width_column_wrapper<double> src(values.begin(), values.end());
  return cudf::cast(src, t);
}

// "standardized" means the parameters sent into generate_typed_percentile_distribution. the intent
// is to provide a standardized set of inputs for use with tdigest generation tests and
// percentile_approx tests. std::vector<double>
// buckets{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}; std::vector<int>
// sizes{50000, 50000, 50000, 50000, 50000, 100000, 100000, 100000, 100000, 100000};
inline std::unique_ptr<column> generate_standardized_percentile_distribution(
  data_type t = data_type{type_id::FLOAT64}, bool sorted = false)
{
  std::vector<double> buckets{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0, 90.0f, 100.0f};
  std::vector<int> b_sizes{
    50000, 50000, 50000, 50000, 50000, 100000, 100000, 100000, 100000, 100000};
  return generate_typed_percentile_distribution(buckets, b_sizes, t, sorted);
}

/**
 * @brief Compare a tdigest column against a sampling of expected values.
 */
void tdigest_sample_compare(cudf::tdigest::tdigest_column_view const& tdv,
                            std::vector<expected_value> const& h_expected);

/**
 * @brief Compare the min/max values of a tdigest against inputs.
 */
template <typename T>
void tdigest_minmax_compare(cudf::tdigest::tdigest_column_view const& tdv,
                            column_view const& input_values)
{
  // verify min/max
  thrust::host_vector<device_span<T const>> h_spans;
  h_spans.push_back({input_values.begin<T>(), static_cast<size_t>(input_values.size())});
  auto spans = cudf::detail::make_device_uvector_async(
    h_spans, cudf::get_default_stream(), rmm::mr::get_current_device_resource());

  auto expected_min = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, spans.size(), mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy(cudf::get_default_stream()),
                    spans.begin(),
                    spans.end(),
                    expected_min->mutable_view().template begin<double>(),
                    column_min<T>{});
  column_view result_min(data_type{type_id::FLOAT64}, tdv.size(), tdv.min_begin(), nullptr, 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_min, *expected_min);

  auto expected_max = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, spans.size(), mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy(cudf::get_default_stream()),
                    spans.begin(),
                    spans.end(),
                    expected_max->mutable_view().template begin<double>(),
                    column_max<T>{});
  column_view result_max(data_type{type_id::FLOAT64}, tdv.size(), tdv.max_begin(), nullptr, 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_max, *expected_max);
}

/// Expected values for tdigest tests
struct expected_tdigest {
  // @cond
  column_view mean;
  column_view weight;
  double min, max;
  // @endcond
};

/**
 * @brief Create an expected tdigest column given component inputs.
 */
std::unique_ptr<column> make_expected_tdigest_column(std::vector<expected_tdigest> const& groups);

// shared test for groupby/reduction.
template <typename T, typename Func>
void tdigest_simple_aggregation(Func op)
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

// shared test for groupby/reduction.
template <typename T, typename Func>
void tdigest_simple_with_nulls_aggregation(Func op)
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

// shared test for groupby/reduction.
template <typename T, typename Func>
void tdigest_simple_all_nulls_aggregation(Func op)
{
  // create a tdigest that has far fewer values in it than the delta value. this should result
  // in every value remaining uncompressed
  cudf::test::fixed_width_column_wrapper<T> values{{122, 15, 1, 99, 67, 101, 100, 84, 44, 2},
                                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  int const delta = 1000;
  auto result     = cudf::type_dispatcher(
    static_cast<column_view>(values).type(), tdigest_gen{}, op, values, delta);

  // NOTE: an empty tdigest column still has 1 row.
  auto expected = cudf::tdigest::detail::make_empty_tdigest_column(
    cudf::get_default_stream(), rmm::mr::get_current_device_resource());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// shared test for groupby/reduction.
template <typename Func>
void tdigest_simple_large_input_double_aggregation(Func op)
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

// shared test for groupby/reduction.
template <typename Func>
void tdigest_simple_large_input_int_aggregation(Func op)
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

// shared test for groupby/reduction.
template <typename Func>
void tdigest_simple_large_input_decimal_aggregation(Func op)
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

// Note: there is no need to test different types here as the internals of a tdigest are always
// the same regardless of input.
template <typename Func, typename MergeFunc>
void tdigest_merge_simple(Func op, MergeFunc merge_op)
{
  auto values = generate_standardized_percentile_distribution(data_type{type_id::FLOAT64});
  CUDF_EXPECTS(values->size() == 750000, "Unexpected distribution size");

  auto split_values = cudf::split(*values, {250000, 500000});

  int const delta = 1000;

  // generate separate digests
  std::vector<std::unique_ptr<column>> parts;
  auto iter = thrust::make_counting_iterator(0);
  std::transform(
    iter, iter + split_values.size(), std::back_inserter(parts), [&split_values, delta, op](int i) {
      return op(split_values[i], delta);
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
    auto result      = merge_op(*merge_input, merge_delta);
    cudf::tdigest::tdigest_column_view tdv(*result);

    // verify centroids
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
    tdigest_sample_compare(tdv, expected);

    // verify min/max
    tdigest_minmax_compare<double>(tdv, *values);
  }
}

// shared test for groupby/reduction.
template <typename MergeFunc>
void tdigest_merge_empty(MergeFunc merge_op)
{
  // 3 empty tdigests all in the same group
  auto a = cudf::tdigest::detail::make_empty_tdigest_column(cudf::get_default_stream(),
                                                            rmm::mr::get_current_device_resource());
  auto b = cudf::tdigest::detail::make_empty_tdigest_column(cudf::get_default_stream(),
                                                            rmm::mr::get_current_device_resource());
  auto c = cudf::tdigest::detail::make_empty_tdigest_column(cudf::get_default_stream(),
                                                            rmm::mr::get_current_device_resource());
  std::vector<column_view> cols;
  cols.push_back(*a);
  cols.push_back(*b);
  cols.push_back(*c);
  auto values = cudf::concatenate(cols);

  auto const delta = 1000;
  auto result      = merge_op(*values, delta);

  auto expected = cudf::tdigest::detail::make_empty_tdigest_column(
    cudf::get_default_stream(), rmm::mr::get_current_device_resource());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *result);
}

}  // namespace test
}  // namespace cudf
