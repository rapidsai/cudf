/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include <cudf/reduction.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

// for use with groupby and reduction aggregation tests.

namespace CUDF_EXPORT cudf {
namespace test {

using expected_value = thrust::tuple<size_type, double, double>;

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
                            cudf::column_view const& input_values)
{
  using ScalarType = cudf::scalar_type_t<T>;

  auto [col_min, col_max] = cudf::minmax(input_values);

  auto min_scalar   = static_cast<ScalarType*>(col_min.get());
  auto max_scalar   = static_cast<ScalarType*>(col_max.get());
  auto expected_min = static_cast<double>(min_scalar->value());
  auto expected_max = static_cast<double>(max_scalar->value());

  double tdv_min, tdv_max;
  EXPECT_EQ(cudaMemcpy(&tdv_min, tdv.min_begin(), sizeof(double), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(&tdv_max, tdv.max_begin(), sizeof(double), cudaMemcpyDeviceToHost),
            cudaSuccess);

  EXPECT_EQ(tdv_min, expected_min);
  EXPECT_EQ(tdv_max, expected_max);
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
  bool is_cpu_cluster_computation_disabled[2] = {true, false};
  for (int idx = 0; idx < 2; idx++) {
    cudf::tdigest::detail::is_cpu_cluster_computation_disabled =
      is_cpu_cluster_computation_disabled[idx];

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
}

// shared test for groupby/reduction.
template <typename T, typename Func>
void tdigest_simple_with_nulls_aggregation(Func op)
{
  bool is_cpu_cluster_computation_disabled[2] = {true, false};
  for (int idx = 0; idx < 2; idx++) {
    cudf::tdigest::detail::is_cpu_cluster_computation_disabled =
      is_cpu_cluster_computation_disabled[idx];

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
}

// shared test for groupby/reduction.
template <typename T, typename Func>
void tdigest_simple_all_nulls_aggregation(Func op)
{
  bool is_cpu_cluster_computation_disabled[2] = {true, false};
  for (int idx = 0; idx < 2; idx++) {
    cudf::tdigest::detail::is_cpu_cluster_computation_disabled =
      is_cpu_cluster_computation_disabled[idx];

    // create a tdigest that has far fewer values in it than the delta value. this should result
    // in every value remaining uncompressed
    cudf::test::fixed_width_column_wrapper<T> values{{122, 15, 1, 99, 67, 101, 100, 84, 44, 2},
                                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    int const delta = 1000;
    auto result     = cudf::type_dispatcher(
      static_cast<column_view>(values).type(), tdigest_gen{}, op, values, delta);

    // NOTE: an empty tdigest column still has 1 row.
    auto expected = cudf::tdigest::detail::make_empty_tdigests_column(
      1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
  }
}

// Note: there is no need to test different types here as the internals of a tdigest are always
// the same regardless of input.
template <typename Func, typename MergeFunc>
void tdigest_merge_simple(Func op, MergeFunc merge_op)
{
  bool is_cpu_cluster_computation_disabled[2] = {true, false};
  for (int idx = 0; idx < 2; idx++) {
    cudf::tdigest::detail::is_cpu_cluster_computation_disabled =
      is_cpu_cluster_computation_disabled[idx];

    auto values = generate_standardized_percentile_distribution(data_type{type_id::FLOAT64});
    CUDF_EXPECTS(values->size() == 750000, "Unexpected distribution size");

    auto split_values = cudf::split(*values, {250000, 500000});

    int const delta = 1000;

    // generate separate digests
    std::vector<std::unique_ptr<column>> parts;
    auto iter = thrust::make_counting_iterator(0);
    std::transform(iter,
                   iter + split_values.size(),
                   std::back_inserter(parts),
                   [&split_values, delta, op](int i) { return op(split_values[i], delta); });
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
}

// shared test for groupby/reduction.
template <typename MergeFunc>
void tdigest_merge_empty(MergeFunc merge_op)
{
  bool is_cpu_cluster_computation_disabled[2] = {true, false};
  for (int idx = 0; idx < 2; idx++) {
    cudf::tdigest::detail::is_cpu_cluster_computation_disabled =
      is_cpu_cluster_computation_disabled[idx];

    // 3 empty tdigests all in the same group
    auto a = cudf::tdigest::detail::make_empty_tdigests_column(
      1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    auto b = cudf::tdigest::detail::make_empty_tdigests_column(
      1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    auto c = cudf::tdigest::detail::make_empty_tdigests_column(
      1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    std::vector<column_view> cols;
    cols.push_back(*a);
    cols.push_back(*b);
    cols.push_back(*c);
    auto values = cudf::concatenate(cols);

    auto const delta = 1000;
    auto result      = merge_op(*values, delta);

    auto expected = cudf::tdigest::detail::make_empty_tdigests_column(
      1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *result);
  }
}

}  // namespace test
}  // namespace CUDF_EXPORT cudf
