/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <random>

namespace cudf {
namespace test {
enum class force_use_sort_impl : bool { NO, YES };

inline void test_groups(column_view const& keys,
                        column_view const& expect_grouped_keys,
                        std::vector<size_type> const& expect_group_offsets,
                        column_view const& values                = {},
                        column_view const& expect_grouped_values = {})
{
  groupby::groupby gb(table_view({keys}));
  groupby::groupby::groups gb_groups;

  if (values.size()) {
    gb_groups = gb.get_groups(table_view({values}));
  } else {
    gb_groups = gb.get_groups();
  }
  CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({expect_grouped_keys}), gb_groups.keys->view());

  auto got_offsets = gb_groups.offsets;
  EXPECT_EQ(expect_group_offsets.size(), got_offsets.size());
  for (auto i = 0u; i != expect_group_offsets.size(); ++i) {
    EXPECT_EQ(expect_group_offsets[i], got_offsets[i]);
  }

  if (values.size()) {
    CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({expect_grouped_values}), gb_groups.values->view());
  }
}

inline void test_single_agg(column_view const& keys,
                            column_view const& values,
                            column_view const& expect_keys,
                            column_view const& expect_vals,
                            std::unique_ptr<groupby_aggregation>&& agg,
                            force_use_sort_impl use_sort           = force_use_sort_impl::NO,
                            null_policy include_null_keys          = null_policy::EXCLUDE,
                            sorted keys_are_sorted                 = sorted::NO,
                            std::vector<order> const& column_order = {},
                            std::vector<null_order> const& null_precedence = {})
{
  std::vector<groupby::aggregation_request> requests;
  requests.emplace_back(groupby::aggregation_request());
  requests[0].values = values;

  requests[0].aggregations.push_back(std::move(agg));

  if (use_sort == force_use_sort_impl::YES) {
    // WAR to force groupby to use sort implementation
    requests[0].aggregations.push_back(make_nth_element_aggregation<groupby_aggregation>(1));
  }

  groupby::groupby gb_obj(
    table_view({keys}), include_null_keys, keys_are_sorted, column_order, null_precedence);

  auto result = gb_obj.aggregate(requests);

  if (use_sort == force_use_sort_impl::YES) {
    CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({expect_keys}), result.first->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      expect_vals, *result.second[0].results[0], debug_output_level::ALL_ERRORS);
  } else {
    auto const sort_order  = sorted_order(result.first->view(), {}, {null_order::AFTER});
    auto const sorted_keys = gather(result.first->view(), *sort_order);
    auto const sorted_vals = gather(table_view({result.second[0].results[0]->view()}), *sort_order);

    CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({expect_keys}), *sorted_keys);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      expect_vals, sorted_vals->get_column(0), debug_output_level::ALL_ERRORS);
  }
}

inline void test_single_scan(column_view const& keys,
                             column_view const& values,
                             column_view const& expect_keys,
                             column_view const& expect_vals,
                             std::unique_ptr<groupby_scan_aggregation>&& agg,
                             null_policy include_null_keys                  = null_policy::EXCLUDE,
                             sorted keys_are_sorted                         = sorted::NO,
                             std::vector<order> const& column_order         = {},
                             std::vector<null_order> const& null_precedence = {})
{
  std::vector<groupby::scan_request> requests;
  requests.emplace_back(groupby::scan_request());
  requests[0].values = values;

  requests[0].aggregations.push_back(std::move(agg));

  groupby::groupby gb_obj(
    table_view({keys}), include_null_keys, keys_are_sorted, column_order, null_precedence);

  // groupby scan uses sort implementation
  auto result = gb_obj.scan(requests);

  CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({expect_keys}), result.first->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    expect_vals, *result.second[0].results[0], debug_output_level::ALL_ERRORS);
}

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

}  // namespace test
}  // namespace cudf
