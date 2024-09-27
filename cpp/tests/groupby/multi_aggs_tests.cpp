/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

#include <initializer_list>
#include <vector>

using namespace cudf::test::iterators;

namespace {
template <typename T, typename Elements>
std::unique_ptr<cudf::table> create_fixed_table(cudf::size_type num_columns,
                                                cudf::size_type num_rows,
                                                bool include_validity,
                                                Elements elements)
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });
  std::vector<cudf::test::fixed_width_column_wrapper<T>> src_cols(num_columns);
  for (int idx = 0; idx < num_columns; idx++) {
    if (include_validity) {
      src_cols[idx] =
        cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_rows, valids);
    } else {
      src_cols[idx] = cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_rows);
    }
  }
  std::vector<std::unique_ptr<cudf::column>> columns(num_columns);
  std::transform(src_cols.begin(),
                 src_cols.end(),
                 columns.begin(),
                 [](cudf::test::fixed_width_column_wrapper<T>& in) {
                   auto ret = in.release();
                   // pre-cache the null count
                   [[maybe_unused]] auto const nulls = ret->has_nulls();
                   return ret;
                 });
  return std::make_unique<cudf::table>(std::move(columns));
}

template <typename T>
std::unique_ptr<cudf::table> create_random_fixed_table(cudf::size_type num_columns,
                                                       cudf::size_type num_rows)
{
  auto rand_elements =
    cudf::detail::make_counting_transform_iterator(0, [](T i) { return rand(); });
  return create_fixed_table<T>(num_columns, num_rows, false, rand_elements);
}
}  // namespace

template <typename V>
struct groupby_multi_aggs_test : public cudf::test::BaseFixture {};

template <typename Target, typename Source>
std::vector<Target> convert(std::initializer_list<Source> in)
{
  std::vector<Target> out(std::cbegin(in), std::cend(in));
  return out;
}

using supported_types = cudf::test::Concat<cudf::test::Types<int32_t, int64_t, float, double>>;
TYPED_TEST_SUITE(groupby_multi_aggs_test, supported_types);
using K = int32_t;

TYPED_TEST(groupby_multi_aggs_test, basic)
{
  using V = TypeParam;

  auto constexpr num_cols = 3'000;
  auto constexpr num_rows = 100'000;
  auto keys               = create_random_fixed_table<K>(1, num_rows);

  auto vals = create_random_fixed_table<V>(num_cols, num_rows);

  std::vector<cudf::groupby::aggregation_request> requests;
  for (auto i = 0; i < num_cols; i++) {
    requests.emplace_back();

    requests[i].values = vals->get_column(i).view();
    requests[i].aggregations.push_back(
      std::move(cudf::make_mean_aggregation<cudf::groupby_aggregation>()));
    requests[i].aggregations.push_back(
      std::move(cudf::make_min_aggregation<cudf::groupby_aggregation>()));
    requests[i].aggregations.push_back(
      std::move(cudf::make_max_aggregation<cudf::groupby_aggregation>()));
    requests[i].aggregations.push_back(
      std::move(cudf::make_count_aggregation<cudf::groupby_aggregation>()));
  }

  cudf::groupby::groupby gb_obj{keys->view()};

  auto result = gb_obj.aggregate(requests, cudf::test::get_default_stream());
}
