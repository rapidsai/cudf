/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cudf_test/type_lists.hpp>

#include <cudf/groupby.hpp>

using K = int32_t;  // Key type.

template <typename V>
struct groupby_stream_test : public cudf::test::BaseFixture {
  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  void test_groupby(std::unique_ptr<cudf::groupby_aggregation>&& agg,
                    force_use_sort_impl use_sort        = force_use_sort_impl::NO,
                    cudf::null_policy include_null_keys = cudf::null_policy::INCLUDE,
                    cudf::sorted keys_are_sorted        = cudf::sorted::NO)
  {
    auto requests = [&] {
      auto requests = std::vector<cudf::groupby::aggregation_request>{};
      requests.push_back(cudf::groupby::aggregation_request{});
      requests.front().values = vals;
      if (use_sort == force_use_sort_impl::YES) {
        requests.front().aggregations.push_back(
          cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));
      }
      requests.front().aggregations.push_back(std::move(agg));
      return requests;
    }();

    auto gby =
      cudf::groupby::groupby{cudf::table_view{{keys}}, include_null_keys, keys_are_sorted, {}, {}};
    gby.aggregate(requests, cudf::test::get_default_stream());
    // No need to verify results, for stream test.
  }
};

TYPED_TEST_SUITE(groupby_stream_test, cudf::test::AllTypes);

TYPED_TEST(groupby_stream_test, test_count)
{
  auto const make_count_agg = [&](cudf::null_policy include_nulls = cudf::null_policy::EXCLUDE) {
    return cudf::make_count_aggregation<cudf::groupby_aggregation>(include_nulls);
  };

  this->test_groupby(make_count_agg());
  this->test_groupby(make_count_agg(), force_use_sort_impl::YES);
  this->test_groupby(make_count_agg(cudf::null_policy::INCLUDE));
}
