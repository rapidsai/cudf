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
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>

using namespace cudf::test::iterators;

namespace {
constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr int32_t null{0};                                       // Mark for null elements
constexpr double NaN{std::numeric_limits<double>::quiet_NaN()};  // Mark for NaN double elements

template <class T>
using keys_col = cudf::test::fixed_width_column_wrapper<T, int32_t>;

template <class T>
using vals_col = cudf::test::fixed_width_column_wrapper<T>;

template <class T>
using M2s_col = cudf::test::fixed_width_column_wrapper<T>;

auto compute_HLL(cudf::column_view const& keys, cudf::column_view const& values)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = values;
  requests[0].aggregations.emplace_back(
    cudf::make_hyper_log_log_aggregation<cudf::groupby_aggregation>(9));
  auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys}));
  auto result = gb_obj.aggregate(requests);
  return std::pair(std::move(result.first->release()[0]), std::move(result.second[0].results[0]));
}
}  // namespace

template <class T>
struct GroupbyHLLTypedTest : public cudf::test::BaseFixture {};

using TestTypes = cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t>,
                                     cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(GroupbyHLLTypedTest, TestTypes);

TYPED_TEST(GroupbyHLLTypedTest, SimpleInput)
{
  using T = TypeParam;

  // key = 1: vals = [0, 3, 6]
  // key = 2: vals = [1, 4, 5, 9]
  // key = 3: vals = [2, 7, 8]
  auto const keys = keys_col<T>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals = vals_col<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  compute_HLL(keys, vals);
}
