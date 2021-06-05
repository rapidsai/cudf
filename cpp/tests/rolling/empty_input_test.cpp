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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar.hpp>

namespace {
// Helper functions to construct rolling window operators.
auto count_valid()
{
  return cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::EXCLUDE);
}

auto count_all()
{
  return cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE);
}

auto sum() { return cudf::make_sum_aggregation<cudf::rolling_aggregation>(); }

auto mean() { return cudf::make_mean_aggregation<cudf::rolling_aggregation>(); }

auto min() { return cudf::make_min_aggregation<cudf::rolling_aggregation>(); }

auto max() { return cudf::make_max_aggregation<cudf::rolling_aggregation>(); }

auto lead() { return cudf::make_lead_aggregation<cudf::rolling_aggregation>(3); }

auto lag() { return cudf::make_lag_aggregation<cudf::rolling_aggregation>(3); }

auto row_number() { return cudf::make_row_number_aggregation<cudf::rolling_aggregation>(); }

auto collect_list() { return cudf::make_collect_list_aggregation<cudf::rolling_aggregation>(); }

auto udf()
{
  return cudf::make_udf_aggregation<cudf::rolling_aggregation>(
    cudf::udf_type::CUDA, "", cudf::data_type{cudf::type_id::INT32});
}

// Constants for rolling_window.
auto const min_periods      = 1;
auto const preceding        = 2;
auto const following        = 2;
auto const preceding_scalar = cudf::numeric_scalar<cudf::size_type>(preceding);
auto const following_scalar = cudf::numeric_scalar<cudf::size_type>(following);
auto const preceding_column = cudf::test::fixed_width_column_wrapper<cudf::size_type>{}.release();
auto const following_column = cudf::test::fixed_width_column_wrapper<cudf::size_type>{}.release();
auto const preceding_col    = preceding_column -> view();
auto const following_col    = following_column -> view();
}  // namespace

struct RollingEmptyInputTest : cudf::test::BaseFixture {
};

template <typename T>
struct TypedRollingEmptyInputTest : RollingEmptyInputTest {
};

TYPED_TEST_CASE(TypedRollingEmptyInputTest, cudf::test::FixedWidthTypes);

using cudf::rolling_aggregation;
using agg_vector_t = std::vector<std::unique_ptr<rolling_aggregation>>;

void rolling_output_type_matches(cudf::column_view const& result,
                                 cudf::type_id expected_type,
                                 cudf::type_id expected_child_type)
{
  using namespace cudf;
  using namespace cudf::test;

  EXPECT_EQ(result.type().id(), expected_type);
  EXPECT_EQ(result.size(), 0);
  if (expected_type == cudf::type_id::LIST) {
    EXPECT_EQ(result.child(cudf::lists_column_view::child_column_index).type().id(),
              expected_child_type);
  }
  if (expected_type == cudf::type_id::STRUCT) {
    EXPECT_EQ(result.child(0).type().id(), expected_child_type);
  }
}

void rolling_output_type_matches(cudf::column_view const& empty_input,
                                 agg_vector_t const& aggs,
                                 cudf::type_id expected_type,
                                 cudf::type_id expected_child_type = cudf::type_id::EMPTY)
{
  using namespace cudf;
  using namespace cudf::test;

  for (auto const& agg : aggs) {
    auto rolling_output_numeric_bounds =
      rolling_window(empty_input, preceding, following, min_periods, *agg);
    rolling_output_type_matches(
      rolling_output_numeric_bounds->view(), expected_type, expected_child_type);

    auto rolling_output_columnar_bounds =
      rolling_window(empty_input, preceding_col, following_col, min_periods, *agg);
    rolling_output_type_matches(
      rolling_output_columnar_bounds->view(), expected_type, expected_child_type);

    auto grouped_rolling_output = grouped_rolling_window(
      table_view{std::vector{empty_input}}, empty_input, preceding, following, min_periods, *agg);
    rolling_output_type_matches(grouped_rolling_output->view(), expected_type, expected_child_type);

    auto grouped_range_rolling_output =
      grouped_range_rolling_window(table_view{std::vector{empty_input}},
                                   empty_input,
                                   order::ASCENDING,
                                   empty_input,
                                   range_window_bounds::get(preceding_scalar),
                                   range_window_bounds::get(following_scalar),
                                   min_periods,
                                   *agg);
    rolling_output_type_matches(
      grouped_range_rolling_output->view(), expected_type, expected_child_type);
  }
}

void rolling_window_throws(cudf::column_view const& empty_input, agg_vector_t const& aggs)
{
  for (auto const& agg : aggs) {
    EXPECT_THROW(rolling_window(empty_input, 2, 2, 1, *agg), cudf::logic_error);
  }
}

TYPED_TEST(TypedRollingEmptyInputTest, EmptyFixedWidthInputs)
{
  using InputType = TypeParam;
  using namespace cudf;
  using namespace cudf::test;

  auto input_col   = fixed_width_column_wrapper<InputType>{}.release();
  auto empty_input = input_col->view();

  /// Test aggregations that yield columns of type `size_type`.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(count_valid());
    aggs.emplace_back(count_all());
    aggs.emplace_back(row_number());

    rolling_output_type_matches(empty_input, aggs, type_to_id<size_type>());
  }

  /// Test aggregations that yield columns of same type as input.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(min());
    aggs.emplace_back(max());
    aggs.emplace_back(lead());
    aggs.emplace_back(lag());
    aggs.emplace_back(udf());

    rolling_output_type_matches(empty_input, aggs, type_to_id<InputType>());
  }

  /// `SUM` returns 64-bit promoted types for integral/decimal input.
  /// For other fixed-width input types, the same type is returned.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(sum());

    using expected_type = cudf::detail::target_type_t<InputType, aggregation::SUM>;
    rolling_output_type_matches(empty_input, aggs, type_to_id<expected_type>());
  }

  /// `MEAN` returns float64 for all numeric types,
  /// except for chrono-types, which yield the same chrono-type.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(mean());

    using expected_type = cudf::detail::target_type_t<InputType, aggregation::MEAN>;
    rolling_output_type_matches(empty_input, aggs, type_to_id<expected_type>());
  }

  /// For an input type `T`, `COLLECT_LIST` returns a column of type `list<T>`.
  {
    auto aggs = std::vector<std::unique_ptr<rolling_aggregation>>{};
    aggs.emplace_back(collect_list());

    rolling_output_type_matches(
      empty_input, aggs, type_to_id<list_view>(), type_to_id<InputType>());
  }
}

TEST_F(RollingEmptyInputTest, Strings)
{
  using namespace cudf;
  using namespace cudf::test;

  auto input_col   = strings_column_wrapper{}.release();
  auto empty_input = input_col->view();

  /// Test aggregations that yield columns of type `size_type`.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(count_valid());
    aggs.emplace_back(count_all());
    aggs.emplace_back(row_number());

    rolling_output_type_matches(empty_input, aggs, type_to_id<size_type>());
  }

  /// Test aggregations that yield columns of same type as input.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(min());
    aggs.emplace_back(max());
    aggs.emplace_back(lead());
    aggs.emplace_back(lag());
    aggs.emplace_back(udf());

    rolling_output_type_matches(empty_input, aggs, type_id::STRING);
  }

  /// For an input type `T`, `COLLECT_LIST` returns a column of type `list<T>`.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(collect_list());

    rolling_output_type_matches(empty_input, aggs, type_to_id<list_view>(), type_id::STRING);
  }

  /// All other aggregations are unsupported.
  {
    auto unsupported_aggs = agg_vector_t{};
    unsupported_aggs.emplace_back(sum());
    unsupported_aggs.emplace_back(mean());

    rolling_window_throws(empty_input, unsupported_aggs);
  }
}

TEST_F(RollingEmptyInputTest, Dictionaries)
{
  using namespace cudf;
  using namespace cudf::test;

  auto input_col   = dictionary_column_wrapper<std::string>{}.release();
  auto empty_input = input_col->view();

  /// Test aggregations that yield columns of type `size_type`.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(count_valid());
    aggs.emplace_back(count_all());
    aggs.emplace_back(row_number());

    rolling_output_type_matches(empty_input, aggs, type_to_id<size_type>());
  }

  /// Test aggregations that yield columns of same type as input.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(min());
    aggs.emplace_back(max());
    aggs.emplace_back(lead());
    aggs.emplace_back(lag());
    aggs.emplace_back(udf());

    rolling_output_type_matches(empty_input, aggs, type_id::DICTIONARY32);
  }

  /// For an input type `T`, `COLLECT_LIST` returns a column of type `list<T>`.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(collect_list());

    rolling_output_type_matches(empty_input, aggs, type_to_id<list_view>(), type_id::DICTIONARY32);
  }

  /// All other aggregations are unsupported.
  {
    auto unsupported_aggs = agg_vector_t{};
    unsupported_aggs.emplace_back(sum());
    unsupported_aggs.emplace_back(mean());

    rolling_window_throws(empty_input, unsupported_aggs);
  }
}

TYPED_TEST(TypedRollingEmptyInputTest, Lists)
{
  using T = TypeParam;
  using namespace cudf;
  using namespace cudf::test;

  auto input_col   = lists_column_wrapper<T>{}.release();
  auto empty_input = input_col->view();

  /// Test aggregations that yield columns of type `size_type`.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(count_valid());
    aggs.emplace_back(count_all());
    aggs.emplace_back(row_number());

    rolling_output_type_matches(empty_input, aggs, type_to_id<size_type>());
  }

  /// Test aggregations that yield columns of same type as input.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(min());
    aggs.emplace_back(max());
    aggs.emplace_back(lead());
    aggs.emplace_back(lag());
    aggs.emplace_back(udf());

    rolling_output_type_matches(empty_input, aggs, type_id::LIST, type_to_id<T>());
  }

  /// For an input type `T`, `COLLECT_LIST` returns a column of type `list<T>`.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(collect_list());

    rolling_output_type_matches(empty_input, aggs, type_id::LIST, type_id::LIST);
  }

  /// All other aggregations are unsupported.
  {
    auto unsupported_aggs = agg_vector_t{};
    unsupported_aggs.emplace_back(sum());
    unsupported_aggs.emplace_back(mean());

    rolling_window_throws(empty_input, unsupported_aggs);
  }
}

TYPED_TEST(TypedRollingEmptyInputTest, Structs)
{
  using T = TypeParam;
  using namespace cudf;
  using namespace cudf::test;

  auto member_col  = fixed_width_column_wrapper<T>{};
  auto input_col   = structs_column_wrapper{{member_col}}.release();
  auto empty_input = input_col->view();

  /// Test aggregations that yield columns of type `size_type`.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(count_valid());
    aggs.emplace_back(count_all());
    aggs.emplace_back(row_number());

    rolling_output_type_matches(empty_input, aggs, type_to_id<size_type>());
  }

  /// Test aggregations that yield columns of same type as input.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(min());
    aggs.emplace_back(max());
    aggs.emplace_back(lead());
    aggs.emplace_back(lag());
    aggs.emplace_back(udf());

    rolling_output_type_matches(empty_input, aggs, type_id::STRUCT, type_to_id<T>());
  }

  /// For an input type `T`, `COLLECT_LIST` returns a column of type `list<T>`.
  {
    auto aggs = agg_vector_t{};
    aggs.emplace_back(collect_list());

    rolling_output_type_matches(empty_input, aggs, type_id::LIST, type_id::STRUCT);
  }

  /// All other aggregations are unsupported.
  {
    auto unsupported_aggs = agg_vector_t{};
    unsupported_aggs.emplace_back(sum());
    unsupported_aggs.emplace_back(mean());

    rolling_window_throws(empty_input, unsupported_aggs);
  }
}
