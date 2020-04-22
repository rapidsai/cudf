/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>

#include <cudf/utilities/bit.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/table/table_view.hpp>
#include <src/rolling/rolling_detail.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <vector>
#include <algorithm>

using cudf::test::fixed_width_column_wrapper;
using cudf::size_type;
using cudf::bitmask_type;

template <typename T>
class GroupedRollingTest : public cudf::test::BaseFixture {
protected:
  // input as column_wrapper
  void run_test_col(cudf::table_view const& keys,
                    cudf::column_view const& input,
                    std::vector<size_type> const& expected_grouping,
                    size_type const& preceding_window,
                    size_type const& following_window,
                    size_type min_periods,
                    std::unique_ptr<cudf::experimental::aggregation> const& op)
  {
    std::unique_ptr<cudf::column> output;

    // wrap windows
    EXPECT_NO_THROW(output = cudf::experimental::grouped_rolling_window(keys, input, preceding_window,
                                                                following_window, min_periods, op));

    auto reference = create_reference_output(op, input, expected_grouping, 
      preceding_window, following_window, min_periods);

#ifndef NDEBUG
    std::cout << "input:\n";
    cudf::test::print(input, std::cout, ", ");
    std::cout << "\n";
    std::cout << "output:\n";
    cudf::test::print(*output, std::cout, ", ");
    std::cout << "\n";
    std::cout << "reference:\n";
    cudf::test::print(*reference, std::cout, ", ");
    std::cout << "\n";
    std::cout << "\n";
#endif

    cudf::test::expect_columns_equal(*output, *reference);
  }

  void run_test_col_agg(cudf::table_view const& keys,
                        cudf::column_view const& input,
                        std::vector<size_type> const& expected_grouping,
                        size_type preceding_window,
                        size_type following_window,
                        size_type min_periods)
  {
    // Skip grouping-tests on bool8 keys. sort_helper does not support this.
    if (cudf::is_boolean(keys.column(0).type())) {
      return;
    }

    // test all supported aggregators
    run_test_col(keys, input, expected_grouping, preceding_window, following_window, min_periods, cudf::experimental::make_min_aggregation());
    run_test_col(keys, input, expected_grouping, preceding_window, following_window, min_periods, cudf::experimental::make_count_aggregation());
    run_test_col(keys, input, expected_grouping, preceding_window, following_window, min_periods, cudf::experimental::make_count_aggregation(cudf::include_nulls::YES));
    run_test_col(keys, input, expected_grouping, preceding_window, following_window, min_periods, cudf::experimental::make_max_aggregation());
    run_test_col(keys, input, expected_grouping, preceding_window, following_window, min_periods, cudf::experimental::make_mean_aggregation());

    if (!cudf::is_timestamp(input.type())) {
      run_test_col(keys, input, expected_grouping, preceding_window, following_window, min_periods, cudf::experimental::make_sum_aggregation());
    }
  }

  private:

  // use SFINAE to only instantiate for supported combinations

  // specialization for COUNT_VALID, COUNT_ALL
  template <bool include_nulls>
  std::unique_ptr<cudf::column> 
  create_count_reference_output(cudf::column_view const& input,
                                std::vector<size_type> const& group_offsets,
                                size_type const& preceding_window,
                                size_type const& following_window,
                                size_type min_periods)
  {
    size_type num_rows = input.size();
    thrust::host_vector<cudf::size_type> ref_data(num_rows);
    thrust::host_vector<bool>            ref_valid(num_rows);

    // input data and mask
  
    std::vector<bitmask_type> in_valid = cudf::test::bitmask_to_host(input);
    bitmask_type* valid_mask = in_valid.data();

    for(size_type i = 0; i < num_rows; i++) {
      // load sizes
      min_periods = std::max(min_periods, 1); // at least one observation is required

      // compute bounds
      auto group_end_index = std::upper_bound(group_offsets.begin(), group_offsets.end(), i);
      auto group_start_index = group_end_index - 1;

      size_type start = std::min(num_rows, std::max(0, i - preceding_window +1));
      size_type end = std::min(num_rows, std::max(0, i + following_window + 1));
      size_type start_index = std::max(*group_start_index, std::min(start, end));
      size_type end_index = std::min(*group_end_index, std::max(start, end));

      // aggregate
      size_type count = 0;
      for (size_type j = start_index; j < end_index; j++) {
        if (include_nulls || !input.nullable() || cudf::bit_is_set(valid_mask, j))
          count++;
      }

      ref_valid[i] = (count >= min_periods);
      if (ref_valid[i])
        ref_data[i] = count;
    }

    fixed_width_column_wrapper<cudf::size_type> col(ref_data.begin(), ref_data.end(),
                                                    ref_valid.begin());
    return col.release();
  }

  template<typename agg_op, cudf::experimental::aggregation::Kind k, typename OutputType, bool is_mean,
           std::enable_if_t<cudf::detail::is_supported<T, agg_op, k, is_mean>()>* = nullptr>
  std::unique_ptr<cudf::column>
  create_reference_output(cudf::column_view const& input,
                          std::vector<size_type> const& group_offsets,
                          size_type const& preceding_window,
                          size_type const& following_window,
                          size_type min_periods)
  {
    size_type num_rows = input.size();
    thrust::host_vector<OutputType> ref_data(num_rows);
    thrust::host_vector<bool> ref_valid(num_rows);

    // input data and mask
    thrust::host_vector<T> in_col;
    std::vector<bitmask_type> in_valid; 
    std::tie(in_col, in_valid) = cudf::test::to_host<T>(input); 
    bitmask_type* valid_mask = in_valid.data();
    
    agg_op op;
    for(size_type i = 0; i < num_rows; i++) {
      OutputType val = agg_op::template identity<OutputType>();

      // load sizes
      min_periods = std::max(min_periods, 1); // at least one observation is required

      // compute bounds
      auto group_end_index = std::upper_bound(group_offsets.begin(), group_offsets.end(), i);
      auto group_start_index = group_end_index - 1;

      size_type start = std::min(num_rows, std::max(0, i - preceding_window +1)); // Preceding window includes current row.
      size_type end = std::min(num_rows, std::max(0, i + following_window + 1));
      size_type start_index = std::max(*group_start_index, std::min(start, end));
      size_type end_index = std::min(*group_end_index, std::max(start, end));
      
      // aggregate
      size_type count = 0;
      for (size_type j = start_index; j < end_index; j++) {
        if (!input.nullable() || cudf::bit_is_set(valid_mask, j)) {
          val = op(static_cast<OutputType>(in_col[j]), val);
          count++;
        }
      }

      ref_valid[i] = (count >= min_periods);
      if (ref_valid[i]) {
        cudf::detail::store_output_functor<OutputType, is_mean>{}(ref_data[i], val, count);
      }
    }

    fixed_width_column_wrapper<OutputType> col(ref_data.begin(), ref_data.end(), ref_valid.begin());
    return col.release();
  }

  template<typename agg_op, cudf::experimental::aggregation::Kind k, typename OutputType, bool is_mean,
           std::enable_if_t<!cudf::detail::is_supported<T, agg_op, k, is_mean>()>* = nullptr>
  std::unique_ptr<cudf::column> create_reference_output(cudf::column_view const& input, 
                                                        std::vector<size_type> const& group_offsets,
                                                        size_type const& preceding_window_col,
                                                        size_type const& following_window_col,
                                                        size_type min_periods)
  {
    CUDF_FAIL("Unsupported combination of type and aggregation");
  }

  std::unique_ptr<cudf::column> create_reference_output(std::unique_ptr<cudf::experimental::aggregation>const& op,
                                                        cudf::column_view const& input,
                                                        std::vector<size_type> const& group_offsets,
                                                        size_type const& preceding_window,
                                                        size_type const& following_window,
                                                        size_type min_periods)
  {
    // unroll aggregation types
    switch(op->kind) {
    case cudf::experimental::aggregation::SUM:
      return create_reference_output<cudf::DeviceSum, cudf::experimental::aggregation::SUM, 
             cudf::experimental::detail::target_type_t<T, cudf::experimental::aggregation::SUM>, false>(input, group_offsets, preceding_window,
                                                             following_window, min_periods);
    case cudf::experimental::aggregation::MIN:
      return create_reference_output<cudf::DeviceMin, cudf::experimental::aggregation::MIN,
             cudf::experimental::detail::target_type_t<T, cudf::experimental::aggregation::MIN>, false>(input, group_offsets, preceding_window,
                                                             following_window, min_periods);
    case cudf::experimental::aggregation::MAX:
      return create_reference_output<cudf::DeviceMax, cudf::experimental::aggregation::MAX,
             cudf::experimental::detail::target_type_t<T, cudf::experimental::aggregation::MAX>, false>(input, group_offsets, preceding_window,
                                                             following_window, min_periods);
    case cudf::experimental::aggregation::COUNT_VALID:
      return create_count_reference_output<false>(input, group_offsets, preceding_window, following_window, min_periods);
    case cudf::experimental::aggregation::COUNT_ALL:
      return create_count_reference_output<true>(input, group_offsets, preceding_window, following_window, min_periods);
    case cudf::experimental::aggregation::MEAN:
      return create_reference_output<cudf::DeviceSum, cudf::experimental::aggregation::MEAN,
             cudf::experimental::detail::target_type_t<T, cudf::experimental::aggregation::MEAN>, true>(input, group_offsets, preceding_window,
                                                            following_window, min_periods);
    default:
      return fixed_width_column_wrapper<T>({}).release();
    }
  }
};


// // ------------- expected failures --------------------

class GroupedRollingErrorTest : public cudf::test::BaseFixture {};

// negative sizes
TEST_F(GroupedRollingErrorTest, NegativeMinPeriods)
{
  // Construct agg column.
  const std::vector<size_type> col_data  {0, 1, 2, 0, 4};
  const std::vector<bool>      col_valid {1, 1, 1, 0, 1};
  fixed_width_column_wrapper<size_type> input{col_data.begin(), col_data.end(), col_valid.begin()};

  // Construct Grouping keys table-view.
  const auto N_ELEMENTS {col_data.size()};
  const std::vector<size_type> grouping_key_vec(N_ELEMENTS, 0);
  fixed_width_column_wrapper<size_type> grouping_keys_col(grouping_key_vec.begin(), grouping_key_vec.end(), col_valid.begin());
  const cudf::table_view grouping_keys {std::vector<cudf::column_view>{grouping_keys_col}}; 

  EXPECT_THROW(cudf::experimental::grouped_rolling_window(grouping_keys, input, 2,  2, -2, cudf::experimental::make_sum_aggregation()),
               cudf::logic_error);
}

TEST_F(GroupedRollingErrorTest, EmptyInput) {
  cudf::test::fixed_width_column_wrapper<int32_t> empty_col{};
  std::unique_ptr<cudf::column> output;
  const cudf::table_view grouping_keys{std::vector<cudf::column_view>{}};
  EXPECT_NO_THROW(output = cudf::experimental::grouped_rolling_window(grouping_keys, empty_col,  2, 0, 2,
                                                              cudf::experimental::make_sum_aggregation()));
  EXPECT_EQ(output->size(), 0);
}

// incorrect type/aggregation combo: sum of timestamps
TEST_F(GroupedRollingErrorTest, SumTimestampNotSupported)
{
  constexpr size_type size{10};
  fixed_width_column_wrapper<cudf::timestamp_D> input_D(thrust::make_counting_iterator(0),
                                                       thrust::make_counting_iterator(size));
  fixed_width_column_wrapper<cudf::timestamp_s> input_s(thrust::make_counting_iterator(0),
                                                       thrust::make_counting_iterator(size));
  fixed_width_column_wrapper<cudf::timestamp_ms> input_ms(thrust::make_counting_iterator(0),
                                                       thrust::make_counting_iterator(size));
  fixed_width_column_wrapper<cudf::timestamp_us> input_us(thrust::make_counting_iterator(0),
                                                       thrust::make_counting_iterator(size));
  fixed_width_column_wrapper<cudf::timestamp_ns> input_ns(thrust::make_counting_iterator(0),
                                                       thrust::make_counting_iterator(size));

  // Construct table-view of grouping keys.
  std::vector<size_type> grouping_keys_vec(size, 0); // `size` elements, each == 0.
  const cudf::table_view grouping_keys{
    std::vector<cudf::column_view>{
      fixed_width_column_wrapper<size_type>( grouping_keys_vec.begin(), grouping_keys_vec.end() )
    }
  };

  EXPECT_THROW(cudf::experimental::grouped_rolling_window(grouping_keys, input_D, 2, 2, 0, cudf::experimental::make_sum_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::grouped_rolling_window(grouping_keys, input_s, 2, 2, 0, cudf::experimental::make_sum_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::grouped_rolling_window(grouping_keys, input_ms, 2, 2, 0, cudf::experimental::make_sum_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::grouped_rolling_window(grouping_keys, input_us, 2, 2, 0, cudf::experimental::make_sum_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::grouped_rolling_window(grouping_keys, input_ns, 2, 2, 0, cudf::experimental::make_sum_aggregation()),
               cudf::logic_error);
}

TYPED_TEST_CASE(GroupedRollingTest, cudf::test::FixedWidthTypes);

TYPED_TEST(GroupedRollingTest, SimplePartitionedStaticWindowsWithGroupKeys)
{
  const auto col_data = cudf::test::make_type_param_vector<TypeParam>({0, 10, 20, 30, 40, 50, 60, 70, 80, 90});
  const size_type DATA_SIZE {static_cast<size_type>(col_data.size())};
  const std::vector<bool>      col_mask (DATA_SIZE, true);
  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());

  // 2 grouping keys, with effectively 3 groups of at most 4 rows each: 
  //   1. key_0 {0, 0, 0, ...0}
  //   2. key_1 {0, 0, 0, 0, 1, 1, 1, 1, 2, 2}
  std::vector<TypeParam> key_0_vec(DATA_SIZE, 0);
  std::vector<TypeParam> key_1_vec;
  int i{0};
  std::generate_n(std::back_inserter(key_1_vec), DATA_SIZE, [&i](){return i++/4;}); // Groups of 4.
  const fixed_width_column_wrapper<TypeParam> key_0 (key_0_vec.begin(), key_0_vec.end());
  const fixed_width_column_wrapper<TypeParam> key_1 (key_1_vec.begin(), key_1_vec.end());
  const cudf::table_view grouping_keys {std::vector<cudf::column_view>{key_0, key_1}};

  size_type preceding_window = 2;
  size_type following_window = 1;
  std::vector<size_type> expected_group_offsets{0, 4, 8, DATA_SIZE};

  this->run_test_col_agg(grouping_keys, input, expected_group_offsets, preceding_window, following_window, 1);
}

// all rows are invalid
TYPED_TEST(GroupedRollingTest, AllInvalid)
{
  const auto col_data = cudf::test::make_type_param_vector<TypeParam>({0, 10, 20, 30, 40, 50, 60, 70, 80, 90});
  const size_type DATA_SIZE {static_cast<size_type>(col_data.size())};
  const std::vector<bool>      col_mask (DATA_SIZE, false);
  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());

  // 2 grouping keys, with effectively 3 groups of at most 4 rows each: 
  //   1. key_0 {0, 0, 0, ...0}
  //   2. key_1 {0, 0, 0, 0, 1, 1, 1, 1, 2, 2}
  std::vector<TypeParam> key_0_vec(DATA_SIZE, 0);
  std::vector<TypeParam> key_1_vec;
  int i{0};
  std::generate_n(std::back_inserter(key_1_vec), DATA_SIZE, [&i](){return i++/4;}); // Groups of 4.
  const fixed_width_column_wrapper<TypeParam> key_0 (key_0_vec.begin(), key_0_vec.end());
  const fixed_width_column_wrapper<TypeParam> key_1 (key_1_vec.begin(), key_1_vec.end());
  const cudf::table_view grouping_keys {std::vector<cudf::column_view>{key_0, key_1}};

  size_type preceding_window = 2;
  size_type following_window = 1;
  std::vector<size_type> expected_group_offsets{0, 4, 8, DATA_SIZE};

  this->run_test_col_agg(grouping_keys, input, expected_group_offsets, preceding_window, following_window, 1);
}

// window = following_window = 0
TYPED_TEST(GroupedRollingTest, ZeroWindow)
{
  const auto col_data = cudf::test::make_type_param_vector<TypeParam>({0, 10, 20, 30, 40, 50, 60, 70, 80, 90});
  const size_type DATA_SIZE {static_cast<size_type>(col_data.size())};
  const std::vector<bool>      col_mask (DATA_SIZE, true);
  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());

  // 2 grouping keys, with effectively 3 groups of at most 4 rows each: 
  //   1. key_0 {0, 0, 0, ...0}
  //   2. key_1 {0, 0, 0, 0, 1, 1, 1, 1, 2, 2}
  std::vector<TypeParam> key_0_vec(DATA_SIZE, 0);
  std::vector<TypeParam> key_1_vec;
  int i{0};
  std::generate_n(std::back_inserter(key_1_vec), DATA_SIZE, [&i](){return i++/4;}); // Groups of 4.
  const fixed_width_column_wrapper<TypeParam> key_0 (key_0_vec.begin(), key_0_vec.end());
  const fixed_width_column_wrapper<TypeParam> key_1 (key_1_vec.begin(), key_1_vec.end());
  const cudf::table_view grouping_keys {std::vector<cudf::column_view>{key_0, key_1}};

  size_type preceding_window = 0;
  size_type following_window = 0;
  std::vector<size_type> expected_group_offsets{0, 4, 8, DATA_SIZE};

  this->run_test_col_agg(grouping_keys, input, expected_group_offsets, preceding_window, following_window, 1);
}

// ------------- non-fixed-width types --------------------

using GroupedRollingTestStrings = GroupedRollingTest<cudf::string_view>;

TEST_F(GroupedRollingTestStrings, StringsUnsupportedOperators)
{
  cudf::test::strings_column_wrapper input{{"This", "is", "not", "a", "string", "type"},
                                           {1, 1, 1, 0, 1, 0}};

  const size_type DATA_SIZE {static_cast<cudf::column_view>(input).size()};
  const std::vector<size_type> key_col_vec(DATA_SIZE, 0);
  const cudf::table_view key_cols{
    std::vector<cudf::column_view> {
      fixed_width_column_wrapper<size_type>(key_col_vec.begin(), key_col_vec.end())
    }
  };
  
  EXPECT_THROW(cudf::experimental::grouped_rolling_window(key_cols, input, 2, 2, 0, cudf::experimental::make_sum_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::grouped_rolling_window(key_cols, input, 2, 2, 0, cudf::experimental::make_mean_aggregation()),
               cudf::logic_error);
}

template <typename T>
class GroupedTimeRangeRollingTest : public cudf::test::BaseFixture {
protected:
  // input as column_wrapper
  void run_test_col(cudf::table_view const& keys,
                    cudf::column_view const& timestamp_column,
                    cudf::column_view const& input,
                    std::vector<size_type> const& expected_grouping,
                    size_type const& preceding_window_in_days,
                    size_type const& following_window_in_days,
                    size_type min_periods,
                    std::unique_ptr<cudf::experimental::aggregation> const& op)
  {
    std::unique_ptr<cudf::column> output;

    // wrap windows
    EXPECT_NO_THROW(output = cudf::experimental::grouped_time_range_rolling_window(keys, timestamp_column, input, preceding_window_in_days,
                                                                following_window_in_days, min_periods, op));

    auto reference = create_reference_output(op, timestamp_column, input, expected_grouping, 
      preceding_window_in_days, following_window_in_days, min_periods);

#ifndef NDEBUG
    std::cout << "input:\n";
    cudf::test::print(input, std::cout, ", ");
    std::cout << "\n";
    std::cout << "output:\n";
    cudf::test::print(*output, std::cout, ", ");
    std::cout << "\n";
    std::cout << "reference:\n";
    cudf::test::print(*reference, std::cout, ", ");
    std::cout << "\n";
    std::cout << "\n";
#endif

    cudf::test::expect_columns_equal(*output, *reference);
  }

  void run_test_col_agg(cudf::table_view const& keys,
                        cudf::column_view const& timestamp_column,
                        cudf::column_view const& input,
                        std::vector<size_type> const& expected_grouping,
                        size_type preceding_window_in_days,
                        size_type following_window_in_days,
                        size_type min_periods)
  {
    // Skip grouping-tests on bool8 keys. sort_helper does not support this.
    if (cudf::is_boolean(keys.column(0).type())) {
      return;
    }

    // test all supported aggregators
    run_test_col(keys, timestamp_column, input, expected_grouping, preceding_window_in_days, following_window_in_days, min_periods, cudf::experimental::make_min_aggregation());
    run_test_col(keys, timestamp_column, input, expected_grouping, preceding_window_in_days, following_window_in_days, min_periods, cudf::experimental::make_count_aggregation());
    run_test_col(keys, timestamp_column, input, expected_grouping, preceding_window_in_days, following_window_in_days, min_periods, cudf::experimental::make_count_aggregation(cudf::include_nulls::YES));
    run_test_col(keys, timestamp_column, input, expected_grouping, preceding_window_in_days, following_window_in_days, min_periods, cudf::experimental::make_max_aggregation());
    run_test_col(keys, timestamp_column, input, expected_grouping, preceding_window_in_days, following_window_in_days, min_periods, cudf::experimental::make_mean_aggregation());

    if (!cudf::is_timestamp(input.type())) {
      run_test_col(keys, timestamp_column, input, expected_grouping, preceding_window_in_days, following_window_in_days, min_periods, cudf::experimental::make_sum_aggregation());
    }
  }

  private:

  // use SFINAE to only instantiate for supported combinations

  // specialization for COUNT_VALID, COUNT_ALL
  template <bool include_nulls>
  std::unique_ptr<cudf::column> 
  create_count_reference_output(cudf::column_view const& timestamp_column,
                                cudf::column_view const& input,
                                std::vector<size_type> const& group_offsets,
                                size_type const& preceding_window_in_days,
                                size_type const& following_window_in_days,
                                size_type min_periods)
  {
    assert(timestamp_column.type().id() == cudf::TIMESTAMP_DAYS); // Testing with DAYS.

    auto timestamp_vec = cudf::test::to_host<int32_t>(timestamp_column).first;

    size_type num_rows = input.size();
    thrust::host_vector<cudf::size_type> ref_data(num_rows);
    thrust::host_vector<bool> ref_valid(num_rows);

    // input data and mask
  
    std::vector<bitmask_type> in_valid = cudf::test::bitmask_to_host(input);
    bitmask_type* valid_mask = in_valid.data();

    for(size_type i = 0; i < num_rows; i++) {
      // load sizes
      min_periods = std::max(min_periods, 1); // at least one observation is required

      // compute bounds
      auto group_end_index = std::upper_bound(group_offsets.begin(), group_offsets.end(), i);
      auto group_start_index = group_end_index - 1;

      size_type start_index = i;
      while ((start_index-1) >= *group_start_index && timestamp_vec[start_index-1] >= (timestamp_vec[i]-preceding_window_in_days)) {
       --start_index; 
      }

      size_type end_index = i;
      while ((end_index+1) < *group_end_index && timestamp_vec[end_index+1] <= (timestamp_vec[i]+following_window_in_days)) {
        ++end_index;
      }
      ++end_index; // One past the last.

      // aggregate
      size_type count = 0;
      for (size_type j = start_index; j < end_index; j++) {
        if (include_nulls || !input.nullable() || cudf::bit_is_set(valid_mask, j))
          count++;
      }

      ref_valid[i] = (count >= min_periods);
      if (ref_valid[i])
        ref_data[i] = count;
    }

    fixed_width_column_wrapper<cudf::size_type> col(ref_data.begin(), ref_data.end(),
                                                    ref_valid.begin());
    return col.release();
  }

  template<typename agg_op, cudf::experimental::aggregation::Kind k, typename OutputType, bool is_mean,
           std::enable_if_t<cudf::detail::is_supported<T, agg_op, k, is_mean>()>* = nullptr>
  std::unique_ptr<cudf::column>
  create_reference_output(cudf::column_view const& timestamp_column,
                          cudf::column_view const& input,
                          std::vector<size_type> const& group_offsets,
                          size_type const& preceding_window_in_days,
                          size_type const& following_window_in_days,
                          size_type min_periods)
  {
    assert(timestamp_column.type().id() == cudf::TIMESTAMP_DAYS); // Testing with DAYS.

    auto timestamp_vec = cudf::test::to_host<int32_t>(timestamp_column).first;

    size_type num_rows = input.size();
    thrust::host_vector<OutputType> ref_data(num_rows);
    thrust::host_vector<bool>       ref_valid(num_rows);

    // input data and mask
    thrust::host_vector<T> in_col;
    std::vector<bitmask_type> in_valid; 
    std::tie(in_col, in_valid) = cudf::test::to_host<T>(input); 
    bitmask_type* valid_mask = in_valid.data();
    
    agg_op op;
    for(size_type i = 0; i < num_rows; i++) {
      OutputType val = agg_op::template identity<OutputType>();

      // load sizes
      min_periods = std::max(min_periods, 1); // at least one observation is required

      // compute bounds
      auto group_end_index = std::upper_bound(group_offsets.begin(), group_offsets.end(), i);
      auto group_start_index = group_end_index - 1;

      size_type start_index = i;
      while ((start_index-1) >= *group_start_index && timestamp_vec[start_index-1] >= (timestamp_vec[i]-preceding_window_in_days)) {
       --start_index; 
      }

      size_type end_index = i;
      while ((end_index+1) < *group_end_index && timestamp_vec[end_index+1] <= (timestamp_vec[i]+following_window_in_days)) {
        ++end_index;
      }
      ++end_index; // One past the last.
      
      // aggregate
      size_type count = 0;
      for (size_type j = start_index; j < end_index; j++) {
        if (!input.nullable() || cudf::bit_is_set(valid_mask, j)) {
          val = op(static_cast<OutputType>(in_col[j]), val);
          count++;
        }
      }

      ref_valid[i] = (count >= min_periods);
      if (ref_valid[i]) {
        cudf::detail::store_output_functor<OutputType, is_mean>{}(ref_data[i], val, count);
      }
    }

    fixed_width_column_wrapper<OutputType> col(ref_data.begin(), ref_data.end(), ref_valid.begin());
    return col.release();
  }

  template<typename agg_op, cudf::experimental::aggregation::Kind k, typename OutputType, bool is_mean,
           std::enable_if_t<!cudf::detail::is_supported<T, agg_op, k, is_mean>()>* = nullptr>
  std::unique_ptr<cudf::column> create_reference_output(cudf::column_view const& timestamp_column,
                                                        cudf::column_view const& input, 
                                                        std::vector<size_type> const& group_offsets,
                                                        size_type const& preceding_window_col,
                                                        size_type const& following_window_col,
                                                        size_type min_periods)
  {
    CUDF_FAIL("Unsupported combination of type and aggregation");
  }

  std::unique_ptr<cudf::column> create_reference_output(std::unique_ptr<cudf::experimental::aggregation>const& op,
                                                        cudf::column_view const& timestamp_column,
                                                        cudf::column_view const& input,
                                                        std::vector<size_type> const& group_offsets,
                                                        size_type const& preceding_window,
                                                        size_type const& following_window,
                                                        size_type min_periods)
  {
    // unroll aggregation types
    switch(op->kind) {
    case cudf::experimental::aggregation::SUM:
      return create_reference_output<cudf::DeviceSum, cudf::experimental::aggregation::SUM, 
             cudf::experimental::detail::target_type_t<T, cudf::experimental::aggregation::SUM>, false>(timestamp_column, input, group_offsets, preceding_window,
                                                             following_window, min_periods);
    case cudf::experimental::aggregation::MIN:
      return create_reference_output<cudf::DeviceMin, cudf::experimental::aggregation::MIN,
             cudf::experimental::detail::target_type_t<T, cudf::experimental::aggregation::MIN>, false>(timestamp_column, input, group_offsets, preceding_window,
                                                             following_window, min_periods);
    case cudf::experimental::aggregation::MAX:
      return create_reference_output<cudf::DeviceMax, cudf::experimental::aggregation::MAX,
             cudf::experimental::detail::target_type_t<T, cudf::experimental::aggregation::MAX>, false>(timestamp_column, input, group_offsets, preceding_window,
                                                             following_window, min_periods);
    case cudf::experimental::aggregation::COUNT_VALID:
      return create_count_reference_output<false>(timestamp_column, input, group_offsets, preceding_window, following_window, min_periods);
    case cudf::experimental::aggregation::COUNT_ALL:
      return create_count_reference_output<true>(timestamp_column, input, group_offsets, preceding_window, following_window, min_periods);
    case cudf::experimental::aggregation::MEAN:
      return create_reference_output<cudf::DeviceSum, cudf::experimental::aggregation::MEAN,
             cudf::experimental::detail::target_type_t<T, cudf::experimental::aggregation::MEAN>, true>(timestamp_column, input, group_offsets, preceding_window,
                                                            following_window, min_periods);
    default:
      return fixed_width_column_wrapper<T>({}).release();
    }
  }
};

TYPED_TEST_CASE(GroupedTimeRangeRollingTest, cudf::test::FixedWidthTypes);

TYPED_TEST(GroupedTimeRangeRollingTest, SimplePartitionedStaticWindowsWithGroupKeysAndTimeRanges)
{
  const size_type DATA_SIZE {static_cast<size_type>(18)};
  const std::vector<TypeParam> col_data (DATA_SIZE, 1);
  const std::vector<bool>      col_mask (DATA_SIZE, true);
  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());

  // 2 grouping keys, with effectively 3 groups of at most 6 rows each: 
  //   1. key_0 {0, 0, 0, ...0}
  //   2. key_1 {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2}
  std::vector<TypeParam> key_0_vec(DATA_SIZE, 0);
  std::vector<TypeParam> key_1_vec;
  int i{0};
  std::generate_n(std::back_inserter(key_1_vec), DATA_SIZE, [&i](){return i++/6;}); // Groups of 6.
  const fixed_width_column_wrapper<TypeParam> key_0 (key_0_vec.begin(), key_0_vec.end());
  const fixed_width_column_wrapper<TypeParam> key_1 (key_1_vec.begin(), key_1_vec.end());
  const cudf::table_view grouping_keys {std::vector<cudf::column_view>{key_0, key_1}};

  size_type preceding_window_in_days = 1;
  size_type following_window_in_days = 1;
  std::vector<size_type> expected_group_offsets{0, 6, 12, DATA_SIZE};

  // Timestamp column.
  std::vector<int32_t> timestamp_days_vec {0, 2, 3, 4, 5, 7, 0, 0, 1, 2, 3, 3, 0, 1, 2, 3, 3, 3};
  fixed_width_column_wrapper<cudf::timestamp_D> timestamp_days(timestamp_days_vec.begin(), timestamp_days_vec.end());

  this->run_test_col_agg(grouping_keys, timestamp_days, input, expected_group_offsets, preceding_window_in_days, following_window_in_days, 1);
}

CUDF_TEST_PROGRAM_MAIN()
