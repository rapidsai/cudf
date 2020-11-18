/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/utilities/bit.hpp>
#include <src/rolling/rolling_detail.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <vector>

using cudf::bitmask_type;
using cudf::size_type;
using cudf::test::fixed_width_column_wrapper;

class RollingStringTest : public cudf::test::BaseFixture {
};

TEST_F(RollingStringTest, NoNullStringMinMaxCount)
{
  cudf::test::strings_column_wrapper input(
    {"This", "is", "rolling", "test", "being", "operated", "on", "string", "column"});
  std::vector<size_type> window{2};
  cudf::test::strings_column_wrapper expected_min(
    {"This", "This", "being", "being", "being", "being", "column", "column", "column"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::strings_column_wrapper expected_max(
    {"rolling", "test", "test", "test", "test", "string", "string", "string", "string"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1});
  fixed_width_column_wrapper<size_type> expected_count({3, 4, 4, 4, 4, 4, 4, 3, 2},
                                                       {1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto got_min = cudf::rolling_window(input, window[0], window[0], 1, cudf::make_min_aggregation());
  auto got_max = cudf::rolling_window(input, window[0], window[0], 1, cudf::make_max_aggregation());
  auto got_count_valid =
    cudf::rolling_window(input, window[0], window[0], 1, cudf::make_count_aggregation());
  auto got_count_all = cudf::rolling_window(
    input, window[0], window[0], 1, cudf::make_count_aggregation(cudf::null_policy::INCLUDE));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, got_min->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, got_max->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count, got_count_valid->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count, got_count_all->view());
}

TEST_F(RollingStringTest, NullStringMinMaxCount)
{
  cudf::test::strings_column_wrapper input(
    {"This", "is", "rolling", "test", "being", "operated", "on", "string", "column"},
    {1, 0, 0, 1, 0, 1, 1, 1, 0});
  std::vector<size_type> window{2};
  cudf::test::strings_column_wrapper expected_min(
    {"This", "This", "test", "operated", "on", "on", "on", "on", "string"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::strings_column_wrapper expected_max(
    {"This", "test", "test", "test", "test", "string", "string", "string", "string"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1});
  fixed_width_column_wrapper<size_type> expected_count_val({1, 2, 1, 2, 3, 3, 3, 2, 1},
                                                           {1, 1, 1, 1, 1, 1, 1, 1, 1});
  fixed_width_column_wrapper<size_type> expected_count_all({3, 4, 4, 4, 4, 4, 4, 3, 2},
                                                           {1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto got_min = cudf::rolling_window(input, window[0], window[0], 1, cudf::make_min_aggregation());
  auto got_max = cudf::rolling_window(input, window[0], window[0], 1, cudf::make_max_aggregation());
  auto got_count_valid =
    cudf::rolling_window(input, window[0], window[0], 1, cudf::make_count_aggregation());
  auto got_count_all = cudf::rolling_window(
    input, window[0], window[0], 1, cudf::make_count_aggregation(cudf::null_policy::INCLUDE));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, got_min->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, got_max->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_val, got_count_valid->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_all, got_count_all->view());
}

TEST_F(RollingStringTest, MinPeriods)
{
  cudf::test::strings_column_wrapper input(
    {"This", "is", "rolling", "test", "being", "operated", "on", "string", "column"},
    {1, 0, 0, 1, 0, 1, 1, 1, 0});
  std::vector<size_type> window{2};
  cudf::test::strings_column_wrapper expected_min(
    {"This", "This", "This", "operated", "on", "on", "on", "on", "on"},
    {0, 0, 0, 0, 1, 1, 1, 0, 0});
  cudf::test::strings_column_wrapper expected_max(
    {"This", "test", "test", "test", "test", "string", "string", "string", "string"},
    {0, 0, 0, 0, 1, 1, 1, 0, 0});
  fixed_width_column_wrapper<size_type> expected_count_val({1, 2, 1, 2, 3, 3, 3, 2, 2},
                                                           {1, 1, 1, 1, 1, 1, 1, 1, 0});
  fixed_width_column_wrapper<size_type> expected_count_all({3, 4, 4, 4, 4, 4, 4, 3, 2},
                                                           {0, 1, 1, 1, 1, 1, 1, 0, 0});

  auto got_min = cudf::rolling_window(input, window[0], window[0], 3, cudf::make_min_aggregation());
  auto got_max = cudf::rolling_window(input, window[0], window[0], 3, cudf::make_max_aggregation());
  auto got_count_valid =
    cudf::rolling_window(input, window[0], window[0], 3, cudf::make_count_aggregation());
  auto got_count_all = cudf::rolling_window(
    input, window[0], window[0], 4, cudf::make_count_aggregation(cudf::null_policy::INCLUDE));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, got_min->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, got_max->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_val, got_count_valid->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_all, got_count_all->view());
}

TEST_F(RollingStringTest, ZeroWindowSize)
{
  cudf::test::strings_column_wrapper input(
    {"This", "is", "rolling", "test", "being", "operated", "on", "string", "column"},
    {1, 0, 0, 1, 0, 1, 1, 1, 0});
  fixed_width_column_wrapper<size_type> expected_count({0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto got_count = cudf::rolling_window(input, 0, 0, 0, cudf::make_count_aggregation());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count, got_count->view());
}

template <typename T>
class RollingTest : public cudf::test::BaseFixture {
 protected:
  // input as column_wrapper
  void run_test_col(cudf::column_view const& input,
                    const std::vector<size_type>& preceding_window,
                    const std::vector<size_type>& following_window,
                    size_type min_periods,
                    std::unique_ptr<cudf::aggregation> const& op)
  {
    std::unique_ptr<cudf::column> output;

    // wrap windows
    if (preceding_window.size() > 1) {
      fixed_width_column_wrapper<size_type> preceding_window_wrapper(preceding_window.begin(),
                                                                     preceding_window.end());
      fixed_width_column_wrapper<size_type> following_window_wrapper(following_window.begin(),
                                                                     following_window.end());

      EXPECT_NO_THROW(
        output = cudf::rolling_window(
          input, preceding_window_wrapper, following_window_wrapper, min_periods, op));
    } else {
      EXPECT_NO_THROW(output = cudf::rolling_window(
                        input, preceding_window[0], following_window[0], min_periods, op));
    }

    auto reference =
      create_reference_output(op, input, preceding_window, following_window, min_periods);

#if 0
    std::cout << "input:\n";
    cudf::test::print(input, std::cout, ", ");
    std::cout << "\n";
    std::cout << "output:\n";
    cudf::test::print(*output, std::cout, ", ");
    std::cout << "\n";
    std::cout << "reference:\n";
    cudf::test::print(reference, std::cout, ", ");
    std::cout << "\n";
    std::cout << "\n";
#endif

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, *reference);
  }

  // helper function to test all aggregators
  void run_test_col_agg(cudf::column_view const& input,
                        const std::vector<size_type>& preceding_window,
                        const std::vector<size_type>& following_window,
                        size_type min_periods)
  {
    // test all supported aggregators
    run_test_col(
      input, preceding_window, following_window, min_periods, cudf::make_min_aggregation());
    run_test_col(
      input, preceding_window, following_window, min_periods, cudf::make_count_aggregation());
    run_test_col(input,
                 preceding_window,
                 following_window,
                 min_periods,
                 cudf::make_count_aggregation(cudf::null_policy::INCLUDE));
    run_test_col(
      input, preceding_window, following_window, min_periods, cudf::make_max_aggregation());

    if (not cudf::is_timestamp(input.type())) {
      run_test_col(
        input, preceding_window, following_window, min_periods, cudf::make_sum_aggregation());
      run_test_col(
        input, preceding_window, following_window, min_periods, cudf::make_mean_aggregation());
    }
  }

 private:
  // use SFINAE to only instantiate for supported combinations

  // specialization for COUNT_VALID, COUNT_ALL
  template <bool include_nulls>
  std::unique_ptr<cudf::column> create_count_reference_output(
    cudf::column_view const& input,
    std::vector<size_type> const& preceding_window_col,
    std::vector<size_type> const& following_window_col,
    size_type min_periods)
  {
    size_type num_rows = input.size();
    std::vector<cudf::size_type> ref_data(num_rows);
    std::vector<bool> ref_valid(num_rows);

    // input data and mask

    std::vector<bitmask_type> in_valid = cudf::test::bitmask_to_host(input);
    bitmask_type* valid_mask           = in_valid.data();

    for (size_type i = 0; i < num_rows; i++) {
      // load sizes
      min_periods = std::max(min_periods, 1);  // at least one observation is required

      // compute bounds
      auto preceding_window = preceding_window_col[i % preceding_window_col.size()];
      auto following_window = following_window_col[i % following_window_col.size()];
      size_type start       = std::min(num_rows, std::max(0, i - preceding_window + 1));
      size_type end         = std::min(num_rows, std::max(0, i + following_window + 1));
      size_type start_index = std::min(start, end);
      size_type end_index   = std::max(start, end);

      // aggregate
      size_type count = 0;
      for (size_type j = start_index; j < end_index; j++) {
        if (include_nulls || !input.nullable() || cudf::bit_is_set(valid_mask, j)) count++;
      }

      ref_valid[i] = ((end_index - start_index) >= min_periods);
      if (ref_valid[i]) ref_data[i] = count;
    }

    fixed_width_column_wrapper<cudf::size_type> col(
      ref_data.begin(), ref_data.end(), ref_valid.begin());
    return col.release();
  }

  template <typename agg_op,
            cudf::aggregation::Kind k,
            typename OutputType,
            bool is_mean,
            std::enable_if_t<cudf::detail::is_rolling_supported<T, agg_op, k>()>* = nullptr>
  std::unique_ptr<cudf::column> create_reference_output(
    cudf::column_view const& input,
    std::vector<size_type> const& preceding_window_col,
    std::vector<size_type> const& following_window_col,
    size_type min_periods)
  {
    size_type num_rows = input.size();
    thrust::host_vector<OutputType> ref_data(num_rows);
    thrust::host_vector<bool> ref_valid(num_rows);

    // input data and mask
    thrust::host_vector<T> in_col;
    std::vector<bitmask_type> in_valid;
    std::tie(in_col, in_valid) = cudf::test::to_host<T>(input);
    bitmask_type* valid_mask   = in_valid.data();

    agg_op op;
    for (size_type i = 0; i < num_rows; i++) {
      OutputType val = agg_op::template identity<OutputType>();

      // load sizes
      min_periods = std::max(min_periods, 1);  // at least one observation is required

      // compute bounds
      auto preceding_window = preceding_window_col[i % preceding_window_col.size()];
      auto following_window = following_window_col[i % following_window_col.size()];
      size_type start       = std::min(num_rows, std::max(0, i - preceding_window + 1));
      size_type end         = std::min(num_rows, std::max(0, i + following_window + 1));
      size_type start_index = std::min(start, end);
      size_type end_index   = std::max(start, end);

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
        cudf::detail::rolling_store_output_functor<OutputType, is_mean>{}(ref_data[i], val, count);
      }
    }

    fixed_width_column_wrapper<OutputType> col(ref_data.begin(), ref_data.end(), ref_valid.begin());
    return col.release();
  }

  template <typename agg_op,
            cudf::aggregation::Kind k,
            typename OutputType,
            bool is_mean,
            std::enable_if_t<!cudf::detail::is_rolling_supported<T, agg_op, k>()>* = nullptr>
  std::unique_ptr<cudf::column> create_reference_output(
    cudf::column_view const& input,
    std::vector<size_type> const& preceding_window_col,
    std::vector<size_type> const& following_window_col,
    size_type min_periods)
  {
    CUDF_FAIL("Unsupported combination of type and aggregation");
  }

  std::unique_ptr<cudf::column> create_reference_output(
    std::unique_ptr<cudf::aggregation> const& op,
    cudf::column_view const& input,
    std::vector<size_type> const& preceding_window,
    std::vector<size_type> const& following_window,
    size_type min_periods)
  {
    // unroll aggregation types
    switch (op->kind) {
      case cudf::aggregation::SUM:
        return create_reference_output<cudf::DeviceSum,
                                       cudf::aggregation::SUM,
                                       cudf::detail::target_type_t<T, cudf::aggregation::SUM>,
                                       false>(
          input, preceding_window, following_window, min_periods);
      case cudf::aggregation::MIN:
        return create_reference_output<cudf::DeviceMin,
                                       cudf::aggregation::MIN,
                                       cudf::detail::target_type_t<T, cudf::aggregation::MIN>,
                                       false>(
          input, preceding_window, following_window, min_periods);
      case cudf::aggregation::MAX:
        return create_reference_output<cudf::DeviceMax,
                                       cudf::aggregation::MAX,
                                       cudf::detail::target_type_t<T, cudf::aggregation::MAX>,
                                       false>(
          input, preceding_window, following_window, min_periods);
      case cudf::aggregation::COUNT_VALID:
        return create_count_reference_output<false>(
          input, preceding_window, following_window, min_periods);
      case cudf::aggregation::COUNT_ALL:
        return create_count_reference_output<true>(
          input, preceding_window, following_window, min_periods);
      case cudf::aggregation::MEAN:
        return create_reference_output<cudf::DeviceSum,
                                       cudf::aggregation::MEAN,
                                       cudf::detail::target_type_t<T, cudf::aggregation::MEAN>,
                                       true>(
          input, preceding_window, following_window, min_periods);
      default: return fixed_width_column_wrapper<T>({}).release();
    }
  }
};

// // ------------- expected failures --------------------

class RollingErrorTest : public cudf::test::BaseFixture {
};

// negative sizes
TEST_F(RollingErrorTest, NegativeMinPeriods)
{
  const std::vector<size_type> col_data = {0, 1, 2, 0, 4};
  const std::vector<bool> col_valid     = {1, 1, 1, 0, 1};
  fixed_width_column_wrapper<size_type> input(col_data.begin(), col_data.end(), col_valid.begin());

  EXPECT_THROW(cudf::rolling_window(input, 2, 2, -2, cudf::make_sum_aggregation()),
               cudf::logic_error);
}

// window array size mismatch
TEST_F(RollingErrorTest, WindowArraySizeMismatch)
{
  const std::vector<size_type> col_data = {0, 1, 2, 0, 4};
  const std::vector<bool> col_valid     = {1, 1, 1, 0, 1};
  fixed_width_column_wrapper<size_type> input(col_data.begin(), col_data.end(), col_valid.begin());

  std::vector<size_type> five({2, 1, 2, 1, 4});
  std::vector<size_type> four({1, 2, 3, 4});
  fixed_width_column_wrapper<size_type> five_elements(five.begin(), five.end());
  fixed_width_column_wrapper<size_type> four_elements(four.begin(), four.end());

  // this runs ok
  EXPECT_NO_THROW(
    cudf::rolling_window(input, five_elements, five_elements, 1, cudf::make_sum_aggregation()));

  // mismatch for the window array
  EXPECT_THROW(
    cudf::rolling_window(input, four_elements, five_elements, 1, cudf::make_sum_aggregation()),
    cudf::logic_error);

  // mismatch for the forward window array
  EXPECT_THROW(
    cudf::rolling_window(input, five_elements, four_elements, 1, cudf::make_sum_aggregation()),
    cudf::logic_error);
}

TEST_F(RollingErrorTest, EmptyInput)
{
  cudf::test::fixed_width_column_wrapper<int32_t> empty_col{};
  std::unique_ptr<cudf::column> output;
  EXPECT_NO_THROW(output = cudf::rolling_window(empty_col, 2, 0, 2, cudf::make_sum_aggregation()));
  EXPECT_EQ(output->size(), 0);

  fixed_width_column_wrapper<int32_t> preceding_window{};
  fixed_width_column_wrapper<int32_t> following_window{};
  EXPECT_NO_THROW(
    output = cudf::rolling_window(
      empty_col, preceding_window, following_window, 2, cudf::make_sum_aggregation()));
  EXPECT_EQ(output->size(), 0);

  fixed_width_column_wrapper<int32_t> nonempty_col{{1, 2, 3}};
  EXPECT_NO_THROW(
    output = cudf::rolling_window(
      nonempty_col, preceding_window, following_window, 2, cudf::make_sum_aggregation()));
  EXPECT_EQ(output->size(), 0);
}

TEST_F(RollingErrorTest, SizeMismatch)
{
  fixed_width_column_wrapper<int32_t> nonempty_col{{1, 2, 3}};
  std::unique_ptr<cudf::column> output;

  {
    fixed_width_column_wrapper<int32_t> preceding_window{{1, 1}};  // wrong size
    fixed_width_column_wrapper<int32_t> following_window{{1, 1, 1}};
    EXPECT_THROW(
      output = cudf::rolling_window(
        nonempty_col, preceding_window, following_window, 2, cudf::make_sum_aggregation()),
      cudf::logic_error);
  }
  {
    fixed_width_column_wrapper<int32_t> preceding_window{{1, 1, 1}};
    fixed_width_column_wrapper<int32_t> following_window{{1, 2}};  // wrong size
    EXPECT_THROW(
      output = cudf::rolling_window(
        nonempty_col, preceding_window, following_window, 2, cudf::make_sum_aggregation()),
      cudf::logic_error);
  }
}

TEST_F(RollingErrorTest, WindowWrongDtype)
{
  fixed_width_column_wrapper<int32_t> nonempty_col{{1, 2, 3}};
  std::unique_ptr<cudf::column> output;

  fixed_width_column_wrapper<float> preceding_window{{1.0f, 1.0f, 1.0f}};
  fixed_width_column_wrapper<float> following_window{{1.0f, 1.0f, 1.0f}};
  EXPECT_THROW(output = cudf::rolling_window(
                 nonempty_col, preceding_window, following_window, 2, cudf::make_sum_aggregation()),
               cudf::logic_error);
}

// incorrect type/aggregation combo: sum of timestamps
TEST_F(RollingErrorTest, SumTimestampNotSupported)
{
  constexpr size_type size{10};
  fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> input_D(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(size));
  fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> input_s(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(size));
  fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep> input_ms(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(size));
  fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep> input_us(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(size));
  fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep> input_ns(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(size));

  EXPECT_THROW(cudf::rolling_window(input_D, 2, 2, 0, cudf::make_sum_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(input_s, 2, 2, 0, cudf::make_sum_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(input_ms, 2, 2, 0, cudf::make_sum_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(input_us, 2, 2, 0, cudf::make_sum_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(input_ns, 2, 2, 0, cudf::make_sum_aggregation()),
               cudf::logic_error);
}

// incorrect type/aggregation combo: mean of timestamps
TEST_F(RollingErrorTest, MeanTimestampNotSupported)
{
  constexpr size_type size{10};
  fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> input_D(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(size));
  fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> input_s(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(size));
  fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep> input_ms(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(size));
  fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep> input_us(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(size));
  fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep> input_ns(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(size));

  EXPECT_THROW(cudf::rolling_window(input_D, 2, 2, 0, cudf::make_mean_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(input_s, 2, 2, 0, cudf::make_mean_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(input_ms, 2, 2, 0, cudf::make_mean_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(input_us, 2, 2, 0, cudf::make_mean_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(input_ns, 2, 2, 0, cudf::make_mean_aggregation()),
               cudf::logic_error);
}

TYPED_TEST_CASE(RollingTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

// simple example from Pandas docs
TYPED_TEST(RollingTest, SimpleStatic)
{
  // https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
  const std::vector<TypeParam> col_data =
    cudf::test::make_type_param_vector<TypeParam>({0, 1, 2, 0, 4});
  const std::vector<bool> col_mask = {1, 1, 1, 0, 1};

  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<size_type> window{2};

  // static sizes
  this->run_test_col_agg(input, window, window, 1);
}

// negative sizes
TYPED_TEST(RollingTest, NegativeWindowSizes)
{
  auto const col_data  = cudf::test::make_type_param_vector<TypeParam>({0, 1, 2, 0, 4});
  auto const col_valid = std::vector<bool>{1, 1, 1, 0, 1};
  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_valid.begin());
  std::vector<size_type> window{3};
  std::vector<size_type> negative_window{-2};

  this->run_test_col_agg(input, negative_window, window, 1);
  this->run_test_col_agg(input, window, negative_window, 1);
  this->run_test_col_agg(input, negative_window, negative_window, 1);
}

// simple example from Pandas docs:
TYPED_TEST(RollingTest, SimpleDynamic)
{
  // https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
  const std::vector<TypeParam> col_data =
    cudf::test::make_type_param_vector<TypeParam>({0, 1, 2, 0, 4});
  const std::vector<bool> col_mask = {1, 1, 1, 0, 1};

  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<size_type> preceding_window({1, 2, 3, 4, 2});
  std::vector<size_type> following_window({2, 1, 2, 1, 2});

  // dynamic sizes
  this->run_test_col_agg(input, preceding_window, following_window, 1);
}

// this is a special test to check the volatile count variable issue (see rolling.cu for detail)
TYPED_TEST(RollingTest, VolatileCount)
{
  const std::vector<TypeParam> col_data =
    cudf::test::make_type_param_vector<TypeParam>({8, 70, 45, 20, 59, 80});
  const std::vector<bool> col_mask = {1, 1, 0, 0, 1, 0};

  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<size_type> preceding_window({5, 9, 4, 8, 3, 3});
  std::vector<size_type> following_window({1, 1, 9, 2, 8, 9});

  // dynamic sizes
  this->run_test_col_agg(input, preceding_window, following_window, 1);
}

// all rows are invalid
TYPED_TEST(RollingTest, AllInvalid)
{
  size_type num_rows = 1000;

  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool> col_mask(num_rows, 0);

  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<size_type> window({100});
  size_type periods = 100;

  this->run_test_col_agg(input, window, window, periods);
}

// window = following_window = 0
TYPED_TEST(RollingTest, ZeroWindow)
{
  size_type num_rows = 1000;

  std::vector<int> col_data(num_rows, 1);
  std::vector<bool> col_mask(num_rows, 1);

  fixed_width_column_wrapper<TypeParam, int> input(
    col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<size_type> window({0});
  size_type periods = num_rows;

  this->run_test_col_agg(input, window, window, periods);
}

// min_periods = 0
TYPED_TEST(RollingTest, ZeroPeriods)
{
  size_type num_rows = 1000;

  std::vector<int> col_data(num_rows, 1);
  std::vector<bool> col_mask(num_rows, 1);

  fixed_width_column_wrapper<TypeParam, int> input(
    col_data.begin(), col_data.end(), col_mask.begin());

  std::vector<size_type> window({num_rows});
  size_type periods = 0;

  this->run_test_col_agg(input, window, window, periods);
}

// window in one direction is not large enough to collect enough samples,
//   but if using both directions we should get == min_periods,
// also tests out of boundary accesses
TYPED_TEST(RollingTest, BackwardForwardWindow)
{
  size_type num_rows = 1000;

  std::vector<int> col_data(num_rows, 1);
  std::vector<bool> col_mask(num_rows, 1);

  fixed_width_column_wrapper<TypeParam, int> input(
    col_data.begin(), col_data.end(), col_mask.begin());

  std::vector<size_type> window({num_rows});
  size_type periods = num_rows;

  this->run_test_col_agg(input, window, window, periods);
}

// random input data, static parameters, no nulls
TYPED_TEST(RollingTest, RandomStaticAllValid)
{
  size_type num_rows = 10000;

  // random input
  std::vector<TypeParam> col_data(num_rows);
  cudf::test::UniformRandomGenerator<TypeParam> rng;
  std::generate(col_data.begin(), col_data.end(), [&rng]() { return rng.generate(); });
  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end());

  std::vector<size_type> window({50});
  size_type periods = 50;

  this->run_test_col_agg(input, window, window, periods);
}

// random input data, static parameters, with nulls
TYPED_TEST(RollingTest, RandomStaticWithInvalid)
{
  size_type num_rows = 10000;

  // random input
  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool> col_valid(num_rows);
  cudf::test::UniformRandomGenerator<TypeParam> rng;
  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(col_data.begin(), col_data.end(), [&rng]() { return rng.generate(); });
  std::generate(col_valid.begin(), col_valid.end(), [&rbg]() { return rbg.generate(); });
  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_valid.begin());

  std::vector<size_type> window({50});
  size_type periods = 50;

  this->run_test_col_agg(input, window, window, periods);
}

// random input data, dynamic parameters, no nulls
TYPED_TEST(RollingTest, RandomDynamicAllValid)
{
  size_type num_rows        = 50000;
  size_type max_window_size = 50;

  // random input
  std::vector<TypeParam> col_data(num_rows);
  cudf::test::UniformRandomGenerator<TypeParam> rng;
  std::generate(col_data.begin(), col_data.end(), [&rng]() { return rng.generate(); });
  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end());

  // random parameters
  cudf::test::UniformRandomGenerator<size_type> window_rng(0, max_window_size);
  auto generator = [&]() { return window_rng.generate(); };

  std::vector<size_type> preceding_window(num_rows);
  std::vector<size_type> following_window(num_rows);

  std::generate(preceding_window.begin(), preceding_window.end(), generator);
  std::generate(following_window.begin(), following_window.end(), generator);

  this->run_test_col_agg(input, preceding_window, following_window, max_window_size);
}

// random input data, dynamic parameters, with nulls
TYPED_TEST(RollingTest, RandomDynamicWithInvalid)
{
  size_type num_rows        = 50000;
  size_type max_window_size = 50;

  // random input with nulls
  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool> col_valid(num_rows);
  cudf::test::UniformRandomGenerator<TypeParam> rng;
  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(col_data.begin(), col_data.end(), [&rng]() { return rng.generate(); });
  std::generate(col_valid.begin(), col_valid.end(), [&rbg]() { return rbg.generate(); });
  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_valid.begin());

  // random parameters
  cudf::test::UniformRandomGenerator<size_type> window_rng(0, max_window_size);
  auto generator = [&]() { return window_rng.generate(); };

  std::vector<size_type> preceding_window(num_rows);
  std::vector<size_type> following_window(num_rows);

  std::generate(preceding_window.begin(), preceding_window.end(), generator);
  std::generate(following_window.begin(), following_window.end(), generator);

  this->run_test_col_agg(input, preceding_window, following_window, max_window_size);
}

// ------------- non-fixed-width types --------------------

using RollingTestStrings = RollingTest<cudf::string_view>;

TEST_F(RollingTestStrings, StringsUnsupportedOperators)
{
  cudf::test::strings_column_wrapper input{{"This", "is", "not", "a", "string", "type"},
                                           {1, 1, 1, 0, 1, 0}};

  std::vector<size_type> window{1};

  EXPECT_THROW(cudf::rolling_window(input, 2, 2, 0, cudf::make_sum_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(input, 2, 2, 0, cudf::make_mean_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(
                 input,
                 2,
                 2,
                 0,
                 cudf::make_udf_aggregation(cudf::udf_type::PTX, std::string{}, cudf::data_type{})),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(input,
                                    2,
                                    2,
                                    0,
                                    cudf::make_udf_aggregation(
                                      cudf::udf_type::CUDA, std::string{}, cudf::data_type{})),
               cudf::logic_error);
}

/*TEST_F(RollingTestStrings, SimpleStatic)
{
  cudf::test::strings_column_wrapper input{{"This", "is", "not", "a", "string", "type"},
                                           {1, 1, 1, 0, 1, 0}};

  std::vector<size_type> window{1};

  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::MIN));
  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::MAX));
  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::COUNT_VALID));
  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::COUNT_ALL));
}*/

struct RollingTestUdf : public cudf::test::BaseFixture {
  const std::string cuda_func{
    R"***(
      template <typename OutType, typename InType>
      __device__ void CUDA_GENERIC_AGGREGATOR(OutType *ret, InType *in_col, cudf::size_type start,
                                              cudf::size_type count) {
        OutType val = 0;
        for (cudf::size_type i = 0; i < count; i++) {
          val += in_col[start + i];
        }
        *ret = val;
      }
    )***"};

  const std::string ptx_func{
    R"***(
    //
    // Generated by NVIDIA NVVM Compiler
    //
    // Compiler Build ID: CL-24817639
    // Cuda compilation tools, release 10.0, V10.0.130
    // Based on LLVM 3.4svn
    //

    .version 6.3
    .target sm_70
    .address_size 64

    // .globl	_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE
    .common .global .align 8 .u64 _ZN08NumbaEnv8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE;

    .visible .func  (.param .b32 func_retval0) _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE(
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_0,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_1,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_2,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_3,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_4,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_5,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_6,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_7
    )
    {
    .reg .pred 	%p<3>;
    .reg .b32 	%r<6>;
    .reg .b64 	%rd<18>;


    ld.param.u64 	%rd6, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_0];
    ld.param.u64 	%rd7, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_5];
    ld.param.u64 	%rd8, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_6];
    ld.param.u64 	%rd9, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_7];
    mov.u64 	%rd15, 0;
    mov.u64 	%rd16, %rd15;

    BB0_1:
    mov.u64 	%rd2, %rd16;
    mov.u32 	%r5, 0;
    setp.ge.s64	%p1, %rd15, %rd8;
    mov.u64 	%rd17, %rd15;
    @%p1 bra 	BB0_3;

    mul.lo.s64 	%rd12, %rd15, %rd9;
    add.s64 	%rd13, %rd12, %rd7;
    ld.u32 	%r5, [%rd13];
    add.s64 	%rd17, %rd15, 1;

    BB0_3:
    cvt.s64.s32	%rd14, %r5;
    add.s64 	%rd16, %rd14, %rd2;
    setp.lt.s64	%p2, %rd15, %rd8;
    mov.u64 	%rd15, %rd17;
    @%p2 bra 	BB0_1;

    st.u64 	[%rd6], %rd2;
    mov.u32 	%r4, 0;
    st.param.b32	[func_retval0+0], %r4;
    ret;
    }
    )***"};
};

TEST_F(RollingTestUdf, StaticWindow)
{
  size_type size = 1000;

  fixed_width_column_wrapper<int32_t> input(thrust::make_counting_iterator(0),
                                            thrust::make_counting_iterator(size),
                                            thrust::make_constant_iterator(true));

  std::unique_ptr<cudf::column> output;

  auto start = cudf::test::make_counting_transform_iterator(0, [size] __device__(size_type row) {
    return std::accumulate(thrust::make_counting_iterator(std::max(0, row - 2 + 1)),
                           thrust::make_counting_iterator(std::min(size, row + 2 + 1)),
                           0);
  });

  auto valid = cudf::test::make_counting_transform_iterator(0, [size] __device__(size_type row) {
    return (row != 0 && row != size - 2 && row != size - 1);
  });

  fixed_width_column_wrapper<int64_t> expected{start, start + size, valid};

  // Test CUDA UDF
  auto cuda_udf_agg = cudf::make_udf_aggregation(
    cudf::udf_type::CUDA, this->cuda_func, cudf::data_type{cudf::type_id::INT64});

  EXPECT_NO_THROW(output = cudf::rolling_window(input, 2, 2, 4, cuda_udf_agg));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected);

  // Test NUMBA UDF
  auto ptx_udf_agg = cudf::make_udf_aggregation(
    cudf::udf_type::PTX, this->ptx_func, cudf::data_type{cudf::type_id::INT64});

  EXPECT_NO_THROW(output = cudf::rolling_window(input, 2, 2, 4, ptx_udf_agg));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected);
}

TEST_F(RollingTestUdf, DynamicWindow)
{
  size_type size = 1000;

  fixed_width_column_wrapper<int32_t> input(thrust::make_counting_iterator(0),
                                            thrust::make_counting_iterator(size),
                                            thrust::make_constant_iterator(true));

  auto prec = cudf::test::make_counting_transform_iterator(
    0, [size] __device__(size_type row) { return row % 2 + 2; });

  auto follow = cudf::test::make_counting_transform_iterator(
    0, [size] __device__(size_type row) { return row % 2; });

  fixed_width_column_wrapper<int32_t> preceding(prec, prec + size);
  fixed_width_column_wrapper<int32_t> following(follow, follow + size);
  std::unique_ptr<cudf::column> output;

  auto start = cudf::test::make_counting_transform_iterator(0, [size] __device__(size_type row) {
    return std::accumulate(thrust::make_counting_iterator(std::max(0, row - (row % 2 + 2) + 1)),
                           thrust::make_counting_iterator(std::min(size, row + (row % 2) + 1)),
                           0);
  });

  auto valid = cudf::test::make_counting_transform_iterator(
    0, [size] __device__(size_type row) { return row != 0; });

  fixed_width_column_wrapper<int64_t> expected{start, start + size, valid};

  // Test CUDA UDF
  auto cuda_udf_agg = cudf::make_udf_aggregation(
    cudf::udf_type::CUDA, this->cuda_func, cudf::data_type{cudf::type_id::INT64});

  EXPECT_NO_THROW(output = cudf::rolling_window(input, preceding, following, 2, cuda_udf_agg));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected);

  // Test PTX UDF
  auto ptx_udf_agg = cudf::make_udf_aggregation(
    cudf::udf_type::PTX, this->ptx_func, cudf::data_type{cudf::type_id::INT64});

  EXPECT_NO_THROW(output = cudf::rolling_window(input, preceding, following, 2, ptx_udf_agg));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected);
}

CUDF_TEST_PROGRAM_MAIN()
