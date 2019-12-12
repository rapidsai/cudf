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

#include <cudf/rolling.hpp>
#include <src/rolling/rolling_detail.hpp>

//#include <cudf/strings/convert/convert_datetime.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <vector>

using cudf::test::fixed_width_column_wrapper;
using rolling_operator = cudf::experimental::rolling_operator;
using cudf::size_type;
using cudf::bitmask_type;

template <typename T>
class RollingTest : public cudf::test::BaseFixture {
protected:
  // input as column_wrapper
  void run_test_col(cudf::column_view const& input,
                    const std::vector<size_type> &preceding_window,
                    const std::vector<size_type> &following_window,
                    size_type min_periods,
                    rolling_operator op)
  {
    std::unique_ptr<cudf::column> output;

    // wrap windows
    if (preceding_window.size() > 1) {
      fixed_width_column_wrapper<size_type> preceding_window_wrapper(preceding_window.begin(),
                                                                     preceding_window.end());
      fixed_width_column_wrapper<size_type> following_window_wrapper(following_window.begin(),
                                                                     following_window.end());

      EXPECT_NO_THROW(output = cudf::experimental::rolling_window(input, preceding_window_wrapper,
                                                                  following_window_wrapper,
                                                                  min_periods, op));
    }
    else {
      EXPECT_NO_THROW(output = cudf::experimental::rolling_window(input, preceding_window[0],
                                                                  following_window[0],
                                                                  min_periods, op));
    }

    auto reference = create_reference_output(op, input, preceding_window, following_window,
                                             min_periods);

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

    cudf::test::expect_columns_equal(*output, *reference);
  }

  // helper function to test all aggregators
  void run_test_col_agg(cudf::column_view const& input,
                        const std::vector<size_type> &preceding_window,
                        const std::vector<size_type> &following_window,
                        size_type min_periods)
  {
    // test all supported aggregators
    run_test_col(input, preceding_window, following_window, min_periods, rolling_operator::MIN);
    run_test_col(input, preceding_window, following_window, min_periods, rolling_operator::COUNT);
    run_test_col(input, preceding_window, following_window, min_periods, rolling_operator::MAX);
    run_test_col(input, preceding_window, following_window, min_periods, rolling_operator::MEAN);

    if (!cudf::is_timestamp(input.type()))
      run_test_col(input, preceding_window, following_window, min_periods, rolling_operator::SUM);
  }

  private:

  // use SFINAE to only instantiate for supported combinations

  // specialization for COUNT
  std::unique_ptr<cudf::column> 
  create_count_reference_output(cudf::column_view const& input,
                                std::vector<size_type> const& preceding_window_col,
                                std::vector<size_type> const& following_window_col,
                                size_type min_periods)
  {
    size_type num_rows = input.size();
    std::vector<cudf::size_type> ref_data(num_rows);
    std::vector<bool> ref_valid(num_rows);

    // input data and mask
  
    std::vector<bitmask_type> in_valid = cudf::test::bitmask_to_host(input);
    bitmask_type* valid_mask = in_valid.data();

    for(size_type i = 0; i < num_rows; i++) {
      // load sizes
      min_periods = std::max(min_periods, 1); // at least one observation is required

      // compute bounds
      auto preceding_window = preceding_window_col[i%preceding_window_col.size()];
      auto following_window = following_window_col[i%following_window_col.size()];
      size_type start_index = std::max((size_type)0, i - preceding_window);
      size_type end_index   = std::min(num_rows, i + following_window + 1);

      // aggregate
      size_type count = 0;
      for (size_type j = start_index; j < end_index; j++) {
        if (!input.nullable() || cudf::bit_is_set(valid_mask, j))
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

  template<typename agg_op, bool is_mean,
           std::enable_if_t<cudf::detail::is_supported<T, agg_op, is_mean>()>* = nullptr>
  std::unique_ptr<cudf::column>
  create_reference_output(cudf::column_view const& input,
                          std::vector<size_type> const& preceding_window_col,
                          std::vector<size_type> const& following_window_col,
                          size_type min_periods)
  {
    size_type num_rows = input.size();
    std::vector<T> ref_data(num_rows);
    std::vector<bool> ref_valid(num_rows);

    // input data and mask
    std::vector<T> in_col;
    std::vector<bitmask_type> in_valid; 
    std::tie(in_col, in_valid) = cudf::test::to_host<T>(input); 
    bitmask_type* valid_mask = in_valid.data();
    
    agg_op op;
    for(size_type i = 0; i < num_rows; i++) {
      T val = agg_op::template identity<T>();

      // load sizes
      min_periods = std::max(min_periods, 1); // at least one observation is required

      // compute bounds
      auto preceding_window = preceding_window_col[i%preceding_window_col.size()];
      auto following_window = following_window_col[i%following_window_col.size()];
      size_type start_index = std::max((size_type)0, i - preceding_window);
      size_type end_index   = std::min(num_rows, i + following_window + 1);
      
      // aggregate
      size_type count = 0;
      for (size_type j = start_index; j < end_index; j++) {
        if (!input.nullable() || cudf::bit_is_set(valid_mask, j)) {
          val = op(in_col[j], val);
          count++;
        }
      }

      ref_valid[i] = (count >= min_periods);
      if (ref_valid[i]) {
        cudf::detail::store_output_functor<T, is_mean>{}(ref_data[i], val, count);
      }
    }

    fixed_width_column_wrapper<T> col(ref_data.begin(), ref_data.end(), ref_valid.begin());
    return col.release();
  }

  template<typename  agg_op, bool is_mean,
           std::enable_if_t<!cudf::detail::is_supported<T, agg_op, is_mean>()>* = nullptr>
  std::unique_ptr<cudf::column> create_reference_output(cudf::column_view const& input,
                                                        std::vector<size_type> const& preceding_window_col,
                                                        std::vector<size_type> const& following_window_col,
                                                        size_type min_periods)
  {
    CUDF_FAIL("Unsupported combination of type and aggregation");
  }

  std::unique_ptr<cudf::column> create_reference_output(rolling_operator op,
                                                        cudf::column_view const& input,
                                                        std::vector<size_type> const& preceding_window,
                                                        std::vector<size_type> const& following_window,
                                                        size_type min_periods)
  {
    // unroll aggregation types
    switch(op) {
    case rolling_operator::SUM:
      return create_reference_output<cudf::DeviceSum, false>(input, preceding_window,
                                                             following_window, min_periods);
    case rolling_operator::MIN:
      return create_reference_output<cudf::DeviceMin, false>(input, preceding_window,
                                                             following_window, min_periods);
    case rolling_operator::MAX:
      return create_reference_output<cudf::DeviceMax, false>(input, preceding_window,
                                                             following_window, min_periods);
    case rolling_operator::COUNT:
      return create_count_reference_output(input, preceding_window, following_window, min_periods);
    case rolling_operator::MEAN:
      return create_reference_output<cudf::DeviceSum, true>(input, preceding_window,
                                                            following_window, min_periods);
    default:
      return fixed_width_column_wrapper<T>({}).release();
    }
  }
};


// // ------------- expected failures --------------------

class RollingErrorTest : public cudf::test::BaseFixture {};

// negative sizes
TEST_F(RollingErrorTest, NegativeSizes)
{
  const std::vector<size_type> col_data = {0, 1, 2, 0, 4};
  const std::vector<bool>      col_valid = {1, 1, 1, 0, 1};
  fixed_width_column_wrapper<size_type> input(col_data.begin(), col_data.end(), col_valid.begin());

  EXPECT_THROW(cudf::experimental::rolling_window(input, -2,  2,  2, rolling_operator::SUM),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::rolling_window(input,  2, -2,  2, rolling_operator::SUM),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::rolling_window(input,  2,  2, -2, rolling_operator::SUM),
               cudf::logic_error);
}

// window array size mismatch
TEST_F(RollingErrorTest, WindowArraySizeMismatch)
{
  const std::vector<size_type> col_data = {0, 1, 2, 0, 4};
  const std::vector<bool>      col_valid = {1, 1, 1, 0, 1};
  fixed_width_column_wrapper<size_type> input(col_data.begin(), col_data.end(), col_valid.begin());

  std::vector<size_type> five({ 2, 1, 2, 1, 4 });
  std::vector<size_type> four({ 1, 2, 3, 4 });
  fixed_width_column_wrapper<size_type> five_elements(five.begin(), five.end());
  fixed_width_column_wrapper<size_type> four_elements(four.begin(), four.end());

  // this runs ok
  EXPECT_NO_THROW(cudf::experimental::rolling_window(input, five_elements, five_elements, 1,
                                                     rolling_operator::SUM));

  // mismatch for the window array
  EXPECT_THROW(cudf::experimental::rolling_window(input, four_elements, five_elements, 1,
                                                  rolling_operator::SUM),
               cudf::logic_error);

  // mismatch for the forward window array
  EXPECT_THROW(cudf::experimental::rolling_window(input, five_elements, four_elements, 1,
                                                  rolling_operator::SUM),
               cudf::logic_error);
}


TEST_F(RollingErrorTest, EmptyInput) {
  cudf::test::fixed_width_column_wrapper<int32_t> empty_col{};
  std::unique_ptr<cudf::column> output;
  EXPECT_NO_THROW(output = cudf::experimental::rolling_window(empty_col, 2, 0, 2,
                                                              rolling_operator::SUM));
  EXPECT_EQ(output->size(), 0);

  fixed_width_column_wrapper<int32_t> preceding_window{};
  fixed_width_column_wrapper<int32_t> following_window{};
  EXPECT_NO_THROW(output = cudf::experimental::rolling_window(empty_col, preceding_window,
                                                             following_window, 2,
                                                             rolling_operator::SUM));
  EXPECT_EQ(output->size(), 0);

  fixed_width_column_wrapper<int32_t> nonempty_col{{1, 2, 3}};
  EXPECT_NO_THROW(output = cudf::experimental::rolling_window(nonempty_col, preceding_window,
                                                              following_window, 2,
                                                              rolling_operator::SUM));
  EXPECT_EQ(output->size(), 0);
}

TEST_F(RollingErrorTest, SizeMismatch) {
  fixed_width_column_wrapper<int32_t> nonempty_col{{1, 2, 3}};
  std::unique_ptr<cudf::column> output;
  
  {
    fixed_width_column_wrapper<int32_t> preceding_window{{1, 1}}; // wrong size
    fixed_width_column_wrapper<int32_t> following_window{{1, 1, 1}};
    EXPECT_THROW(output = cudf::experimental::rolling_window(nonempty_col, preceding_window,
                                                             following_window,
                                                             2, rolling_operator::SUM),
                 cudf::logic_error);
  }
  {
    fixed_width_column_wrapper<int32_t> preceding_window{{1, 1, 1}};
    fixed_width_column_wrapper<int32_t> following_window{{1, 2}}; // wrong size
    EXPECT_THROW(output = cudf::experimental::rolling_window(nonempty_col, preceding_window,
                                                             following_window,
                                                             2, rolling_operator::SUM),
                 cudf::logic_error);
  }
}

TEST_F(RollingErrorTest, WindowWrongDtype) {
  fixed_width_column_wrapper<int32_t> nonempty_col{{1, 2, 3}};
  std::unique_ptr<cudf::column> output;
  
  fixed_width_column_wrapper<float> preceding_window{{1.0f, 1.0f, 1.0f}}; 
  fixed_width_column_wrapper<float> following_window{{1.0f, 1.0f, 1.0f}};
  EXPECT_THROW(output = cudf::experimental::rolling_window(nonempty_col, preceding_window,
                                                            following_window,
                                                            2, rolling_operator::SUM),
              cudf::logic_error);
}

// incorrect type/aggregation combo: sum of timestamps
TEST_F(RollingErrorTest, SumTimestampNotSupported)
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

  EXPECT_THROW(cudf::experimental::rolling_window(input_D, 2, 2, 0, rolling_operator::SUM),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::rolling_window(input_s, 2, 2, 0, rolling_operator::SUM),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::rolling_window(input_ms, 2, 2, 0, rolling_operator::SUM),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::rolling_window(input_us, 2, 2, 0, rolling_operator::SUM),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::rolling_window(input_ns, 2, 2, 0, rolling_operator::SUM),
               cudf::logic_error);
}

TYPED_TEST_CASE(RollingTest, cudf::test::FixedWidthTypes);

// simple example from Pandas docs
TYPED_TEST(RollingTest, SimpleStatic)
{
  // https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
  const std::vector<TypeParam> col_data = {0, 1, 2, 0, 4};
  const std::vector<bool>      col_mask = {1, 1, 1, 0, 1};

  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<size_type> window{2};

  // static sizes
  this->run_test_col_agg(input, window, window, 1);
}

// simple example from Pandas docs:
TYPED_TEST(RollingTest, SimpleDynamic)
{
  // https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
  const std::vector<TypeParam> col_data = {0, 1, 2, 0, 4};
  const std::vector<bool>      col_mask = {1, 1, 1, 0, 1};

  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<size_type> preceding_window({ 1, 2, 3, 4, 2 });
  std::vector<size_type> following_window({ 2, 1, 2, 1, 2 });

  // dynamic sizes
  this->run_test_col_agg(input, preceding_window, following_window, 1);
}

// this is a special test to check the volatile count variable issue (see rolling.cu for detail)
TYPED_TEST(RollingTest, VolatileCount)
{
  const std::vector<TypeParam> col_data = { 8, 70, 45, 20, 59, 80 };
  const std::vector<bool>      col_mask = { 1, 1, 0, 0, 1, 0 };

  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<size_type> preceding_window({ 5, 9, 4, 8, 3, 3 });
  std::vector<size_type> following_window({ 1, 1, 9, 2, 8, 9 });
  
  // dynamic sizes
  this->run_test_col_agg(input, preceding_window, following_window, 1);
}

// all rows are invalid
TYPED_TEST(RollingTest, AllInvalid)
{
  size_type num_rows = 1000;

  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool>      col_mask(num_rows, 0);

  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<size_type> window({100});
  size_type periods = 100;

  this->run_test_col_agg(input, window, window, periods);
}

// window = following_window = 0
TYPED_TEST(RollingTest, ZeroWindow)
{
  size_type num_rows = 1000;

  std::vector<TypeParam> col_data(num_rows, 1);
  std::vector<bool>      col_mask(num_rows, 1);
  
  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<size_type> window({0});
  size_type periods = num_rows;

  this->run_test_col_agg(input, window, window, periods);
}

// min_periods = 0
TYPED_TEST(RollingTest, ZeroPeriods)
{
  size_type num_rows = 1000;

  std::vector<TypeParam> col_data(num_rows, 1);
  std::vector<bool>      col_mask(num_rows, 1);

  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());

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

  std::vector<TypeParam> col_data(num_rows, 1);
  std::vector<bool>      col_mask(num_rows, 1);
  
  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());

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
  size_type num_rows = 50000;
  size_type max_window_size = 50;

  // random input
  std::vector<TypeParam> col_data(num_rows);
  cudf::test::UniformRandomGenerator<TypeParam> rng;
  std::generate(col_data.begin(), col_data.end(), [&rng]() { return rng.generate(); });
  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end());

  // random parameters
  cudf::test::UniformRandomGenerator<size_type> window_rng(0, max_window_size);
  auto generator = [&](){ return window_rng.generate(); };

  std::vector<size_type> preceding_window(num_rows);
  std::vector<size_type> following_window(num_rows);

  std::generate(preceding_window.begin(), preceding_window.end(), generator);
  std::generate(following_window.begin(), following_window.end(), generator);

  this->run_test_col_agg(input, preceding_window, following_window, max_window_size);
}

// random input data, dynamic parameters, with nulls
TYPED_TEST(RollingTest, RandomDynamicWithInvalid)
{
  size_type num_rows = 50000;
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
  auto generator = [&](){ return window_rng.generate(); };

  std::vector<size_type> preceding_window(num_rows);
  std::vector<size_type> following_window(num_rows);

  std::generate(preceding_window.begin(), preceding_window.end(), generator);
  std::generate(following_window.begin(), following_window.end(), generator);

  this->run_test_col_agg(input, preceding_window, following_window, max_window_size);
}

// ------------- non-fixed-width types --------------------

/*using RollingTestSeconds = RollingTest<cudf::timestamp_s>;

TEST_F(RollingTestSeconds, Foo)
{
  std::vector<cudf::timestamp_s> h_timestamps{ 131246625 , 1563399277, 1553085296, 1582934400 };
  //  std::vector<const char*> h_expected{ "1974-02-28T01:23:45Z", "2019-07-17T21:34:37Z", nullptr, "2019-03-20T12:34:56Z", "2020-02-29T00:00:00Z" };

  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s> input( h_timestamps.begin(), h_timestamps.end());
        //thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));

  auto results = cudf::strings::from_timestamps(input);
  cudf::test::print(*results);
   
  std::vector<size_type> window{1};

  std::cout << "MIN\n";
  //EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::MIN));
  std::cout << "MAX\n";
  //EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::MAX));
  std::cout << "COUNT\n";
  //EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::COUNT));
  std::cout << "MEAN\n";
  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::MEAN));
}*/

using RollingTestStrings = RollingTest<cudf::string_view>;

TEST_F(RollingTestStrings, StringsUnsupportedOperators)
{
  cudf::test::strings_column_wrapper input{{"This", "is", "not", "a", "string", "type"},
                                           {1, 1, 1, 0, 1, 0}};
  
  std::vector<size_type> window{1};

  EXPECT_THROW(cudf::experimental::rolling_window(input, 2, 2, 0, rolling_operator::SUM),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::rolling_window(input, 2, 2, 0, rolling_operator::MEAN),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::rolling_window(input, 2, 2, 0, rolling_operator::NUMBA_UDF),
               cudf::logic_error);
  EXPECT_THROW(cudf::experimental::rolling_window(input, 2, 2, 0, rolling_operator::CUDA_UDF),
               cudf::logic_error);
}

/*TEST_F(RollingTestStrings, SimpleStatic)
{
  cudf::test::strings_column_wrapper input{{"This", "is", "not", "a", "string", "type"},
                                           {1, 1, 1, 0, 1, 0}};

  std::vector<size_type> window{1};

  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::MIN));
  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::MAX));
  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::COUNT));
}*/




// class RollingTestNumba : public cudf::test::BaseFixture {};

// TEST_F(RollingTestNumba, NumbaGeneric)
// {

//   const char ptx[] =
//   R"***(
//   //
//   // Generated by NVIDIA NVVM Compiler
//   //
//   // Compiler Build ID: CL-24817639
//   // Cuda compilation tools, release 10.0, V10.0.130
//   // Based on LLVM 3.4svn
//   //

//   .version 6.3
//   .target sm_70
//   .address_size 64

//   // .globl	_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE
//   .common .global .align 8 .u64 _ZN08NumbaEnv8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE;

//   .visible .func  (.param .b32 func_retval0) _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE(
//   .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_0,
//   .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_1,
//   .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_2,
//   .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_3,
//   .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_4,
//   .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_5,
//   .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_6,
//   .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_7
//   )
//   {
//   .reg .pred 	%p<3>;
//   .reg .b32 	%r<6>;
//   .reg .b64 	%rd<18>;


//   ld.param.u64 	%rd6, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_0];
//   ld.param.u64 	%rd7, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_5];
//   ld.param.u64 	%rd8, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_6];
//   ld.param.u64 	%rd9, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_7];
//   mov.u64 	%rd15, 0;
//   mov.u64 	%rd16, %rd15;

//   BB0_1:
//   mov.u64 	%rd2, %rd16;
//   mov.u32 	%r5, 0;
//   setp.ge.s64	%p1, %rd15, %rd8;
//   mov.u64 	%rd17, %rd15;
//   @%p1 bra 	BB0_3;

//   mul.lo.s64 	%rd12, %rd15, %rd9;
//   add.s64 	%rd13, %rd12, %rd7;
//   ld.u32 	%r5, [%rd13];
//   add.s64 	%rd17, %rd15, 1;

//   BB0_3:
//   cvt.s64.s32	%rd14, %r5;
//   add.s64 	%rd16, %rd14, %rd2;
//   setp.lt.s64	%p2, %rd15, %rd8;
//   mov.u64 	%rd15, %rd17;
//   @%p2 bra 	BB0_1;

//   st.u64 	[%rd6], %rd2;
//   mov.u32 	%r4, 0;
//   st.param.b32	[func_retval0+0], %r4;
//   ret;
//   }
//   )***";

//   size_type size = 12;

//   fixed_width_column_wrapper<int32_t> input(thrust::make_counting_iterator(0),
//                                             thrust::make_counting_iterator(size),
//                                             thrust::make_constant_iterator(true));

//   std::unique_ptr<cudf::column> output;

//   EXPECT_NO_THROW(output = cudf::experimental::rolling_window(input, 2, 4, 2, ptx, 
//                                                               rolling_operator::NUMBA_UDF,
//                                                               cudf::data_type{cudf::INT64}));

//   auto start = cudf::test::make_counting_transform_iterator(0,
//     [] __device__(size_type row) { return row * 4 + 2; });

//   auto valid = cudf::test::make_counting_transform_iterator(0, 
//     [size] __device__ (size_type row) { return (row != 0 && row != size - 2 && row != size - 1); });

//   fixed_width_column_wrapper<int64_t> expected{start, start+size, valid};

//   cudf::test::expect_columns_equal(*output, expected);
// }

