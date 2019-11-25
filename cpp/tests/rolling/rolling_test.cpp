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

#include <cudf/strings/convert/convert_datetime.hpp>

#include <vector>

using cudf::test::fixed_width_column_wrapper;
using rolling_operator = cudf::experimental::rolling_operator;
using cudf::size_type;
using cudf::bitmask_type;

template <typename T>
class RollingTest : public cudf::test::BaseFixture {
protected:
  // input as column_wrapper
  void run_test_col(fixed_width_column_wrapper<T> const& input,
                    const std::vector<size_type> &window,
                    const std::vector<size_type> &forward_window,
                    size_type min_periods,
                    rolling_operator op)
  {
    std::unique_ptr<cudf::column> output;

    // wrap windows
    if (window.size() > 1) {
      fixed_width_column_wrapper<size_type> window_wrapper(window.begin(), window.end());
      fixed_width_column_wrapper<size_type> forward_window_wrapper(forward_window.begin(),
                                                                   forward_window.end());

      EXPECT_NO_THROW(output = cudf::experimental::rolling_window(input, window_wrapper,
                                                                  forward_window_wrapper,
                                                                  min_periods, op));
    }
    else {
      EXPECT_NO_THROW(output = cudf::experimental::rolling_window(input, window[0],
                                                                  forward_window[0],
                                                                  min_periods, op));
    }

    auto reference = create_reference_output(op, input, window, forward_window, min_periods);

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

    cudf::test::expect_columns_equal(*output, reference);
  }

  // helper function to test all aggregators
  void run_test_col_agg(fixed_width_column_wrapper<T> const& input,
                        const std::vector<size_type> &window,
                        const std::vector<size_type> &forward_window,
                        size_type min_periods)
  {
    // test all supported aggregators
    run_test_col(input, window, forward_window, min_periods, rolling_operator::SUM);
    run_test_col(input, window, forward_window, min_periods, rolling_operator::MIN);
    run_test_col(input, window, forward_window, min_periods, rolling_operator::MAX);
    run_test_col(input, window, forward_window, min_periods, rolling_operator::COUNT);
    run_test_col(input, window, forward_window, min_periods, rolling_operator::MEAN);
    
    // this aggregation function is not supported yet - expected to throw an exception
    //EXPECT_THROW(run_test_col(input, window, forward_window, min_periods, rolling_operator::COUNT_DISTINCT), cudf::logic_error);
  }

  private:

  // use SFINAE to only instantiate for supported combinations
  template<class agg_op, bool is_mean,
           std::enable_if_t<cudf::detail::is_supported<T, agg_op, is_mean>()>* = nullptr>
  fixed_width_column_wrapper<T> create_reference_output(cudf::column_view const& input,
                                                        std::vector<size_type> const& window_col,
                                                        std::vector<size_type> const& forward_window_col,
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
      size_type count = 0;
      // load sizes
      min_periods = std::max(min_periods, 1); // at least one observation is required

      // compute bounds
      auto window = window_col[i%window_col.size()];
      auto forward_window = forward_window_col[i%forward_window_col.size()];
      size_type start_index = std::max((size_type)0, i - window + 1);
      size_type end_index = std::min(num_rows, i + forward_window + 1); // exclusive
      
      // aggregate
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

    return fixed_width_column_wrapper<T>(ref_data.begin(), ref_data.end(), ref_valid.begin());
  }

  template<class agg_op, bool is_mean,
           std::enable_if_t<!cudf::detail::is_supported<T, agg_op, is_mean>()>* = nullptr>
  fixed_width_column_wrapper<T> create_reference_output(cudf::column_view const& input,
                                                        std::vector<size_type> const& window_col,
                                                        std::vector<size_type> const& forward_window_col,
                                                        size_type min_periods)
  {
    CUDF_FAIL("Unsupported combination of type and aggregation");
  }

  fixed_width_column_wrapper<T> create_reference_output(rolling_operator op,
                                                        cudf::column_view const& input,
                                                        std::vector<size_type> const& window,
                                                        std::vector<size_type> const& forward_window,
                                                        size_type min_periods)
  {
    // unroll aggregation types
    switch(op) {
    case rolling_operator::SUM:
      return create_reference_output<cudf::DeviceSum, false>(input, window, forward_window, min_periods);
    case rolling_operator::MIN:
      return create_reference_output<cudf::DeviceMin, false>(input, window, forward_window, min_periods);
    case rolling_operator::MAX:
      return create_reference_output<cudf::DeviceMax, false>(input, window, forward_window, min_periods);
    case rolling_operator::COUNT:
      return create_reference_output<cudf::DeviceCount, false>(input, window, forward_window, min_periods);
    case rolling_operator::MEAN:
      return create_reference_output<cudf::DeviceSum, true>(input, window, forward_window, min_periods);
    default:
      return fixed_width_column_wrapper<T>({});
    }
  }
};

TYPED_TEST_CASE(RollingTest, cudf::test::NumericTypes);

TYPED_TEST(RollingTest, EmptyInput) {
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> empty_col{};
  std::unique_ptr<cudf::column> output;
  EXPECT_NO_THROW(output = cudf::experimental::rolling_window(empty_col, 2, 0, 2,
                                                              rolling_operator::SUM));
  EXPECT_EQ(output->size(), 0);

  fixed_width_column_wrapper<int32_t> window{};
  fixed_width_column_wrapper<int32_t> forward_window{};
  EXPECT_NO_THROW(output = cudf::experimental::rolling_window(empty_col, window, forward_window,
                                                              2, rolling_operator::SUM));
  EXPECT_EQ(output->size(), 0);

  fixed_width_column_wrapper<int32_t> nonempty_col{{1, 2, 3}};
  EXPECT_NO_THROW(output = cudf::experimental::rolling_window(nonempty_col, window, forward_window,
                                                              2, rolling_operator::SUM));
  EXPECT_EQ(output->size(), 0);
}

TYPED_TEST(RollingTest, SizeMismatch) {
  using T = TypeParam;

  fixed_width_column_wrapper<int32_t> nonempty_col{{1, 2, 3}};
  std::unique_ptr<cudf::column> output;
  
  {
    fixed_width_column_wrapper<int32_t> window{{1, 1}}; // wrong size
    fixed_width_column_wrapper<int32_t> forward_window{{1, 1, 1}};
    EXPECT_THROW(output = cudf::experimental::rolling_window(nonempty_col, window,
                                                             forward_window,
                                                             2, rolling_operator::SUM),
                 cudf::logic_error);
  }
  {
    fixed_width_column_wrapper<int32_t> window{{1, 1, 1}};
    fixed_width_column_wrapper<int32_t> forward_window{{1, 2}}; // wrong size
    EXPECT_THROW(output = cudf::experimental::rolling_window(nonempty_col, window,
                                                             forward_window,
                                                             2, rolling_operator::SUM),
                 cudf::logic_error);
  }
}

TYPED_TEST(RollingTest, WindowWrongDtype) {
  using T = TypeParam;

  fixed_width_column_wrapper<int32_t> nonempty_col{{1, 2, 3}};
  std::unique_ptr<cudf::column> output;
  
  {
    fixed_width_column_wrapper<float> window{{1.0f, 1.0f, 1.0f}}; 
    fixed_width_column_wrapper<float> forward_window{{1.0f, 1.0f, 1.0f}};
    EXPECT_THROW(output = cudf::experimental::rolling_window(nonempty_col, window,
                                                             forward_window,
                                                             2, rolling_operator::SUM),
                 cudf::logic_error);
  }
}

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
  std::vector<size_type> window({ 1, 2, 3, 4, 2 });
  std::vector<size_type> forward_window({ 2, 1, 2, 1, 2 });

  // dynamic sizes
  this->run_test_col_agg(input, window, forward_window, 1);
}

// this is a special test to check the volatile count variable issue (see rolling.cu for detail)
TYPED_TEST(RollingTest, VolatileCount)
{
  const std::vector<TypeParam> col_data = { 8, 70, 45, 20, 59, 80 };
  const std::vector<bool>      col_mask = { 1, 1, 0, 0, 1, 0 };

  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<size_type> window({ 5, 9, 4, 8, 3, 3 });
  std::vector<size_type> forward_window({ 1, 1, 9, 2, 8, 9 });
  
  // dynamic sizes
  this->run_test_col_agg(input, window, forward_window, 1);
}

// all rows are invalid
TYPED_TEST(RollingTest, AllInvalid)
{
  size_type num_rows = 1000;

  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool>      col_mask(num_rows);
  std::fill(col_mask.begin(), col_mask.end(), 0);

  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<size_type> window({100});
  size_type periods = 100;

  this->run_test_col_agg(input, window, window, periods);
}

// window = forward_window = 0
TYPED_TEST(RollingTest, ZeroWindow)
{
  size_type num_rows = 1000;

  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool>      col_mask(num_rows);
  std::fill(col_data.begin(), col_data.end(), 1);
  std::fill(col_mask.begin(), col_mask.end(), 1);

  fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<size_type> window({0});
  size_type periods = num_rows;

  this->run_test_col_agg(input, window, window, periods);
}

// min_periods = 0
TYPED_TEST(RollingTest, ZeroPeriods)
{
  size_type num_rows = 1000;

  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool>      col_mask(num_rows);
  std::fill(col_data.begin(), col_data.end(), 1);
  std::fill(col_mask.begin(), col_mask.end(), 1);
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

  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool>      col_mask(num_rows);
  std::fill(col_data.begin(), col_data.end(), 1);
  std::fill(col_mask.begin(), col_mask.end(), 1);
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
  cudf::test::UniformRandomGenerator<TypeParam> rbg;
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

  std::vector<size_type> window(num_rows);
  std::vector<size_type> forward_window(num_rows);

  std::generate(window.begin(), window.end(), generator);
  std::generate(forward_window.begin(), forward_window.end(), generator);

  this->run_test_col_agg(input, window, forward_window, max_window_size);
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

  std::vector<size_type> window(num_rows);
  std::vector<size_type> forward_window(num_rows);

  std::generate(window.begin(), window.end(), generator);
  std::generate(forward_window.begin(), forward_window.end(), generator);

  this->run_test_col_agg(input, window, forward_window, max_window_size);
}

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

// ------------- non-arithmetic types --------------------

template<typename T>
using RollingTestTimestamp = RollingTest<T>;

TYPED_TEST_CASE(RollingTestTimestamp, cudf::test::TimestampTypes);

// incorrect type/aggregation combo: sum / mean for non-arithmetic types
TYPED_TEST(RollingTestTimestamp, SumTimestampNotSupported)
{
  constexpr size_type size{10};
  fixed_width_column_wrapper<TypeParam> input(thrust::make_counting_iterator(0),
                                              thrust::make_counting_iterator(size));

  EXPECT_THROW(cudf::experimental::rolling_window(input, 2, 2, 0, rolling_operator::SUM),
               cudf::logic_error);
}

// min/max/count should work for non-arithmetic types
TYPED_TEST(RollingTestTimestamp, TimestampNoNulls)
{
  constexpr size_type size{1000};
  fixed_width_column_wrapper<TypeParam> input(thrust::make_counting_iterator(0),
                                              thrust::make_counting_iterator(size));
  std::vector<size_type> window{2};

  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::MIN));
  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::MAX));

  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::COUNT));
  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::MEAN));
}

using RollingTestSeconds = RollingTest<cudf::timestamp_s>;

/*TEST_F(RollingTestSeconds, Foo)
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


// class RollingTestNumba : public GdfTest {};
// TEST_F(RollingTestNumba, NumbaGeneric)
// {
 
// const char ptx[] =
// R"***(
// //
// // Generated by NVIDIA NVVM Compiler
// //
// // Compiler Build ID: CL-24817639
// // Cuda compilation tools, release 10.0, V10.0.130
// // Based on LLVM 3.4svn
// //

// .version 6.3
// .target sm_70
// .address_size 64

// 	// .globl	_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE
// .common .global .align 8 .u64 _ZN08NumbaEnv8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE;

// .visible .func  (.param .b32 func_retval0) _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE(
// 	.param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_0,
// 	.param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_1,
// 	.param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_2,
// 	.param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_3,
// 	.param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_4,
// 	.param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_5,
// 	.param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_6,
// 	.param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_7
// )
// {
// 	.reg .pred 	%p<3>;
// 	.reg .b32 	%r<6>;
// 	.reg .b64 	%rd<18>;


// 	ld.param.u64 	%rd6, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_0];
// 	ld.param.u64 	%rd7, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_5];
// 	ld.param.u64 	%rd8, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_6];
// 	ld.param.u64 	%rd9, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_7];
// 	mov.u64 	%rd15, 0;
// 	mov.u64 	%rd16, %rd15;

// BB0_1:
// 	mov.u64 	%rd2, %rd16;
// 	mov.u32 	%r5, 0;
// 	setp.ge.s64	%p1, %rd15, %rd8;
// 	mov.u64 	%rd17, %rd15;
// 	@%p1 bra 	BB0_3;

// 	mul.lo.s64 	%rd12, %rd15, %rd9;
// 	add.s64 	%rd13, %rd12, %rd7;
// 	ld.u32 	%r5, [%rd13];
// 	add.s64 	%rd17, %rd15, 1;

// BB0_3:
// 	cvt.s64.s32	%rd14, %r5;
// 	add.s64 	%rd16, %rd14, %rd2;
// 	setp.lt.s64	%p2, %rd15, %rd8;
// 	mov.u64 	%rd15, %rd17;
// 	@%p2 bra 	BB0_1;

// 	st.u64 	[%rd6], %rd2;
// 	mov.u32 	%r4, 0;
// 	st.param.b32	[func_retval0+0], %r4;
// 	ret;
// }
// )***";
 
//   constexpr size_type size{12};
//   cudf::test::column_wrapper<int> input{
//     size,
//     [](size_type row) { return static_cast<int>(row); },
//     [](size_type row) { return true; }
//   };

//   gdf_column output;
  
//   EXPECT_NO_THROW( output = cudf::rolling_window(*input.get(), 2, 4, 2, ptx, GDF_NUMBA_GENERIC_AGG_OPS, GDF_INT64, nullptr, nullptr, nullptr) );

//   auto output_wrapper = cudf::test::column_wrapper<int64_t>(output);

//   cudf::test::column_wrapper<int64_t> expect{
//     size,
//     [](size_type row) { return static_cast<int>(row*4+2); },
//     [](size_type row) { return (row != 0 && row != size-2 && row != size-1); }
//   };

//   EXPECT_TRUE(output_wrapper == expect);

//   gdf_column_free(&output);

// }

