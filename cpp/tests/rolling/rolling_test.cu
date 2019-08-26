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

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>

#include <cudf/rolling.hpp>
#include <src/rolling/rolling_detail.hpp>
#include <cudf/cudf.h>

#include <utilities/error_utils.hpp>
#include <utilities/cudf_utils.h>
#include <tests/utilities/column_wrapper.cuh>

#include <gtest/gtest.h>

#include <vector>
#include <random>
#include <algorithm>
#include <memory>

template <typename T>
class RollingTest : public GdfTest {

protected:
  // integral types
  template <typename U = T, typename std::enable_if_t<std::is_integral<U>::value, std::nullptr_t> = nullptr>
  const T random_value(std::mt19937 &rng)
  {
    return rng() % std::numeric_limits<T>::max() + 1;
  }
 
  // non-integral types (e.g. floating point)
  template <typename U = T, typename std::enable_if_t<!std::is_integral<U>::value, std::nullptr_t> = nullptr>
  const T random_value(std::mt19937 &rng)
  {
    return rng() / 10000.0;
  }

  // input as column_wrapper
  void run_test_col(const cudf::test::column_wrapper<T> &input,
		    gdf_size_type w, gdf_size_type m, gdf_size_type f,
                    const std::vector<gdf_size_type> &window, const std::vector<gdf_size_type> &min_periods, const std::vector<gdf_size_type> &forward_window,
                    gdf_agg_op agg)
  {
    // it's not possible to check sizes in the rolling window API since we pass raw pointers for window/periods
    // so we check here that the tests are setup correctly
    CUDF_EXPECTS(window.size() == 0 || window.size() == (size_t)input.size(), "Window array size != input column size");
    CUDF_EXPECTS(min_periods.size() == 0 || min_periods.size() == (size_t)input.size(), "Min periods array size != input column size");
    CUDF_EXPECTS(forward_window.size() == 0 || forward_window.size() == (size_t)input.size(), "Forward window array size != input column size");

    // copy the input to host
    std::vector<gdf_valid_type> valid;
    std::tie(in_col, valid) = input.to_host();
    in_col_valid.resize(in_col.size());
    for (size_t row = 0; row < in_col.size(); row++)
      in_col_valid[row] = gdf_is_valid(valid.data(), row);

    gdf_size_type *d_window = NULL;
    gdf_size_type *d_min_periods = NULL;
    gdf_size_type *d_forward_window = NULL;

    // copy sizes to the gpu
    if (window.size() > 0) {
      EXPECT_EQ(RMM_ALLOC(&d_window, window.size() * sizeof(gdf_size_type), 0), RMM_SUCCESS);
      CUDA_TRY(cudaMemcpy(d_window, window.data(), window.size() * sizeof(gdf_size_type), cudaMemcpyDefault));
    }
    if (min_periods.size() > 0) {
      EXPECT_EQ(RMM_ALLOC(&d_min_periods, min_periods.size() * sizeof(gdf_size_type), 0), RMM_SUCCESS);
      CUDA_TRY(cudaMemcpy(d_min_periods, min_periods.data(), min_periods.size() * sizeof(gdf_size_type), cudaMemcpyDefault));
    }
    if (forward_window.size() > 0) {
      EXPECT_EQ(RMM_ALLOC(&d_forward_window, forward_window.size() * sizeof(gdf_size_type), 0), RMM_SUCCESS);
      CUDA_TRY(cudaMemcpy(d_forward_window, forward_window.data(), forward_window.size() * sizeof(gdf_size_type), cudaMemcpyDefault));
    }

    out_gdf_col = { cudf::rolling_window(*input.get(), w, m, f, agg, d_window, d_min_periods, d_forward_window), deleter };

    create_reference_output(agg, w, m, f, window, min_periods, forward_window);

    compare_gdf_result();

    // free GPU memory 
    if (d_window != NULL) EXPECT_EQ(RMM_FREE(d_window, 0), RMM_SUCCESS);
    if (d_min_periods != NULL) EXPECT_EQ(RMM_FREE(d_min_periods, 0), RMM_SUCCESS);
    if (d_forward_window != NULL) EXPECT_EQ(RMM_FREE(d_forward_window, 0), RMM_SUCCESS);
  }

  // input as data and validity mask
  void run_test_col(const std::vector<T> &data,
		    const std::vector<bool> &mask,
		    gdf_size_type w, gdf_size_type m, gdf_size_type f,
                    const std::vector<gdf_size_type> &window, const std::vector<gdf_size_type> &min_periods, const std::vector<gdf_size_type> &forward_window,
                    gdf_agg_op agg)
  {
    CUDF_EXPECTS(data.size() == mask.size(), "Validity array size != input column size");
    cudf::test::column_wrapper<T> input{
	(gdf_size_type)data.size(),
	[&](gdf_index_type row) { return data[row]; },
	[&](gdf_index_type row) { return mask[row]; }
    };
    run_test_col(input, w, m, f, window, min_periods, forward_window, agg);
  }

  // helper function to test all aggregators
  template<class... TArgs>
  void run_test_col_agg(TArgs... FArgs)
  {
    // test all supported aggregators
    run_test_col(FArgs..., GDF_SUM);
    run_test_col(FArgs..., GDF_MIN);
    run_test_col(FArgs..., GDF_MAX);
    run_test_col(FArgs..., GDF_COUNT);
    run_test_col(FArgs..., GDF_AVG);
    
    // this aggregation function is not supported yet - expected to throw an exception
    EXPECT_THROW(run_test_col(FArgs..., GDF_COUNT_DISTINCT), cudf::logic_error);
  }

private:
  // use SFINAE to only instantiate for supported combinations
  template<class agg_op, bool average,
	   typename std::enable_if_t<cudf::detail::is_supported<T, agg_op>(), std::nullptr_t> = nullptr>
  void create_reference_output(gdf_size_type window,
			       gdf_size_type min_periods,
			       gdf_size_type forward_window,
			       const std::vector<gdf_size_type> &window_col,
			       const std::vector<gdf_size_type> &min_periods_col,
			       const std::vector<gdf_size_type> &forward_window_col)
  {
    // compute the reference solution on the cpu
    gdf_size_type nrows = in_col.size();
    ref_data.resize(nrows);
    ref_data_valid.resize(nrows);
    agg_op op;
    for(gdf_size_type i = 0; i < nrows; i++) {
      T val = agg_op::template identity<T>();
      gdf_size_type count = 0;
      // load sizes
      if (window_col.size() > 0) window = window_col[i];
      if (min_periods_col.size() > 0) min_periods = min_periods_col[i];
      min_periods = std::max(min_periods, 1);	// at least one observation is required
      if (forward_window_col.size() > 0) forward_window = forward_window_col[i];
      // compute bounds
      gdf_size_type start_index = std::max((gdf_size_type)0, i - window + 1);
      gdf_size_type end_index = std::min(nrows, i + forward_window + 1);	// exclusive
      // aggregate
      for (gdf_size_type j = start_index; j < end_index; j++) {
        if (in_col_valid.size() == 0 || in_col_valid[j]) {
          val = op(in_col[j], val);
          count++;
        }
      }
      ref_data_valid[i] = (count >= min_periods);
      if (ref_data_valid[i]) {
	cudf::detail::store_output_functor<T, average>{}(ref_data[i], val, count);
      }
    }
  }

  template<class agg_op, bool average,
	   typename std::enable_if_t<!cudf::detail::is_supported<T, agg_op>(), std::nullptr_t> = nullptr>
  void create_reference_output(gdf_size_type window,
			       gdf_size_type min_periods,
			       gdf_size_type forward_window,
			       const std::vector<gdf_size_type> &window_col,
			       const std::vector<gdf_size_type> &min_periods_col,
			       const std::vector<gdf_size_type> &forward_window_col)
  {
    CUDF_FAIL("Unsupported combination of type and aggregation");
  }

  void create_reference_output(gdf_agg_op agg,
			       gdf_size_type window,
			       gdf_size_type min_periods,
			       gdf_size_type forward_window,
			       const std::vector<gdf_size_type> &window_col,
			       const std::vector<gdf_size_type> &min_periods_col,
			       const std::vector<gdf_size_type> &forward_window_col)
  {
    // unroll aggregation types
    switch(agg) {
    case GDF_SUM:
      create_reference_output<cudf::DeviceSum, false>(window, min_periods, forward_window, window_col, min_periods_col, forward_window_col);
      break;
    case GDF_MIN:
      create_reference_output<cudf::DeviceMin, false>(window, min_periods, forward_window, window_col, min_periods_col, forward_window_col);
      break;
    case GDF_MAX:
      create_reference_output<cudf::DeviceMax, false>(window, min_periods, forward_window, window_col, min_periods_col, forward_window_col);
      break;
    case GDF_COUNT:
      create_reference_output<cudf::DeviceCount, false>(window, min_periods, forward_window, window_col, min_periods_col, forward_window_col);
      break;
    case GDF_AVG:
      create_reference_output<cudf::DeviceSum, true>(window, min_periods, forward_window, window_col, min_periods_col, forward_window_col);
      break;
    default:
      FAIL() << "aggregation type not supported";
    }
  }

  void compare_gdf_result()
  {
    // convert to column_wrapper to compare

    // copy output data to host
    gdf_size_type nrows = in_col.size();
    std::vector<T> out_col(nrows);
    CUDA_TRY(cudaMemcpy(out_col.data(), static_cast<T*>(out_gdf_col->data), nrows * sizeof(T), cudaMemcpyDefault));
      
    // copy output valid mask to host
    gdf_size_type nmasks = gdf_valid_allocation_size(nrows);
    std::vector<gdf_valid_type> out_col_mask(nmasks);
    CUDA_TRY(cudaMemcpy(out_col_mask.data(), static_cast<gdf_valid_type*>(out_gdf_col->valid), nmasks * sizeof(gdf_valid_type), cudaMemcpyDefault));
      
    // create column wrappers and compare
    cudf::test::column_wrapper<T> out(out_col, [&](gdf_index_type i) { return gdf_is_valid(out_col_mask.data(), i); } );
    cudf::test::column_wrapper<T> ref(ref_data, [&](gdf_index_type i) { return ref_data_valid[i]; } );

    // print the columns for debugging
    //out.print();
    //ref.print();

    ASSERT_TRUE(out == ref);
  }

  // input
  std::vector<T> in_col;
  std::vector<bool> in_col_valid;

  // reference
  std::vector<T> ref_data;
  std::vector<bool> ref_data_valid;

  // output
  gdf_col_pointer out_gdf_col;

  // column deleter
  const std::function<void(gdf_column*)> deleter = [](gdf_column* col) {
    col->size = 0;
    RMM_FREE(col->data, 0);
    RMM_FREE(col->valid, 0);
  };
};

// ------------- arithmetic types --------------------

using ArithmeticTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, double>;

TYPED_TEST_CASE(RollingTest, ArithmeticTypes);

TYPED_TEST(RollingTest, EmptyInput)
{
  cudf::test::column_wrapper<TypeParam> input(0);

  this->run_test_col(input,
		     2, 2, 2,
		     std::vector<gdf_size_type>(),
		     std::vector<gdf_size_type>(),
		     std::vector<gdf_size_type>(),
		     GDF_SUM);
}

// simple example from Pandas docs
TYPED_TEST(RollingTest, SimpleStatic)
{
  // https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
  const std::vector<TypeParam> col_data = {0, 1, 2, 0, 4};
  const std::vector<bool>      col_mask = {1, 1, 1, 0, 1};

  // static sizes
  this->run_test_col_agg(col_data, col_mask,
		         2, 2, 0,
		         std::vector<gdf_size_type>(),
		         std::vector<gdf_size_type>(),
		         std::vector<gdf_size_type>());
}

// simple example from Pandas docs:
TYPED_TEST(RollingTest, SimpleDynamic)
{
  // https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
  const std::vector<TypeParam> col_data = {0, 1, 2, 0, 4};
  const std::vector<bool>      col_mask = {1, 1, 1, 0, 1};

  // dynamic sizes
  this->run_test_col_agg(col_data, col_mask,
			 0, 0, 0,
			 std::vector<gdf_size_type>({ 1, 2, 3, 4, 2 }),
			 std::vector<gdf_size_type>({ 2, 1, 2, 1, 2 }),
			 std::vector<gdf_size_type>({ 1, 0, 1, 0, 1 }));
}

// this is a special test to check the volatile count variable issue (see rolling.cu for detail)
TYPED_TEST(RollingTest, VolatileCount)
{
  const std::vector<TypeParam> col_data = { 8, 70, 45, 20, 59, 80 };
  const std::vector<bool>      col_mask = { 1, 1, 0, 0, 1, 0 };

  // dynamic sizes
  this->run_test_col_agg(col_data, col_mask,
			 0, 0, 0,
			 std::vector<gdf_size_type>({ 5, 9, 4, 8, 3, 3 }),
			 std::vector<gdf_size_type>({ 1, 1, 9, 2, 8, 9 }),
			 std::vector<gdf_size_type>({ 6, 3, 3, 0, 2, 1 }));
}

// all rows are invalid
TYPED_TEST(RollingTest, AllInvalid)
{
  gdf_size_type num_rows = 1000;
  gdf_size_type window = 100;
  gdf_size_type periods = window;

  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool>      col_mask(num_rows);
  std::fill(col_mask.begin(), col_mask.end(), 0);

  this->run_test_col_agg(col_data, col_mask,
			 window, periods, window,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>());
}

// window = forward_window = 0
TYPED_TEST(RollingTest, ZeroWindow)
{
  gdf_size_type num_rows = 1000;
  gdf_size_type window = 0;
  gdf_size_type periods = num_rows;

  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool>      col_mask(num_rows);
  std::fill(col_data.begin(), col_data.end(), 1);
  std::fill(col_mask.begin(), col_mask.end(), 1);

  this->run_test_col_agg(col_data, col_mask,
			 window, periods, window,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>());
}

// min_periods = 0
TYPED_TEST(RollingTest, ZeroPeriods)
{
  gdf_size_type num_rows = 1000;
  gdf_size_type window = num_rows;
  gdf_size_type periods = 0;

  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool>      col_mask(num_rows);
  std::fill(col_data.begin(), col_data.end(), 1);
  std::fill(col_mask.begin(), col_mask.end(), 1);

  this->run_test_col_agg(col_data, col_mask,
			 window, periods, window,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>());
}

// window in one direction is not large enough to collect enough samples, 
//   but if using both directions we should get == min_periods,
// also tests out of boundary accesses
TYPED_TEST(RollingTest, BackwardForwardWindow)
{
  gdf_size_type num_rows = 1000;
  gdf_size_type window = num_rows;
  gdf_size_type periods = num_rows;

  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool>      col_mask(num_rows);
  std::fill(col_data.begin(), col_data.end(), 1);
  std::fill(col_mask.begin(), col_mask.end(), 1);

  this->run_test_col_agg(col_data, col_mask,
			 window, periods, window,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>());
}

// random input data, static parameters, no nulls
TYPED_TEST(RollingTest, RandomStaticAllValid)
{
  gdf_size_type num_rows = 10000;
  gdf_size_type window = 50;
  gdf_size_type min_periods = 50;

  // random input
  std::mt19937 rng(1);
  cudf::test::column_wrapper<TypeParam> input{
	num_rows,
	[&](gdf_index_type row) { return this->random_value(rng); }
  };

  this->run_test_col_agg(input,
			 window, min_periods, 0,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>());
}

// random input data, static parameters, with nulls
TYPED_TEST(RollingTest, RandomStaticWithInvalid)
{
  gdf_size_type num_rows = 10000;
  gdf_size_type window = 50;
  gdf_size_type min_periods = 25;

  // random input with nulls
  std::mt19937 rng(1);
  cudf::test::column_wrapper<TypeParam> input{
	num_rows,
	[&](gdf_index_type row) { return this->random_value(rng); },
	[&](gdf_index_type row) { return static_cast<bool>(rng() % 2); }
  };

  this->run_test_col_agg(input,
			 window, min_periods, window,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>());
}

// random input data, dynamic parameters, no nulls
TYPED_TEST(RollingTest, RandomDynamicAllValid)
{
  gdf_size_type num_rows = 50000;
  gdf_size_type max_window_size = 50;

  // random input
  std::mt19937 rng(1);
  cudf::test::column_wrapper<TypeParam> input{
	num_rows,
	[&](gdf_index_type row) { return this->random_value(rng); }
  };

  // random parameters
  auto generator = [&](){ return rng() % max_window_size; };

  std::vector<gdf_size_type> window(num_rows);
  std::vector<gdf_size_type> min_periods(num_rows);
  std::vector<gdf_size_type> forward_window(num_rows);

  std::generate(window.begin(), window.end(), generator);
  std::generate(min_periods.begin(), min_periods.end(), generator);
  std::generate(forward_window.begin(), forward_window.end(), generator);

  this->run_test_col_agg(input,
			 0, 0, 0,
			 window,
			 min_periods,
			 forward_window);
}

// random input data, dynamic parameters, with nulls
TYPED_TEST(RollingTest, RandomDynamicWithInvalid)
{
  gdf_size_type num_rows = 50000;
  gdf_size_type max_window_size = 50;

  // random input with nulls
  std::mt19937 rng(1);
  cudf::test::column_wrapper<TypeParam> input{
	num_rows,
	[&](gdf_index_type row) { return this->random_value(rng); },
	[&](gdf_index_type row) { return static_cast<bool>(rng() % 2); }
  };

  // random parameters
  auto generator = [&](){ return rng() % max_window_size; };

  std::vector<gdf_size_type> window(num_rows);
  std::vector<gdf_size_type> min_periods(num_rows);
  std::vector<gdf_size_type> forward_window(num_rows);

  std::generate(window.begin(), window.end(), generator);
  std::generate(min_periods.begin(), min_periods.end(), generator);
  std::generate(forward_window.begin(), forward_window.end(), generator);

  this->run_test_col_agg(input,
			 0, 0, 0,
			 window,
			 min_periods,
			 forward_window);
}

// mix of static and dynamic parameters
TYPED_TEST(RollingTest, RandomDynamicWindowStaticPeriods)
{
  gdf_size_type num_rows = 50000;
  gdf_size_type max_window_size = 50;
  gdf_size_type min_periods = 25;

  // random input with nulls
  std::mt19937 rng(1);
  cudf::test::column_wrapper<TypeParam> input{
	num_rows,
	[&](gdf_index_type row) { return this->random_value(rng); },
	[&](gdf_index_type row) { return static_cast<bool>(rng() % 2); }
  };

  // random parameters
  auto generator = [&](){ return rng() % max_window_size; };

  std::vector<gdf_size_type> window(num_rows);
  std::vector<gdf_size_type> forward_window(num_rows);

  std::generate(window.begin(), window.end(), generator);
  std::generate(forward_window.begin(), forward_window.end(), generator);

  this->run_test_col_agg(input,
			 0, min_periods, 0,
			 window,
			 std::vector<gdf_size_type>(),
			 forward_window);
}

// ------------- expected failures --------------------

// negative sizes
TYPED_TEST(RollingTest, NegativeSizes)
{
  const std::vector<TypeParam> col_data = {0, 1, 2, 0, 4};
  const std::vector<bool>      col_mask = {1, 1, 1, 0, 1};

  EXPECT_THROW(this->run_test_col_agg(col_data, col_mask,
			 -2, 2, 2,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>()), cudf::logic_error);
  EXPECT_THROW(this->run_test_col_agg(col_data, col_mask,
			 2, -2, 2,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>()), cudf::logic_error);
  EXPECT_THROW(this->run_test_col_agg(col_data, col_mask,
			 2, 2, -2,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>()), cudf::logic_error);
}

// validity size mismatch
TYPED_TEST(RollingTest, ValidSizeMismatch)
{
  const std::vector<TypeParam> col_data = {0, 1, 2, 0, 4};
  const std::vector<bool>      col_mask = {1, 1, 1, 0};
 
  // validity mask size mismatch
  EXPECT_THROW(this->run_test_col_agg(col_data, col_mask,
			 0, 0, 0,
			 std::vector<gdf_size_type>({ 1, 2, 3, 2, 3 }),
			 std::vector<gdf_size_type>({ 2, 1, 2, 1, 4 }),
			 std::vector<gdf_size_type>({ 1, 0, 1, 0, 4 })), cudf::logic_error);
}

// window array size mismatch
TYPED_TEST(RollingTest, WindowArraySizeMismatch)
{
  const std::vector<TypeParam> col_data = {0, 1, 2, 0, 4};
  const std::vector<bool>      col_mask = {1, 1, 1, 0, 1};

  // this runs ok
  this->run_test_col_agg(col_data, col_mask,
			 0, 0, 0,
			 std::vector<gdf_size_type>({ 1, 2, 3, 2, 3 }),
			 std::vector<gdf_size_type>({ 2, 1, 2, 1, 4 }),
			 std::vector<gdf_size_type>({ 1, 0, 1, 0, 4 }));

  // mismatch for the window array
  EXPECT_THROW(this->run_test_col_agg(col_data, col_mask,
			 0, 0, 0,
			 std::vector<gdf_size_type>({ 1, 2, 3, 2 }),
			 std::vector<gdf_size_type>({ 2, 1, 2, 1, 4 }),
			 std::vector<gdf_size_type>({ 1, 0, 1, 0, 4 })), cudf::logic_error);

  // mismatch for the periods array
  EXPECT_THROW(this->run_test_col_agg(col_data, col_mask,
			 0, 0, 0,
			 std::vector<gdf_size_type>({ 1, 2, 3, 4, 3 }),
			 std::vector<gdf_size_type>({ 1, 2, 3, 4 }),
			 std::vector<gdf_size_type>({ 2, 1, 2, 1, 4 })), cudf::logic_error);

  // mismatch for the forward window array
  EXPECT_THROW(this->run_test_col_agg(col_data, col_mask,
			 0, 0, 0,
			 std::vector<gdf_size_type>({ 1, 2, 3, 4, 3 }),
			 std::vector<gdf_size_type>({ 1, 2, 3, 4, 6 }),
			 std::vector<gdf_size_type>({ 2, 1, 2, 1 })), cudf::logic_error);
}

// ------------- non-arithmetic types --------------------

using NonArithmeticTypes = ::testing::Types<cudf::category, cudf::timestamp, cudf::date32,
                                  	    cudf::date64, cudf::bool8>;

template<typename T>
using RollingTestNonArithmetic = RollingTest<T>;

TYPED_TEST_CASE(RollingTestNonArithmetic, NonArithmeticTypes);

// incorrect type/aggregation combo: sum or avg for non-arithmetic types
TYPED_TEST(RollingTestNonArithmetic, SumAvgNonArithmetic)
{
  constexpr gdf_size_type size{1000};
  cudf::test::column_wrapper<TypeParam> input{
	size,
	[](gdf_index_type row) { return static_cast<TypeParam>(row); },
	[](gdf_index_type row) { return row % 2; }
  };
  EXPECT_THROW(this->run_test_col(
			 input,
			 2, 2, 0,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 GDF_SUM),
	       cudf::logic_error);
  EXPECT_THROW(this->run_test_col(
			 input,
			 2, 2, 0,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 GDF_AVG),
	       cudf::logic_error);
}

// min/max/count should work for non-arithmetic types
TYPED_TEST(RollingTestNonArithmetic, MinMaxCountNonArithmetic)
{
  constexpr gdf_size_type size{1000};
  cudf::test::column_wrapper<TypeParam> input{
	size,
	[](gdf_index_type row) { return static_cast<TypeParam>(row); },
	[](gdf_index_type row) { return row % 2; }
  };
  this->run_test_col(input,
		     2, 2, 0,
		     std::vector<gdf_size_type>(),
		     std::vector<gdf_size_type>(),
		     std::vector<gdf_size_type>(),
		     GDF_MIN);
  this->run_test_col(input,
		     2, 2, 0,
		     std::vector<gdf_size_type>(),
		     std::vector<gdf_size_type>(),
		     std::vector<gdf_size_type>(),
		     GDF_MAX);
  this->run_test_col(input,
		     2, 2, 0,
		     std::vector<gdf_size_type>(),
		     std::vector<gdf_size_type>(),
		     std::vector<gdf_size_type>(),
		     GDF_COUNT);
}

class RollingTestNumba : public GdfTest {};
TEST_F(RollingTestNumba, NumbaGeneric)
{
 
const char ptx[] =
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
)***";
 
  constexpr gdf_size_type size{12};
  cudf::test::column_wrapper<int> input{
    size,
    [](gdf_index_type row) { return static_cast<int>(row); },
    [](gdf_index_type row) { return true; }
  };

  gdf_column output;
  
  EXPECT_NO_THROW( output = cudf::rolling_window(*input.get(), 2, 4, 2, ptx, GDF_NUMBA_GENERIC_AGG_OPS, GDF_INT64, nullptr, nullptr, nullptr) );

  auto output_wrapper = cudf::test::column_wrapper<int64_t>(output);

  cudf::test::column_wrapper<int64_t> expect{
    size,
    [](gdf_index_type row) { return static_cast<int>(row*4+2); },
    [](gdf_index_type row) { return (row != 0 && row != size-2 && row != size-1); }
  };

  EXPECT_TRUE(output_wrapper == expect);

  gdf_column_free(&output);

}

