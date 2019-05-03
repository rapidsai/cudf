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

#include <rolling.hpp>
#include <cudf.h>

#include <utilities/error_utils.hpp>
#include <utilities/cudf_utils.h>
#include "tests/utilities/column_wrapper.cuh"

#include <groupby/aggregation_operations.hpp>

#include <gtest/gtest.h>

#include <vector>
#include <random>
#include <algorithm>
#include <memory>

template <typename T>
class RollingTest : public GdfTest {

protected:
  // only for integral types
  template <typename U = T, typename std::enable_if_t<std::is_integral<U>::value, std::nullptr_t> = nullptr>
  void make_random_input(size_t nrows, bool add_nulls)
  {
    // generate random numbers for the vector
    std::mt19937 rng(1);
    auto generator = [&](){ 
      return rng() % std::numeric_limits<T>::max() + 1;
    };

    in_col.resize(nrows);
    std::generate(in_col.begin(), in_col.end(), generator);

    // generate a random validity mask
    if (add_nulls) {
      auto valid_generator = [&](){ 
        return static_cast<bool>((rng() % std::numeric_limits<T>::max()) % 2);
      };
     
      in_col_valid.resize(nrows);
      std::generate(in_col_valid.begin(), in_col_valid.end(), valid_generator);
    }
    else {
      in_col_valid.resize(0);
    }
  }
 
  // floats go here
  template <typename U = T, typename std::enable_if_t<!std::is_integral<U>::value, std::nullptr_t> = nullptr>
  void make_random_input(size_t nrows, bool add_nulls)
  {
    // generate random numbers for the vector
    std::mt19937 rng(1);
    auto generator = [&](){ 
      return rng() / 10000.0;
    };

    in_col.resize(nrows);
    std::generate(in_col.begin(), in_col.end(), generator);

    // generate a random validity mask
    if (add_nulls) {
      auto valid_generator = [&](){ 
        return static_cast<bool>(rng() % 2);
      };
     
      in_col_valid.resize(nrows);
      std::generate(in_col_valid.begin(), in_col_valid.end(), valid_generator);
    }
    else {
      in_col_valid.resize(0);
    }
  }

  void create_gdf_input_buffers()
  {
    if (in_col_valid.size() > 0)
      in_gdf_col = std::make_unique<cudf::test::column_wrapper<T>>(in_col, [&](gdf_index_type row) { return in_col_valid[row]; });
    else
      in_gdf_col = std::make_unique<cudf::test::column_wrapper<T>>(in_col);
  }

  template<template <typename AggType> class agg_op, bool average>
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
    agg_op<T> op;
    for(gdf_size_type i = 0; i < nrows; i++) {
      T val = agg_op<T>::IDENTITY;
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
        if (average)
          ref_data[i] = val / count;
        else
          ref_data[i] = val;
      }
    }
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
      create_reference_output<sum_op, false>(window, min_periods, forward_window, window_col, min_periods_col, forward_window_col);
      break;
    case GDF_MIN:
      create_reference_output<min_op, false>(window, min_periods, forward_window, window_col, min_periods_col, forward_window_col);
      break;
    case GDF_MAX:
      create_reference_output<max_op, false>(window, min_periods, forward_window, window_col, min_periods_col, forward_window_col);
      break;
    case GDF_COUNT:
      create_reference_output<count_op, false>(window, min_periods, forward_window, window_col, min_periods_col, forward_window_col);
      break;
    case GDF_AVG:
      create_reference_output<sum_op, true>(window, min_periods, forward_window, window_col, min_periods_col, forward_window_col);
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

  void run_test(gdf_size_type w, gdf_size_type m, gdf_size_type f,
		std::vector<gdf_size_type> &window, std::vector<gdf_size_type> &min_periods, std::vector<gdf_size_type> &forward_window,
		gdf_agg_op agg)
  {
    create_gdf_input_buffers();

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

    out_gdf_col = { cudf::rolling_window(*in_gdf_col->get(), w, m, f, agg, d_window, d_min_periods, d_forward_window), deleter };

    create_reference_output(agg, w, m, f, window, min_periods, forward_window);

    compare_gdf_result();
  
    // free GPU memory 
    if (d_window != NULL) EXPECT_EQ(RMM_FREE(d_window, 0), RMM_SUCCESS);
    if (d_min_periods != NULL) EXPECT_EQ(RMM_FREE(d_min_periods, 0), RMM_SUCCESS);
    if (d_forward_window != NULL) EXPECT_EQ(RMM_FREE(d_forward_window, 0), RMM_SUCCESS);
  }

  template<class... TArgs>
  void run_test_all_agg(TArgs... FArgs)
  {
    // test all supported aggregators
    run_test(FArgs..., GDF_SUM);
    run_test(FArgs..., GDF_MIN);
    run_test(FArgs..., GDF_MAX);
    run_test(FArgs..., GDF_COUNT);
    run_test(FArgs..., GDF_AVG);
    
    // this aggregation function is not supported yet - expected to throw an exception
    EXPECT_THROW(run_test(FArgs..., GDF_COUNT_DISTINCT), cudf::logic_error);
  }

  // input
  std::vector<T> in_col;
  std::vector<bool> in_col_valid;
  std::unique_ptr<cudf::test::column_wrapper<T>> in_gdf_col;

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

using TestTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, double>;

TYPED_TEST_CASE(RollingTest, TestTypes);

TYPED_TEST(RollingTest, SimpleStatic)
{
  // simple example from Pandas docs:
  //   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
  this->in_col 	     = {0, 1, 2, 0, 4};
  this->in_col_valid = {1, 1, 1, 0, 1};

  // static sizes
  this->run_test_all_agg(2, 2, 0,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>());
}

TYPED_TEST(RollingTest, SimpleDynamic)
{
  // simple example from Pandas docs:
  //   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
  this->in_col 	     = {0, 1, 2, 0, 4};
  this->in_col_valid = {1, 1, 1, 0, 1};

  // dynamic sizes
  this->run_test_all_agg(0, 0, 0,
			 std::vector<gdf_size_type>({ 1, 2, 3, 4, 2 }),
			 std::vector<gdf_size_type>({ 2, 1, 2, 1, 2 }),
			 std::vector<gdf_size_type>({ 1, 0, 1, 0, 1 }));
}

TYPED_TEST(RollingTest, VolatileCount)
{
  // this is a test to verify the count volatile fix
  this->in_col 	     = { 8, 70, 45, 20, 59, 80 };
  this->in_col_valid = { 1, 1, 0, 0, 1, 0 };

  // dynamic sizes
  this->run_test_all_agg(0, 0, 0,
			 std::vector<gdf_size_type>({ 5, 9, 4, 8, 3, 3 }),
			 std::vector<gdf_size_type>({ 1, 1, 9, 2, 8, 9 }),
			 std::vector<gdf_size_type>({ 6, 3, 3, 0, 2, 1 }));
}

// all rows are invalid, easy check
TYPED_TEST(RollingTest, AllInvalid)
{
  gdf_size_type num_rows = 1000;
  gdf_size_type window = 100;
  gdf_size_type periods = window;

  this->in_col.resize(num_rows);
  this->in_col_valid.resize(num_rows);

  std::fill(this->in_col_valid.begin(), this->in_col_valid.end(), 0);
  this->run_test_all_agg(window, periods, window,
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

  this->in_col.resize(num_rows);
  this->in_col_valid.resize(num_rows);

  std::fill(this->in_col_valid.begin(), this->in_col_valid.end(), 1);
  std::fill(this->in_col.begin(), this->in_col.end(), 1);
  this->run_test_all_agg(window, periods, window,
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

  this->in_col.resize(num_rows);
  this->in_col_valid.resize(num_rows);

  std::fill(this->in_col_valid.begin(), this->in_col_valid.end(), 1);
  std::fill(this->in_col.begin(), this->in_col.end(), 1);
  this->run_test_all_agg(window, periods, window,
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

  this->in_col.resize(num_rows);
  this->in_col_valid.resize(num_rows);

  std::fill(this->in_col_valid.begin(), this->in_col_valid.end(), 1);
  std::fill(this->in_col.begin(), this->in_col.end(), 1);
  this->run_test_all_agg(window, periods, window,
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
  this->make_random_input(num_rows, false);

  this->run_test_all_agg(window, min_periods, 0,
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
  this->make_random_input(num_rows, true);

  this->run_test_all_agg(window, min_periods, window,
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
  this->make_random_input(num_rows, false);

  // random parameters
  std::mt19937 rng(1);
  auto generator = [&](){ return rng() % max_window_size; };

  std::vector<gdf_size_type> window(num_rows);
  std::vector<gdf_size_type> min_periods(num_rows);
  std::vector<gdf_size_type> forward_window(num_rows);

  std::generate(window.begin(), window.end(), generator);
  std::generate(min_periods.begin(), min_periods.end(), generator);
  std::generate(forward_window.begin(), forward_window.end(), generator);

  this->run_test_all_agg(0, 0, 0,
			 window,
			 min_periods,
			 forward_window);
}

// random input data, dynamic parameters, with nulls
TYPED_TEST(RollingTest, RandomDynamicWithInvalid)
{
  gdf_size_type num_rows = 50000;
  gdf_size_type max_window_size = 50;

  // random input
  this->make_random_input(num_rows, true);

  // random parameters
  std::mt19937 rng(1);
  auto generator = [&](){ return rng() % max_window_size; };

  std::vector<gdf_size_type> window(this->in_col.size());
  std::vector<gdf_size_type> min_periods(this->in_col.size());
  std::vector<gdf_size_type> forward_window(this->in_col.size());

  std::generate(window.begin(), window.end(), generator);
  std::generate(min_periods.begin(), min_periods.end(), generator);
  std::generate(forward_window.begin(), forward_window.end(), generator);

  this->run_test_all_agg(0, 0, 0,
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

  // random input
  this->make_random_input(num_rows, true);

  // random parameters
  std::mt19937 rng(1);
  auto generator = [&](){ return rng() % max_window_size; };

  std::vector<gdf_size_type> window(num_rows);
  std::vector<gdf_size_type> forward_window(num_rows);

  std::generate(window.begin(), window.end(), generator);
  std::generate(forward_window.begin(), forward_window.end(), generator);

  this->run_test_all_agg(0, min_periods, 0,
			 window,
			 std::vector<gdf_size_type>(),
			 forward_window);
}

using RollingTestInt = RollingTest<int32_t>;

// negative sizes
TEST_F(RollingTestInt, NegativeSizes)
{
  this->in_col 	     = {0, 1, 2, 0, 4};
  this->in_col_valid = {1, 1, 1, 0, 1};
  EXPECT_THROW(this->run_test_all_agg(-2, 2, 2,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>()), cudf::logic_error);
  EXPECT_THROW(this->run_test_all_agg(2, -2, 2,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>()), cudf::logic_error);
  EXPECT_THROW(this->run_test_all_agg(2, 2, -2,
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>(),
			 std::vector<gdf_size_type>()), cudf::logic_error);
}

/* TODO: test the following failures
   - window array size < input col size
   - min periods array size < input col size
   - sum of non-arithmetic types
   - average for non-arithmetic types
 */

