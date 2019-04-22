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
#include <utilities/column_wrapper.cuh>

#include <groupby/aggregation_operations.hpp>

#include <gtest/gtest.h>

#include <vector>
#include <random>
#include <algorithm>
#include <memory>

template <typename T>
class RollingTest : public GdfTest {

protected:
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
  }

  void create_gdf_input_buffers()
  {
    if (in_col_valid.size() > 0) {
      auto valid_generator = [&](size_t row, size_t col){
        return in_col_valid[row];
      };
      // note that if gdf_col_pointer contained something it will be freed since it's a unique pointer
      in_gdf_col = init_gdf_column(in_col, 0, valid_generator);
    }
    else
      in_gdf_col = create_gdf_column(in_col);
  }

  template<template <typename AggType> class agg_op, bool average>
  void create_reference_output(size_t window,
			       size_t min_periods,
			       size_t forward_window,
			       const std::vector<gdf_size_type> &window_col,
			       const std::vector<gdf_size_type> &min_periods_col,
			       const std::vector<gdf_size_type> &forward_window_col)
  {
    // compute the reference solution on the cpu
    size_t nrows = in_col.size();
    ref_data.resize(nrows);
    ref_data_valid.resize(nrows);
    agg_op<T> op;
    for(size_t i = 0; i < nrows; i++) {
      T val = agg_op<T>::IDENTITY;
      size_t count = 0;
      // load sizes
      if (window_col.size() > 0) window = window_col[i];
      if (min_periods_col.size() > 0) min_periods = min_periods_col[i];
      if (forward_window_col.size() > 0) forward_window = forward_window_col[i];
      // compute bounds
      size_t start_index = std::max((size_t)0, i - window + 1);
      size_t end_index = std::min(nrows, i + forward_window + 1);	// exclusive
      // aggregate
      for (size_t j = start_index; j < end_index; j++) {
        if (in_col_valid[j]) {
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
			       size_t window,
			       size_t min_periods,
			       size_t forward_window,
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
    cudaMemcpy(out_col.data(), static_cast<T*>(out_gdf_col->data), nrows * sizeof(T), cudaMemcpyDefault);
      
    // copy output valid mask to host
    gdf_size_type nmasks = gdf_valid_allocation_size(nrows);
    std::vector<gdf_valid_type> out_col_mask(nmasks);
    cudaMemcpy(out_col_mask.data(), static_cast<gdf_valid_type*>(out_gdf_col->valid), nmasks * sizeof(gdf_valid_type), cudaMemcpyDefault);
      
    // create column wrappers and compare
    cudf::test::column_wrapper<T> out(out_col, [&](gdf_index_type i) { return gdf_is_valid(out_col_mask.data(), i); } );
    cudf::test::column_wrapper<T> ref(ref_data, [&](gdf_index_type i) { return ref_data_valid[i]; } );
    ASSERT_TRUE(out == ref);

    // print the column
    out.print();
  }

  // static windows
  void run_test(size_t window, size_t min_periods, size_t forward_window, gdf_agg_op agg)
  {
    create_gdf_input_buffers();

    out_gdf_col = { cudf::rolling_window(in_gdf_col.get(), window, min_periods, forward_window, agg, NULL, NULL, NULL), deleter };

    create_reference_output(agg, window, min_periods, forward_window, std::vector<gdf_size_type>(), std::vector<gdf_size_type>(), std::vector<gdf_size_type>());

    compare_gdf_result();
  }

  // dynamic windows
  void run_test(std::vector<gdf_size_type> window, std::vector<gdf_size_type> min_periods, std::vector<gdf_size_type> forward_window, gdf_agg_op agg)
  {
    create_gdf_input_buffers();

    // copy sizes to the gpu
    // TODO: use RMM to allocate stuff?
    gdf_size_type *d_window;
    gdf_size_type *d_min_periods;
    gdf_size_type *d_forward_window;
    cudaMalloc(&d_window, window.size() * sizeof(gdf_size_type));
    cudaMalloc(&d_min_periods, min_periods.size() * sizeof(gdf_size_type));
    cudaMalloc(&d_forward_window, forward_window.size() * sizeof(gdf_size_type));
    cudaMemcpy(d_window, window.data(), window.size() * sizeof(gdf_size_type), cudaMemcpyDefault);
    cudaMemcpy(d_min_periods, min_periods.data(), min_periods.size() * sizeof(gdf_size_type), cudaMemcpyDefault);
    cudaMemcpy(d_forward_window, forward_window.data(), forward_window.size() * sizeof(gdf_size_type), cudaMemcpyDefault);

    out_gdf_col = { cudf::rolling_window(in_gdf_col.get(), 0, 0, 0, agg, d_window, d_min_periods, d_forward_window), deleter };

    create_reference_output(agg, 0, 0, 0, window, min_periods, forward_window);

    compare_gdf_result();
  
    // free GPU memory 
    cudaFree(d_window);
    cudaFree(d_min_periods);
    cudaFree(d_forward_window);
  }

  // input
  std::vector<T> in_col;
  std::vector<bool> in_col_valid;
  gdf_col_pointer in_gdf_col;

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

TYPED_TEST(RollingTest, Simple)
{
  // simple example from Pandas docs:
  //   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
  this->in_col 	     = {0, 1, 2, 0, 4};
  this->in_col_valid = {1, 1, 1, 0, 1};

  this->run_test(2, 2, 0, GDF_SUM);
  this->run_test(2, 2, 0, GDF_MIN);
  this->run_test(2, 2, 0, GDF_MAX);
  this->run_test(2, 2, 0, GDF_COUNT);
  this->run_test(2, 2, 0, GDF_AVG);

  // dynamic sizes
  this->run_test({ 1, 2, 3, 4, 2 }, { 2, 1, 2, 1, 2 }, { 1, 0, 1, 0, 1 }, GDF_SUM);
  
  /*
     correct output (int):
	@ 1 3 @ @ 
	@ 0 1 @ @ 
	@ 1 2 @ @ 
	@ 2 2 @ @ 
	@ 0 1 @ @ 
	1 1 3 3 @ 

     corrrect output (float):
	@ 1 3 @ @ 
	@ 0 1 @ @ 
	@ 1 2 @ @ 
	@ 2 2 @ @ 
	@ 0.5 1.5 @ @ 
	1 1 3 3 @ 
  */  
}
