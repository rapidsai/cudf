/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

//Quantile (percentile) testing


#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <iostream>
#include <vector>
#include <string>

#include <cassert>
#include <cmath>

#include "gtest/gtest.h"

#include <cudf.h>
#include <cudf/functions.h>
#include <utilities/cudf_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/error_utils.hpp>
#include <quantiles/quantiles.h>
#include <bitmask/legacy_bitmask.hpp>

#include "tests/utilities/cudf_test_fixtures.h"



template<typename T, typename Allocator, template<typename, typename> class Vector>
__host__ __device__
void print_v(const Vector<T, Allocator>& v, std::ostream& os)
{
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(os,","));
  os<<"\n";
}


template<typename VType>
void f_quantile_tester(gdf_column* col_in, std::vector<VType>& v_out_exact, std::vector<std::vector<double>>& v_out_m, const gdf_error expected_error = GDF_SUCCESS)
{
  std::vector<std::string> methods{"lin_interp", "lower", "higher", "midpoint", "nearest"};
  size_t n_methods = methods.size();
  
  std::vector<double> qvals{0.0, 0.25, 0.33, 0.5, 1.0};
  size_t n_qs = qvals.size();
  
  assert( n_methods == methods.size() );
  gdf_context ctxt{0, static_cast<gdf_method>(0), 0, 1};
  
  for(size_t j = 0; j<n_qs; ++j)
    {
      VType res = 0;
      auto q = qvals[j];
      gdf_error ret = gdf_quantile_approx(col_in, q, &res, &ctxt);
      v_out_exact[j] = res;
      EXPECT_EQ( ret, expected_error) << "approx " << " returns unexpected failure\n";
      
      for(size_t i = 0;i<n_methods;++i)
        {
          double rt = 0;
          ret = gdf_quantile_exact(col_in, static_cast<gdf_quantile_method>(i), q, &rt, &ctxt);
          v_out_m[j][i] = rt;
          
          EXPECT_EQ( ret, expected_error) << "exact " << methods[i] << " returns unexpected failure\n";
        }
    }
}

struct gdf_quantile : public GdfTest {};

TEST_F(gdf_quantile, DoubleVector)
{
  using VType = double;
  std::vector<VType> v{6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7};
  rmm::device_vector<VType> d_in = v;
  rmm::device_vector<gdf_valid_type> d_valid(gdf_valid_allocation_size(d_in.size()));
  
  gdf_column col_in;
  col_in.size = d_in.size();
  col_in.data = d_in.data().get();
  col_in.valid = d_valid.data().get();
  col_in.null_count = 0;
  col_in.dtype = GDF_FLOAT64;

  size_t n_qs = 5;
  size_t n_methods = 5;

  std::vector<VType> v_baseline_approx{-1.01, 0.15, 0.15, 1.11, 6.8};
  std::vector<std::vector<double>> v_baseline_exact{
    {-1.01, -1.01, 0.15, -0.43, -1.01},
      {0.3125, 0.15, 0.8, 0.475, 0.15},
        {0.7805, 0.15, 0.8, 0.475, 0.8},
          {1.62, 1.11, 2.13, 1.62, 2.13},
            {6.8, 6.8, 6.8, 6.8, 6.8}};
  
  std::vector<VType> v_out_approx(n_qs, 0);
  std::vector<std::vector<double>> v_out_exact(n_qs, std::vector<double>(n_methods,0.0));

  f_quantile_tester<VType>(&col_in, v_out_approx, v_out_exact);

  for(size_t i=0; i<n_qs;++i)
    {
      double delta = std::abs(static_cast<double>(v_baseline_approx[i] - v_out_approx[i]));
      bool flag = delta < 1.0e-8;
      EXPECT_EQ( flag, true ) << i <<"-th quantile deviates from baseline by: " << delta;
    }

  for(size_t i=0; i<n_qs;++i)
    {
      for(size_t j=0; j < n_methods; ++j)
        {
          double delta = std::abs(static_cast<double>(v_baseline_exact[i][j] - v_out_exact[i][j]));
          bool flag = delta < 1.0e-8;
          EXPECT_EQ( flag, true ) << i <<"-th quantile on " << j << "-th deviates from baseline by: " << delta;
        }
    }
}

TEST_F(gdf_quantile, IntegerVector)
{
  using VType = int32_t;
  std::vector<VType> v{7, 0, 3, 4, 2, 1, -1, 1, 6};;
  rmm::device_vector<VType> d_in = v;
  rmm::device_vector<gdf_valid_type> d_valid(gdf_valid_allocation_size(d_in.size()));
  
  gdf_column col_in;
  col_in.size = d_in.size();
  col_in.data = d_in.data().get();
  col_in.valid = d_valid.data().get();
  col_in.null_count = 0;
  col_in.dtype = GDF_INT32;

  size_t n_qs = 5;
  size_t n_methods = 5;

  std::vector<VType> v_baseline_approx{-1, 0, 0, 1, 7};
  std::vector<std::vector<double>> v_baseline_exact{
    {-1, -1, 0, -0.5, -1},
      {0.25, 0, 1, 0.5, 0},
        {0.97, 0, 1, 0.5, 1},
          {1.5, 1, 2, 1.5, 2},
            {7, 7, 7, 7, 7}};
  
  std::vector<VType> v_out_approx(n_qs, 0);
  std::vector<std::vector<double>> v_out_exact(n_qs, std::vector<double>(n_methods,0.0));

  f_quantile_tester<VType>(&col_in, v_out_approx, v_out_exact);

  for(size_t i=0; i<n_qs;++i)
    {
      double delta = std::abs(static_cast<double>(v_baseline_approx[i] - v_out_approx[i]));
      bool flag = delta < 1.0e-8;
      EXPECT_EQ( flag, true ) << i <<"-th quantile deviates from baseline by: " << delta;
    }

  for(size_t i=0; i<n_qs;++i)
    {
      for(size_t j=0; j < n_methods; ++j)
        {
          double delta = std::abs(static_cast<double>(v_baseline_exact[i][j] - v_out_exact[i][j]));
          bool flag = delta < 1.0e-8;
          EXPECT_EQ( flag, true ) << i <<"-th quantile on " << j << "-th deviates from baseline by: " << delta;
        }
    }
}

TEST_F(gdf_quantile, ReportValidMaskError)
{
  using VType = int32_t;
  std::vector<VType> v{7, 0, 3, 4, 2, 1, -1, 1, 6};;
  rmm::device_vector<VType> d_in = v;
  rmm::device_vector<gdf_valid_type> d_valid(gdf_valid_allocation_size(d_in.size()));
  
  gdf_column col_in;
  col_in.size = d_in.size();
  col_in.data = d_in.data().get();
  col_in.valid = d_valid.data().get();
  col_in.null_count = 1;//Should cause the quantile calls to fail
  col_in.dtype = GDF_INT32;

  size_t n_qs = 5;
  size_t n_methods = 5;

  std::vector<VType> v_baseline_approx{-1, 0, 0, 1, 7};
  std::vector<std::vector<double>> v_baseline_exact{
      {-1, -1, 0, -0.5, -1},
      {0.25, 0, 1, 0.5, 0},
      {0.97, 0, 1, 0.5, 1},
      {1.5, 1, 2, 1.5, 2},
      {7, 7, 7, 7, 7}};
  
  std::vector<VType> v_out_approx(n_qs, 0);
  std::vector<std::vector<double>> v_out_exact(n_qs, std::vector<double>(n_methods,0.0));

  f_quantile_tester<VType>(&col_in, v_out_approx, v_out_exact, GDF_VALIDITY_UNSUPPORTED);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


