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


#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/scalar_wrapper.cuh>

#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>
#include <quantiles/quantiles_util.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/cudf.h>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#include <string>

#include <cassert>
#include <cmath>


template<typename VType>
void f_quantile_tester(
  gdf_column* col_in,                           ///< input column
  std::vector<VType>& v_appox,                  ///< expected result for quantile_approx
  std::vector<std::vector<double>>& v_exact,    ///< expected result for quantile_exact
  const gdf_error expected_error = GDF_SUCCESS) ///< expected returned state for quantiles
{
  cudf::test::scalar_wrapper<VType>  result_approx(VType{0});
  cudf::test::scalar_wrapper<double> result_exact(double{0});

  std::vector<std::string> methods{"lin_interp", "lower", "higher", "midpoint", "nearest"};
  size_t n_methods = methods.size();
  
  std::vector<double> qvals{0.0, 0.25, 0.33, 0.5, 1.0};
  size_t n_qs = qvals.size();
  
  assert( n_methods == methods.size() );
  gdf_context ctxt{0, static_cast<gdf_method>(0), 0, 1};
  
  for(size_t j = 0; j<n_qs; ++j)
    {
      auto q = qvals[j];
      gdf_error ret = cudf::quantile_approx(col_in, q, result_approx.get(), &ctxt);
      EXPECT_EQ( ret, expected_error) << "approx " << " returns unexpected failure\n";
      
      if( ret == GDF_SUCCESS ){
        double delta = std::abs(static_cast<double>(result_approx.value() - v_appox[j]));
        bool flag = delta < 1.0e-8;
        EXPECT_EQ( flag, true ) << " " << q << " appox quantile "
          << " val = " << result_approx.value() << ", " <<  v_appox[j];
      }

      for(size_t i = 0;i<n_methods;++i)
        {
          ret = cudf::quantile_exact(col_in, static_cast<cudf::interpolation>(i), q, result_exact.get(), &ctxt);
          EXPECT_EQ( ret, expected_error) << "exact " << methods[i] << " returns unexpected failure\n";

          if( ret == GDF_SUCCESS ){
            double delta = std::abs(static_cast<double>(result_exact.value() - v_exact[i][j]));
            bool flag = delta < 1.0e-8;
            EXPECT_EQ( flag, true ) << " "  << q  <<" exact quantile on " << methods[i]
              << " val = " << result_exact.value() << ", " <<  v_exact[i][j];
          }
        }
    }
}

struct gdf_quantile : public GdfTest {};

TEST_F(gdf_quantile, DoubleVector)
{
  using VType = double;
  std::vector<VType> v{6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7};
  cudf::test::column_wrapper<VType> col(v);

  std::vector<VType> v_baseline_approx{-1.01,   0.8,  0.8,    2.13,   6.8};
  std::vector<std::vector<double>> v_baseline_exact{
    {-1.01,   0.8,  0.9984, 2.13,   6.8},
    {-1.01,   0.8,  0.8,    2.13,   6.8},
    {-1.01,   0.8,  1.11,   2.13,   6.8},
    {-1.01,   0.8,  0.955,  2.13,   6.8},
    {-1.01,   0.8,  1.11,   2.13,   6.8}};

  f_quantile_tester<VType>(col.get(), v_baseline_approx, v_baseline_exact);
}

TEST_F(gdf_quantile, IntegerVector)
{
  using VType = int32_t;
  std::vector<VType> v{7, 0, 3, 4, 2, 1, -1, 1, 6};
  cudf::test::column_wrapper<VType> col(v);

  std::vector<VType> v_baseline_approx{-1,     1,     1,     2,     7};
  std::vector<std::vector<double>> v_baseline_exact{
    {-1.0,   1.0,   1.0,   2.0,   7.0},
    {-1,     1,     1,     2,     7},
    {-1,     1,     1,     2,     7},
    {-1.0,   1.0,   1.0,   2.0,   7.0},
    {-1,     1,     1,     2,     7}};

  f_quantile_tester<VType>(col.get(), v_baseline_approx, v_baseline_exact);
}

TEST_F(gdf_quantile, ReportValidMaskError)
{
  using VType = int32_t;
  std::vector<VType> v{7, 0, 3, 4, 2, 1, -1, 1, 6};
  std::vector<gdf_valid_type> bitmask(gdf_valid_allocation_size(v.size()), 0xF3);
  cudf::test::column_wrapper<VType> col(v, bitmask);

  std::vector<VType> v_baseline_approx{-1,     1,     1,     2,     7};
  std::vector<std::vector<double>> v_baseline_exact{
    {-1.0,   1.0,   1.0,   2.0,   7.0},
    {-1,     1,     1,     2,     7},
    {-1,     1,     1,     2,     7},
    {-1.0,   1.0,   1.0,   2.0,   7.0},
    {-1,     1,     1,     2,     7}};
  
  f_quantile_tester<VType>(col.get(), v_baseline_approx, v_baseline_exact, GDF_VALIDITY_UNSUPPORTED);
}


// ----------------------------------------------------

template<typename T_in>
struct InterpolateTest : public GdfTest {};

using TestingTypes = ::testing::Types<
    int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_CASE(InterpolateTest, TestingTypes);


template<typename T_out, typename T_in>
void midpoint_test(T_out exact, T_in lhs, T_in rhs)
{
    T_out result;
    cudf::interpolate::midpoint(result, lhs, rhs);

    EXPECT_EQ(exact, result) << "midpoint( " << lhs << ", " << rhs << ")";
}

TYPED_TEST(InterpolateTest, MidpointTest)
{
    using T = TypeParam;

    T t_max = std::numeric_limits<T>::max();
    T t_min = std::numeric_limits<T>::lowest();

    // integer overflow test
    midpoint_test(t_max, t_max, t_max);
    midpoint_test(t_min, t_min, t_min);
    midpoint_test(T(t_max-1), t_max, T(t_max-2));
    midpoint_test(T(t_min+1), t_min, T(t_min+T(2)));

    double mid_of_mimax = ( std::is_integral<T>::value == true ) ?
        -0.5 : 0.0;

    midpoint_test(mid_of_mimax, t_max, t_min);
    midpoint_test(mid_of_mimax, t_min, t_max);

    midpoint_test(T{0}, t_max, t_min);
    midpoint_test(T{0}, t_min, t_max);
    midpoint_test(T{0}, T(t_max-1), T(t_min+1));
    midpoint_test(T{0}, T(t_min+1), T(t_max-1));
    midpoint_test(T{0},   t_max,    T(t_min+1));
    midpoint_test(T{0}, T(t_min+1),   t_max);

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


