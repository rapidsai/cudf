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

#include <tests/utilities/cudf_test_fixtures.h>

#include <sqls/sqls_rtti_comp.h>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.h>
#include <cudf.h>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#include <tuple>
#include <iterator>
#include <type_traits>
#include <numeric>
#include <unordered_map>

#include <cassert>
#include <cmath>


///using IndexT = int;//okay...
using IndexT = size_t;

template<typename T, typename Allocator, template<typename, typename> class Vector>
__host__ 
void print_v(const Vector<T, Allocator>& v, std::ostream& os)
{
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(os,","));
  os<<"\n";
}

template<typename T,
   typename Allocator,
   template<typename, typename> class Vector>
__host__
void print_v(const Vector<T, Allocator>& v, typename Vector<T, Allocator>::const_iterator pos, std::ostream& os)
{ 
  thrust::copy(v.begin(), pos, std::ostream_iterator<T>(os,","));//okay
  os<<"\n";
}

template<typename T,
   typename Allocator,
   template<typename, typename> class Vector>
__host__
void print_v(const Vector<T, Allocator>& v, size_t n, std::ostream& os)
{ 
  thrust::copy_n(v.begin(), n, std::ostream_iterator<T>(os,","));//okay
  os<<"\n";
}

template<typename T>
bool compare(const rmm::device_vector<T>& d_v, const std::vector<T>& baseline, T eps)
{
  size_t n = baseline.size();//because d_v might be larger
  
  std::vector<T> h_v(n);
  std::vector<int> h_b(n, 0);

  thrust::copy_n(d_v.begin(), n, h_v.begin());//D-H okay...
  
  return std::inner_product(h_v.begin(), h_v.end(),
          baseline.begin(),
          true,
          [](bool b1, bool b2){
            return b1 && b2;
          },
          [eps](T v1, T v2){
            auto diff = (v1 <= v2) ? v2-v1 : v1-v2; // can't use std::abse due to type ambiguity
            return (diff < eps);
          });
}

struct gdf_group_by : public GdfTest {};

TEST_F(gdf_group_by, UsageTestSum)
{
  std::vector<int> vc1{1,1,1,1,1,1};
  std::vector<int> vi1{1,3,3,5,5,0};
  std::vector<double> vd1{12., 13., 13., 17., 17., 17};

  rmm::device_vector<int> dc1 = vc1;
  rmm::device_vector<int> di1 = vi1;
  rmm::device_vector<double> dd1 = vd1;
  
  size_t sz = dc1.size();
  assert( sz > 0 );
  assert( sz == di1.size() );
  assert( sz == dd1.size() );
 

  rmm::device_vector<IndexT> d_indx(sz, 0);
  rmm::device_vector<IndexT> d_keys(sz, 0);
  rmm::device_vector<IndexT> d_vals(sz, 0);

  size_t ncols = 3;
  size_t& nrows = sz;

  rmm::device_vector<void*> d_cols(ncols, nullptr);
  rmm::device_vector<int>   d_types(ncols, 0);

  std::vector<gdf_column> v_gdf_cols(ncols);
  v_gdf_cols[0].data = static_cast<void*>(dc1.data().get());
  v_gdf_cols[0].size = nrows;
  v_gdf_cols[0].dtype = GDF_INT32;

  v_gdf_cols[1].data = static_cast<void*>(di1.data().get());
  v_gdf_cols[1].size = nrows;
  v_gdf_cols[1].dtype = GDF_INT32;

  v_gdf_cols[2].data = static_cast<void*>(dd1.data().get());
  v_gdf_cols[2].size = nrows;
  v_gdf_cols[2].dtype = GDF_FLOAT64;

  gdf_column c_agg;
  gdf_column c_vout;

  rmm::device_vector<double> d_outd(sz, 0);

  c_agg.dtype = GDF_FLOAT64;
  c_agg.data = dd1.data().get();
  c_agg.valid = nullptr;
  c_agg.size = nrows;

  c_vout.dtype = GDF_FLOAT64;
  c_vout.data = d_outd.data().get();
  c_vout.size = nrows;

  size_t n_group = 0;
  //int flag_sorted = 0;

  std::cout<<"aggregate = sum on column:\n";
  print_v(dd1, std::cout);

  //input
  //{
  gdf_context ctxt{0, GDF_SORT, 0, 0};
  std::vector<gdf_column*> v_pcols(ncols);
  for(size_t i = 0; i < ncols; ++i)
    {
      v_pcols[i] = &v_gdf_cols[i];
    }
  gdf_column** cols = &v_pcols[0];//pointer semantic (2);
  //}

  //output:
  //{
  rmm::device_vector<int32_t> d_vc_out(nrows);
  rmm::device_vector<int32_t> d_vi_out(nrows);
  rmm::device_vector<double> d_vd_out(nrows);
    
  std::vector<gdf_column> v_gdf_cols_out(ncols);
  v_gdf_cols_out[0].data = d_vc_out.data().get();
  v_gdf_cols_out[0].dtype = GDF_INT32;
  v_gdf_cols_out[0].size = nrows;

  v_gdf_cols_out[1].data = d_vi_out.data().get();
  v_gdf_cols_out[1].dtype = GDF_INT32;
  v_gdf_cols_out[1].size = nrows;

  v_gdf_cols_out[2].data = d_vd_out.data().get();
  v_gdf_cols_out[2].dtype = GDF_FLOAT64;
  v_gdf_cols_out[2].size = nrows;

  std::vector<gdf_column*> h_cols_out(ncols);
  for(size_t i=0; i<ncols; ++i)
    h_cols_out[i] = &v_gdf_cols_out[i];//
  
  gdf_column** cols_out = &h_cols_out[0];//pointer semantics (2)

  d_keys.assign(nrows, 0);
  gdf_column c_indx;
  c_indx.data = d_keys.data().get();
  c_indx.valid = nullptr;
  c_indx.size = nrows;
  c_indx.dtype = GDF_INT32;
  //}

  ///EXPECT_EQ( 1, 1);
    
  gdf_group_by_sum((int)ncols,      // # columns
                   cols,            //input cols
                   &c_agg,          //column to aggregate on
                   &c_indx,         //if not null return indices of re-ordered rows
                   cols_out,        //if not null return the grouped-by columns
                   &c_vout,         //aggregation result
                   &ctxt);          //struct with additional info;
    
  n_group = c_vout.size;
  const size_t n_rows_expected = 4;
  const double deps = 1.e-8;
  const int ieps = 1;
  const IndexT szeps = 1;
  
  EXPECT_EQ( n_group, n_rows_expected ) << "GROUP-BY SUM returns unexpected #rows:" << n_group;

  //EXPECTED:
  //d_vc_out: 1,1,1,1,
  //d_vi_out: 0,1,3,5
  //d_vd_out: 17,12,13,17,
  vc1 = {1,1,1,1};
  vi1 = {0,1,3,5};
  vd1 = {17,12,13,17};

  bool flag = compare(d_vc_out, vc1, ieps);
  EXPECT_EQ( flag, true ) << "column 1 GROUP-BY returns unexpected result";

  flag = compare(d_vi_out, vi1, ieps);
  EXPECT_EQ( flag, true ) << "column 2 GROUP-BY returns unexpected result";

  flag = compare(d_vd_out, vd1, deps);
  EXPECT_EQ( flag, true ) << "column 3 GROUP-BY returns unexpected result";
  
  //d_keys: 5,0,2,4,
  //d_outd: 17,12,26,34,

  std::vector<IndexT> vk{5,0,2,4};
  vd1 = {17,12,26,34};

  flag = compare(d_keys, vk, szeps);
  EXPECT_EQ( flag, true ) << "GROUP-BY row indices return unexpected result";

  flag = compare(d_outd, vd1, deps);
  EXPECT_EQ( flag, true ) << "GROUP-BY SUM aggregation returns unexpected result";
}

TEST_F(gdf_group_by, UsageTestCount)
{
  std::vector<int> vc1{1,1,1,1,1,1};
  std::vector<int> vi1{1,3,3,5,5,0};
  std::vector<double> vd1{12., 13., 13., 17., 17., 17};

  rmm::device_vector<int> dc1 = vc1;
  rmm::device_vector<int> di1 = vi1;
  rmm::device_vector<double> dd1 = vd1;
  
  size_t sz = dc1.size();
  assert( sz > 0 );
  assert( sz == di1.size() );
  assert( sz == dd1.size() );
 
  rmm::device_vector<IndexT> d_indx(sz, 0);
  rmm::device_vector<IndexT> d_keys(sz, 0);
  rmm::device_vector<int32_t> d_vals(sz, 0);

  size_t ncols = 3;
  size_t& nrows = sz;

  rmm::device_vector<void*> d_cols(ncols, nullptr);
  rmm::device_vector<int>   d_types(ncols, 0);

  std::vector<gdf_column> v_gdf_cols(ncols);
  v_gdf_cols[0].data = static_cast<void*>(dc1.data().get());
  v_gdf_cols[0].size = nrows;
  v_gdf_cols[0].dtype = GDF_INT32;

  v_gdf_cols[1].data = static_cast<void*>(di1.data().get());
  v_gdf_cols[1].size = nrows;
  v_gdf_cols[1].dtype = GDF_INT32;

  v_gdf_cols[2].data = static_cast<void*>(dd1.data().get());
  v_gdf_cols[2].size = nrows;
  v_gdf_cols[2].dtype = GDF_FLOAT64;

  gdf_column c_agg;
  gdf_column c_vout;

  rmm::device_vector<double> d_outd(sz, 0);

  c_agg.dtype = GDF_FLOAT64;
  c_agg.data = dd1.data().get();
  c_agg.valid = nullptr;
  c_agg.size = nrows;

  c_vout.dtype = GDF_INT32;
  c_vout.data = d_vals.data().get();
  c_vout.size = nrows;

  size_t n_group = 0;
  //int flag_sorted = 0;

  std::cout<<"aggregate = count on column:\n";
  print_v(dd1, std::cout);

  //input
  //{
  gdf_context ctxt{0, GDF_SORT, 0, 0};
  std::vector<gdf_column*> v_pcols(ncols);
  for(size_t i = 0; i < ncols; ++i)
    {
      v_pcols[i] = &v_gdf_cols[i];
    }
  gdf_column** cols = &v_pcols[0];//pointer semantic (2);
  //}

  //output:
  //{
  rmm::device_vector<int32_t> d_vc_out(nrows);
  rmm::device_vector<int32_t> d_vi_out(nrows);
  rmm::device_vector<double> d_vd_out(nrows);
    
  std::vector<gdf_column> v_gdf_cols_out(ncols);
  v_gdf_cols_out[0].data = d_vc_out.data().get();
  v_gdf_cols_out[0].dtype = GDF_INT32;
  v_gdf_cols_out[0].size = nrows;

  v_gdf_cols_out[1].data = d_vi_out.data().get();
  v_gdf_cols_out[1].dtype = GDF_INT32;
  v_gdf_cols_out[1].size = nrows;

  v_gdf_cols_out[2].data = d_vd_out.data().get();
  v_gdf_cols_out[2].dtype = GDF_FLOAT64;
  v_gdf_cols_out[2].size = nrows;

  std::vector<gdf_column*> h_cols_out(ncols);
  for(size_t i=0; i<ncols; ++i)
    h_cols_out[i] = &v_gdf_cols_out[i];//
  
  gdf_column** cols_out = &h_cols_out[0];//pointer semantics (2)

  d_keys.assign(nrows, 0);
  gdf_column c_indx;
  c_indx.data = d_keys.data().get();
  c_indx.valid = nullptr;
  c_indx.size = nrows;
  c_indx.dtype = GDF_INT32;
  //}

  gdf_group_by_count((int)ncols,      // # columns
                   cols,            //input cols
                   &c_agg,          //column to aggregate on
                   &c_indx,         //if not null return indices of re-ordered rows
                   cols_out,        //if not null return the grouped-by columns
                   &c_vout,         //aggregation result
                   &ctxt);          //struct with additional info;
    
  n_group = c_vout.size;
  const size_t n_rows_expected = 4;
  const double deps = 1.e-8;
  const int ieps = 1;
  const IndexT szeps = 1;
  
  EXPECT_EQ( n_group, n_rows_expected ) << "GROUP-BY COUNT returns unexpected #rows:" << n_group;

  //EXPECTED:
  //d_vc_out: 1,1,1,1,
  //d_vi_out: 0,1,3,5
  //d_vd_out: 17,12,13,17,
  vc1 = {1,1,1,1};
  vi1 = {0,1,3,5};
  vd1 = {17,12,13,17};

  bool flag = compare(d_vc_out, vc1, ieps);
  EXPECT_EQ( flag, true ) << "column 1 GROUP-BY returns unexpected result";

  flag = compare(d_vi_out, vi1, ieps);
  EXPECT_EQ( flag, true ) << "column 2 GROUP-BY returns unexpected result";

  flag = compare(d_vd_out, vd1, deps);
  EXPECT_EQ( flag, true ) << "column 3 GROUP-BY returns unexpected result";
  
  //d_keys: 5,0,2,4,
  //d_vals: 1,1,2,2,

  std::vector<IndexT> vk{5,0,2,4};
  std::vector<int32_t> vals{1,1,2,2};

  flag = compare(d_keys, vk, szeps);
  EXPECT_EQ( flag, true ) << "GROUP-BY row indices return unexpected result";

  flag = compare(d_vals, vals, ieps);
  EXPECT_EQ( flag, true ) << "GROUP-BY COUNT aggregation returns unexpected result";
}

TEST_F(gdf_group_by, UsageTestAvg)
{
  std::vector<int> vc1{1,1,1,1,1,1};
  std::vector<int> vi1{1,3,3,5,5,0};
  std::vector<double> vd1{12., 13., 13., 17., 17., 17};

  rmm::device_vector<int> dc1 = vc1;
  rmm::device_vector<int> di1 = vi1;
  rmm::device_vector<double> dd1 = vd1;

  size_t sz = dc1.size();
  assert( sz == di1.size() );
  assert( sz == dd1.size() );
    
  rmm::device_vector<IndexT> d_indx(sz, 0);
  rmm::device_vector<IndexT> d_keys(sz, 0);
  rmm::device_vector<IndexT> d_vals(sz, 0);

  size_t ncols = 3;
  size_t& nrows = sz;

  rmm::device_vector<void*> d_cols(ncols, nullptr);
  rmm::device_vector<int>   d_types(ncols, 0);

  std::vector<gdf_column> v_gdf_cols(ncols);
  v_gdf_cols[0].data = static_cast<void*>(dc1.data().get());
  v_gdf_cols[0].size = nrows;
  v_gdf_cols[0].dtype = GDF_INT32;

  v_gdf_cols[1].data = static_cast<void*>(di1.data().get());
  v_gdf_cols[1].size = nrows;
  v_gdf_cols[1].dtype = GDF_INT32;

  v_gdf_cols[2].data = static_cast<void*>(dd1.data().get());
  v_gdf_cols[2].size = nrows;
  v_gdf_cols[2].dtype = GDF_FLOAT64;


  gdf_column c_agg;
  gdf_column c_vout;

  rmm::device_vector<double> d_outd(sz, 0);

  c_agg.dtype = GDF_FLOAT64;
  c_agg.data = dd1.data().get();
  c_agg.valid = nullptr;
  c_agg.size = nrows;

  c_vout.dtype = GDF_FLOAT64;
  c_vout.data = d_outd.data().get();
  c_vout.size = nrows;

  size_t n_group = 0;
  //int flag_sorted = 0;

  std::cout<<"aggregate = avg on column:\n";
  print_v(dd1, std::cout);

  //input
  //{
  gdf_context ctxt{0, GDF_SORT, 0, 0};
  std::vector<gdf_column*> v_pcols(ncols);
  for(size_t i = 0; i < ncols; ++i)
    {
      v_pcols[i] = &v_gdf_cols[i];
    }
  gdf_column** cols = &v_pcols[0];//pointer semantic (2);
  //}

  //output:
  //{
  rmm::device_vector<int32_t> d_vc_out(nrows);
  rmm::device_vector<int32_t> d_vi_out(nrows);
  rmm::device_vector<double> d_vd_out(nrows);
    
  std::vector<gdf_column> v_gdf_cols_out(ncols);
  v_gdf_cols_out[0].data = d_vc_out.data().get();
  v_gdf_cols_out[0].dtype = GDF_INT32;
  v_gdf_cols_out[0].size = nrows;

  v_gdf_cols_out[1].data = d_vi_out.data().get();
  v_gdf_cols_out[1].dtype = GDF_INT32;
  v_gdf_cols_out[1].size = nrows;

  v_gdf_cols_out[2].data = d_vd_out.data().get();
  v_gdf_cols_out[2].dtype = GDF_FLOAT64;
  v_gdf_cols_out[2].size = nrows;

  std::vector<gdf_column*> h_cols_out(ncols);
  for(size_t i=0; i<ncols; ++i)
    h_cols_out[i] = &v_gdf_cols_out[i];//
  
  gdf_column** cols_out = &h_cols_out[0];//pointer semantics (2)

  d_keys.assign(nrows, 0);
  gdf_column c_indx;
  c_indx.data = d_keys.data().get();
  c_indx.valid = nullptr;
  c_indx.size = nrows;
  c_indx.dtype = GDF_INT32;
  //}
    
  gdf_group_by_avg((int)ncols,      // # columns
                   cols,            //input cols
                   &c_agg,          //column to aggregate on
                   &c_indx,         //if not null return indices of re-ordered rows
                   cols_out,        //if not null return the grouped-by columns
                   &c_vout,         //aggregation result
                   &ctxt);          //struct with additional info;
    
  n_group = c_vout.size;
  const size_t n_rows_expected = 4;
  const double deps = 1.e-8;
  const int ieps = 1;
  const IndexT szeps = 1;
  
  EXPECT_EQ( n_group, n_rows_expected ) << "GROUP-BY AVG returns unexpected #rows:" << n_group;

  //EXPECTED:
  //d_vc_out: 1,1,1,1,
  //d_vi_out: 0,1,3,5
  //d_vd_out: 17,12,13,17,
  vc1 = {1,1,1,1};
  vi1 = {0,1,3,5};
  vd1 = {17,12,13,17};

  bool flag = compare(d_vc_out, vc1, ieps);
  EXPECT_EQ( flag, true ) << "column 1 GROUP-BY returns unexpected result";

  flag = compare(d_vi_out, vi1, ieps);
  EXPECT_EQ( flag, true ) << "column 2 GROUP-BY returns unexpected result";

  flag = compare(d_vd_out, vd1, deps);
  EXPECT_EQ( flag, true ) << "column 3 GROUP-BY returns unexpected result";
  
  //d_keys: 5,0,2,4,
  //d_outd: 17,12,13,17,

  std::vector<IndexT> vk{5,0,2,4};
  vd1 = {17,12,13,17};

  flag = compare(d_keys, vk, szeps);
  EXPECT_EQ( flag, true ) << "GROUP-BY row indices return unexpected result";

  flag = compare(d_outd, vd1, deps);
  EXPECT_EQ( flag, true ) << "GROUP-BY AVG aggregation returns unexpected result";
}

TEST_F(gdf_group_by, UsageTestMin)
{
  std::vector<int> vc1{1,1,1,1,1,1};
  std::vector<int> vi1{1,3,3,5,5,0};
  std::vector<double> vd1{12., 13., 13., 17., 17., 17};

  rmm::device_vector<int> dc1 = vc1;
  rmm::device_vector<int> di1 = vi1;
  rmm::device_vector<double> dd1 = vd1;

  size_t sz = dc1.size();
  assert( sz == di1.size() );
  assert( sz == dd1.size() );
    
  rmm::device_vector<IndexT> d_indx(sz, 0);
  rmm::device_vector<IndexT> d_keys(sz, 0);
  rmm::device_vector<IndexT> d_vals(sz, 0);

  size_t ncols = 3;
  size_t& nrows = sz;

  rmm::device_vector<void*> d_cols(ncols, nullptr);
  rmm::device_vector<int>   d_types(ncols, 0);

  std::vector<gdf_column> v_gdf_cols(ncols);
  v_gdf_cols[0].data = static_cast<void*>(dc1.data().get());
  v_gdf_cols[0].size = nrows;
  v_gdf_cols[0].dtype = GDF_INT32;

  v_gdf_cols[1].data = static_cast<void*>(di1.data().get());
  v_gdf_cols[1].size = nrows;
  v_gdf_cols[1].dtype = GDF_INT32;

  v_gdf_cols[2].data = static_cast<void*>(dd1.data().get());
  v_gdf_cols[2].size = nrows;
  v_gdf_cols[2].dtype = GDF_FLOAT64;

  gdf_column c_agg;
  gdf_column c_vout;

  rmm::device_vector<double> d_outd(sz, 0);

  c_agg.dtype = GDF_FLOAT64;
  ///c_agg.data = dd1.data().get();
  c_agg.size = nrows;

  c_vout.dtype = GDF_FLOAT64;
  c_vout.data = d_outd.data().get();
  c_vout.size = nrows;

  size_t n_group = 0;
  //int flag_sorted = 0;

  std::vector<double> v_col{2., 4., 5., 7., 11., 3.};
  rmm::device_vector<double> d_col = v_col;

  std::cout<<"aggregate = min on column:\n";
  print_v(d_col, std::cout);

  c_agg.dtype = GDF_FLOAT64;
  c_agg.data = d_col.data().get();
  c_agg.valid = nullptr;

  //input
  //{
  gdf_context ctxt{0, GDF_SORT, 0, 0};
  std::vector<gdf_column*> v_pcols(ncols);
  for(size_t i = 0; i < ncols; ++i)
    {
      v_pcols[i] = &v_gdf_cols[i];
    }
  gdf_column** cols = &v_pcols[0];//pointer semantic (2);
  //}

  //output:
  //{
  rmm::device_vector<int32_t> d_vc_out(nrows);
  rmm::device_vector<int32_t> d_vi_out(nrows);
  rmm::device_vector<double> d_vd_out(nrows);
    
  std::vector<gdf_column> v_gdf_cols_out(ncols);
  v_gdf_cols_out[0].data = d_vc_out.data().get();
  v_gdf_cols_out[0].dtype = GDF_INT32;
  v_gdf_cols_out[0].size = nrows;

  v_gdf_cols_out[1].data = d_vi_out.data().get();
  v_gdf_cols_out[1].dtype = GDF_INT32;
  v_gdf_cols_out[1].size = nrows;

  v_gdf_cols_out[2].data = d_vd_out.data().get();
  v_gdf_cols_out[2].dtype = GDF_FLOAT64;
  v_gdf_cols_out[2].size = nrows;

  std::vector<gdf_column*> h_cols_out(ncols);
  for(size_t i=0; i<ncols; ++i)
    h_cols_out[i] = &v_gdf_cols_out[i];//
  
  gdf_column** cols_out = &h_cols_out[0];//pointer semantics (2)

  d_keys.assign(nrows, 0);
  gdf_column c_indx;
  c_indx.data = d_keys.data().get();
  c_indx.valid = nullptr;
  c_indx.size = nrows;
  c_indx.dtype = GDF_INT32;
  //}
    
  gdf_group_by_min((int)ncols,      // # columns
                   cols,            //input cols
                   &c_agg,          //column to aggregate on
                   &c_indx,         //if not null return indices of re-ordered rows
                   cols_out,        //if not null return the grouped-by columns
                   &c_vout,         //aggregation result
                   &ctxt);          //struct with additional info;
    
  n_group = c_vout.size;
  const size_t n_rows_expected = 4;
  const double deps = 1.e-8;
  const int ieps = 1;
  const IndexT szeps = 1;
  
  EXPECT_EQ( n_group, n_rows_expected ) << "GROUP-BY MIN returns unexpected #rows:" << n_group;

  //EXPECTED:
  //d_vc_out: 1,1,1,1,
  //d_vi_out: 0,1,3,5
  //d_vd_out: 17,12,13,17,
  vc1 = {1,1,1,1};
  vi1 = {0,1,3,5};
  vd1 = {17,12,13,17};

  bool flag = compare(d_vc_out, vc1, ieps);
  EXPECT_EQ( flag, true ) << "column 1 GROUP-BY returns unexpected result";

  flag = compare(d_vi_out, vi1, ieps);
  EXPECT_EQ( flag, true ) << "column 2 GROUP-BY returns unexpected result";

  flag = compare(d_vd_out, vd1, deps);
  EXPECT_EQ( flag, true ) << "column 3 GROUP-BY returns unexpected result";
    
  //d_keys: 5,0,2,4,
  //d_outd: 3,2,4,7,

  std::vector<IndexT> vk{5,0,2,4};
  vd1 = {3,2,4,7};

  flag = compare(d_keys, vk, szeps);
  EXPECT_EQ( flag, true ) << "GROUP-BY row indices return unexpected result";

  flag = compare(d_outd, vd1, deps);
  EXPECT_EQ( flag, true ) << "GROUP-BY MIN aggregation returns unexpected result";
}

TEST_F(gdf_group_by, UsageTestMax)
{
  std::vector<int> vc1{1,1,1,1,1,1};
  std::vector<int> vi1{1,3,3,5,5,0};
  std::vector<double> vd1{12., 13., 13., 17., 17., 17};

  rmm::device_vector<int> dc1 = vc1;
  rmm::device_vector<int> di1 = vi1;
  rmm::device_vector<double> dd1 = vd1;

  size_t sz = dc1.size();
  assert( sz == di1.size() );
  assert( sz == dd1.size() );
    
  rmm::device_vector<IndexT> d_indx(sz, 0);
  rmm::device_vector<IndexT> d_keys(sz, 0);
  rmm::device_vector<IndexT> d_vals(sz, 0);

  size_t ncols = 3;
  size_t& nrows = sz;

  rmm::device_vector<void*> d_cols(ncols, nullptr);
  rmm::device_vector<int>   d_types(ncols, 0);

  std::vector<gdf_column> v_gdf_cols(ncols);
  v_gdf_cols[0].data = static_cast<void*>(dc1.data().get());
  v_gdf_cols[0].size = nrows;
  v_gdf_cols[0].dtype = GDF_INT32;

  v_gdf_cols[1].data = static_cast<void*>(di1.data().get());
  v_gdf_cols[1].size = nrows;
  v_gdf_cols[1].dtype = GDF_INT32;

  v_gdf_cols[2].data = static_cast<void*>(dd1.data().get());
  v_gdf_cols[2].size = nrows;
  v_gdf_cols[2].dtype = GDF_FLOAT64;

  gdf_column c_agg;
  gdf_column c_vout;

  rmm::device_vector<double> d_outd(sz, 0);

  c_agg.dtype = GDF_FLOAT64;
  ///c_agg.data = dd1.data().get();
  c_agg.size = nrows;

  c_vout.dtype = GDF_FLOAT64;
  c_vout.data = d_outd.data().get();
  c_vout.size = nrows;

  size_t n_group = 0;
  //int flag_sorted = 0;

  std::vector<double> v_col{2., 4., 5., 7., 11., 3.};
  rmm::device_vector<double> d_col = v_col;

  std::cout<<"aggregate = max on column:\n";
  print_v(d_col, std::cout);

  c_agg.dtype = GDF_FLOAT64;
  c_agg.data = d_col.data().get();
  c_agg.valid = nullptr;

  //input
  //{
  gdf_context ctxt{0, GDF_SORT, 0, 0};
  std::vector<gdf_column*> v_pcols(ncols);
  for(size_t i = 0; i < ncols; ++i)
    {
      v_pcols[i] = &v_gdf_cols[i];
    }
  gdf_column** cols = &v_pcols[0];//pointer semantic (2);
  //}

  //output:
  //{
  rmm::device_vector<int32_t> d_vc_out(nrows);
  rmm::device_vector<int32_t> d_vi_out(nrows);
  rmm::device_vector<double> d_vd_out(nrows);
    
  std::vector<gdf_column> v_gdf_cols_out(ncols);
  v_gdf_cols_out[0].data = d_vc_out.data().get();
  v_gdf_cols_out[0].dtype = GDF_INT32;
  v_gdf_cols_out[0].size = nrows;

  v_gdf_cols_out[1].data = d_vi_out.data().get();
  v_gdf_cols_out[1].dtype = GDF_INT32;
  v_gdf_cols_out[1].size = nrows;

  v_gdf_cols_out[2].data = d_vd_out.data().get();
  v_gdf_cols_out[2].dtype = GDF_FLOAT64;
  v_gdf_cols_out[2].size = nrows;

  std::vector<gdf_column*> h_cols_out(ncols);
  for(size_t i=0; i<ncols; ++i)
    h_cols_out[i] = &v_gdf_cols_out[i];//
  
  gdf_column** cols_out = &h_cols_out[0];//pointer semantics (2)

  d_keys.assign(nrows, 0);
  gdf_column c_indx;
  c_indx.data = d_keys.data().get();
  c_indx.valid = nullptr;
  c_indx.size = nrows;
  c_indx.dtype = GDF_INT32;
  //}
    
  gdf_group_by_max((int)ncols,      // # columns
                   cols,            //input cols
                   &c_agg,          //column to aggregate on
                   &c_indx,         //if not null return indices of re-ordered rows
                   cols_out,        //if not null return the grouped-by columns
                   &c_vout,         //aggregation result
                   &ctxt);          //struct with additional info;
    
  n_group = c_vout.size;
  const size_t n_rows_expected = 4;
  const double deps = 1.e-8;
  const int ieps = 1;
  const IndexT szeps = 1;
  
  EXPECT_EQ( n_group, n_rows_expected ) << "GROUP-BY MAX returns unexpected #rows:" << n_group;

  //EXPECTED:
  //d_vc_out: 1,1,1,1,
  //d_vi_out: 0,1,3,5
  //d_vd_out: 17,12,13,17,
  vc1 = {1,1,1,1};
  vi1 = {0,1,3,5};
  vd1 = {17,12,13,17};

  bool flag = compare(d_vc_out, vc1, ieps);
  EXPECT_EQ( flag, true ) << "column 1 GROUP-BY returns unexpected result";

  flag = compare(d_vi_out, vi1, ieps);
  EXPECT_EQ( flag, true ) << "column 2 GROUP-BY returns unexpected result";

  flag = compare(d_vd_out, vd1, deps);
  EXPECT_EQ( flag, true ) << "column 3 GROUP-BY returns unexpected result";
    
  //d_keys: 5,0,2,4,
  //d_outd: 3,2,5,11,

  std::vector<IndexT> vk{5,0,2,4};
  vd1 = {3,2,5,11};

  flag = compare(d_keys, vk, szeps);
  EXPECT_EQ( flag, true ) << "GROUP-BY row indices return unexpected result";

  flag = compare(d_outd, vd1, deps);
  EXPECT_EQ( flag, true ) << "GROUP-BY MAX aggregation returns unexpected result";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


