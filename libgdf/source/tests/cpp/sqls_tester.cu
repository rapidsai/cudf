//Andrei Schaffer, 4/6/2018: For join project ():
//                           group_by, multicolumn join
//                           nvcc -w -std=c++11 --expt-extended-lambda <src_name>.cu -o <src_name>.exe
//
//                           nvcc -I/$HOME/Development/Cuda_Thrust -c -w -std=c++11 --expt-extended-lambda sqls_join4.cu
//                           nvcc -I/$HOME/Development/Cuda_Thrust -c -w -std=c++11 --expt-extended-lambda sqls_ops.cu
//                           nvcc -I/$HOME/Development/Cuda_Thrust -w -std=c++11 --expt-extended-lambda sqls_join4.cu sqls_ops.cu -o sqls_join.exe
//
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

#include "thrust_rmm_allocator.h"

#include <iostream>
#include <vector>
#include <tuple>
#include <iterator>
#include <cassert>
#include <type_traits>

#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>


#include "sqls_rtti_comp.hpp"

extern gdf_error gdf_filter(size_t nrows,
		     gdf_column* cols,
		     size_t ncols,
		     void** d_cols,//device-side data slicing of gdf_column array (host)
		     int* d_types, //device-side dtype slicing of gdf_column array (host)
		     void** d_vals,
		     size_t* d_indx,
		     size_t* new_sz);

// Vector set to use rmmAlloc and rmmFree.
template <typename T>
using Vector = thrust::device_vector<T, rmm_allocator<T>>;

///using IndexT = int;//okay...
using IndexT = size_t;

template<typename T, typename Allocator, template<typename, typename> class Vector>
__host__ __device__
void print_v(const Vector<T, Allocator>& v, std::ostream& os)
{
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(os,","));
  os<<"\n";
}

template<typename T,
	 typename Allocator,
	 template<typename, typename> class Vector>
__host__ __device__
void print_v(const Vector<T, Allocator>& v, typename Vector<T, Allocator>::const_iterator pos, std::ostream& os)
{ 
  thrust::copy(v.begin(), pos, std::ostream_iterator<T>(os,","));//okay
  os<<"\n";
}

template<typename T,
	 typename Allocator,
	 template<typename, typename> class Vector>
__host__ __device__
void print_v(const Vector<T, Allocator>& v, size_t n, std::ostream& os)
{ 
  thrust::copy_n(v.begin(), n, std::ostream_iterator<T>(os,","));//okay
  os<<"\n";
}

void f_test_multi_filter(void)
{
  std::vector<int> vc1{1,1,1,1,1,1};
  std::vector<int> vi1{1,3,3,5,5,5};
  std::vector<double> vd1{12., 13., 13., 17., 17., 17};

  Vector<int> dc1 = vc1;
  Vector<int> di1 = vi1;
  Vector<double> dd1 = vd1;

  size_t nrows = dc1.size();
  assert( nrows == di1.size() );
  assert( nrows == dd1.size() );

  ///thrust::tuple<int*, int*, double*> tptrs{dc1.data().get(), di1.data().get(), dd1.data().get()};

  int i = 1;
  int j = 3;
  double d = 13.;

  vc1.resize(1);
  vi1.resize(1);
  vd1.resize(1);
  
  vc1[0] = i;
  vi1[0] = j;
  vd1[0] = d;
  
  Vector<int> d_ci = vc1;
  Vector<int> d_cj = vi1;
  Vector<double> d_cd = vd1;

  ///thrust::tuple<int, int, double> tvals{i,j,d};

  size_t new_sz = 0;
  Vector<size_t> d_indices(nrows, 0);

  // thrust::tuple<int*, int*, int*, int*, double*, double*>
  //   tpairs{dc1.data().get(), d_ci.data().get(),
  //          di1.data().get(), d_cj.data().get(),
  //          dd1.data().get(), d_cd.data().get()};

  ///new_sz = multi_col_filter(nrows, tpairs, d_indices.data().get());//ok

  size_t ncols = 3;
  Vector<void*> d_cols(ncols, nullptr);
  Vector<int>   d_types(ncols, 0);
  Vector<size_t> d_indx(nrows, 0);

  std::vector<gdf_column> v_gdf_cols(ncols);
  v_gdf_cols[0].data = static_cast<void*>(dc1.data().get());
  v_gdf_cols[0].size = sizeof(int);
  v_gdf_cols[0].dtype = GDF_INT32;

  v_gdf_cols[1].data = static_cast<void*>(di1.data().get());
  v_gdf_cols[1].size = sizeof(int);
  v_gdf_cols[1].dtype = GDF_INT32;

  v_gdf_cols[2].data = static_cast<void*>(dd1.data().get());
  v_gdf_cols[2].size = sizeof(double);
  v_gdf_cols[2].dtype = GDF_FLOAT64;

  std::vector<void*> v_vals{static_cast<void*>(d_ci.data().get()),
                            static_cast<void*>(d_cj.data().get()),
	                    static_cast<void*>(d_cd.data().get())};

  Vector<void*> d_vals = v_vals;

  gdf_column* h_columns = &v_gdf_cols[0];
  void** d_col_data = d_cols.data().get();
  int* d_col_types = d_types.data().get();
  size_t* ptr_d_indx = d_indices.data().get();
  void** ptr_d_vals = d_vals.data().get();

  gdf_filter(nrows, h_columns, ncols, d_col_data, d_col_types, ptr_d_vals, ptr_d_indx, &new_sz);

  bool res = (new_sz > 0);

  if( res )
    {
      std::cout<<"filtered size: "<<new_sz<<"; filtered indices:\n";
      print_v(d_indices, d_indices.begin()+new_sz, std::cout);
    }
  else
    std::cout << "table unfiltered.\n";
}



extern gdf_error gdf_order_by(size_t nrows,     //in: # rows
                              gdf_column* cols, //in: host-side array of gdf_columns
                              size_t ncols,     //in: # cols
                              void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
                              int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
                              size_t* d_indx);   //out: device-side array of re-rdered row indices


extern gdf_error gdf_filter(size_t nrows,     //in: # rows
                            gdf_column* cols, //in: host-side array of gdf_columns
                            size_t ncols,     //in: # cols
                            void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
                            int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
                            void** d_vals,    //in: device-side array of values to filter against (type-erased)
                            size_t* d_indx,   //out: device-side array of row indices that remain after filtering
                            size_t* new_sz);   //out: host-side # rows that remain after filtering


void test_gb_sum_api_2(const std::vector<int>& vc1,
                       const std::vector<int>& vi1,
                       const std::vector<double>& vd1)
{
  Vector<int> dc1 = vc1;
  Vector<int> di1 = vi1;
  Vector<double> dd1 = vd1;

  size_t sz = dc1.size();
  assert( sz == di1.size() );
  assert( sz == dd1.size() );
    
  Vector<IndexT> d_indx(sz, 0);
  Vector<IndexT> d_keys(sz, 0);
  Vector<IndexT> d_vals(sz, 0);

  size_t ncols = 3;
  size_t& nrows = sz;

  Vector<void*> d_cols(ncols, nullptr);
  Vector<int>   d_types(ncols, 0);

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

  Vector<double> d_outd(sz, 0);

  c_agg.dtype = GDF_FLOAT64;
  c_agg.data = dd1.data().get();
  c_agg.size = nrows;

  c_vout.dtype = GDF_FLOAT64;
  c_vout.data = d_outd.data().get();
  c_vout.size = nrows;

  size_t n_group = 0;
  int flag_sorted = 0;

  std::cout<<"aggregate = sum on column:\n";
  print_v(dd1, std::cout);

  //input
  //{
  gdf_context ctxt{0, GDF_SORT, 0};
  std::vector<gdf_column*> v_pcols(ncols);
  for(int i = 0; i < ncols; ++i)
    {
      v_pcols[i] = &v_gdf_cols[i];
    }
  gdf_column** cols = &v_pcols[0];//pointer semantic (2);
  //}

  //output:
  //{
  Vector<int32_t> d_vc_out(nrows);
  Vector<int32_t> d_vi_out(nrows);
  Vector<double> d_vd_out(nrows);
    
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
  for(int i=0; i<ncols; ++i)
    h_cols_out[i] = &v_gdf_cols_out[i];//
  
  gdf_column** cols_out = &h_cols_out[0];//pointer semantics (2)

  d_keys.assign(nrows, 0);
  gdf_column c_indx;
  c_indx.data = d_keys.data().get();
  c_indx.size = nrows;
  c_indx.dtype = GDF_INT32;
  //}
    
  gdf_group_by_sum((int)ncols,      // # columns
                   cols,            //input cols
                   &c_agg,          //column to aggregate on
                   &c_indx,         //if not null return indices of re-ordered rows
                   cols_out,        //if not null return the grouped-by columns
                   &c_vout,         //aggregation result
                   &ctxt);          //struct with additional info;
    
  n_group = c_vout.size;
  std::cout<<"selected rows:\n";
  print_v(d_vc_out, n_group, std::cout);
  print_v(d_vi_out, n_group, std::cout);
  print_v(d_vd_out, n_group, std::cout);
    
  size_t n_keys = n_group;
  size_t n_vals = n_group;
    
  std::cout<<"multi-col generic group-by (sum):\n";
  std::cout<<"indices of grouped-by, new keys end position: "<<n_keys<<";\n";
  std::cout<<"indices of grouped-by, new vals end position: "<<n_vals<<";\n";
    
  std::cout<<"grouped-by keys:\n";
  print_v(d_keys, n_keys, std::cout);

  std::cout<<"grouped-by vals:\n";
  print_v(d_outd, n_vals, std::cout);
}

void test_gb_count_api_2(const std::vector<int>& vc1,
                       const std::vector<int>& vi1,
                       const std::vector<double>& vd1)
{
  Vector<int> dc1 = vc1;
  Vector<int> di1 = vi1;
  Vector<double> dd1 = vd1;

  size_t sz = dc1.size();
  assert( sz == di1.size() );
  assert( sz == dd1.size() );
    
  Vector<IndexT> d_indx(sz, 0);
  Vector<IndexT> d_keys(sz, 0);
  Vector<IndexT> d_vals(sz, 0);

  size_t ncols = 3;
  size_t& nrows = sz;

  Vector<void*> d_cols(ncols, nullptr);
  Vector<int>   d_types(ncols, 0);

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

  Vector<double> d_outd(sz, 0);

  c_agg.dtype = GDF_FLOAT64;
  c_agg.data = dd1.data().get();
  c_agg.size = nrows;

  c_vout.dtype = GDF_INT32;
  c_vout.data = d_vals.data().get();
  c_vout.size = nrows;

  size_t n_group = 0;
  int flag_sorted = 0;

  std::cout<<"aggregate = count on column:\n";
  print_v(dd1, std::cout);

  //input
  //{
  gdf_context ctxt{0, GDF_SORT, 0};
  std::vector<gdf_column*> v_pcols(ncols);
  for(int i = 0; i < ncols; ++i)
    {
      v_pcols[i] = &v_gdf_cols[i];
    }
  gdf_column** cols = &v_pcols[0];//pointer semantic (2);
  //}

  //output:
  //{
  Vector<int32_t> d_vc_out(nrows);
  Vector<int32_t> d_vi_out(nrows);
  Vector<double> d_vd_out(nrows);
    
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
  for(int i=0; i<ncols; ++i)
    h_cols_out[i] = &v_gdf_cols_out[i];//
  
  gdf_column** cols_out = &h_cols_out[0];//pointer semantics (2)

  d_keys.assign(nrows, 0);
  gdf_column c_indx;
  c_indx.data = d_keys.data().get();
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
  std::cout<<"selected rows:\n";
  print_v(d_vc_out, n_group, std::cout);
  print_v(d_vi_out, n_group, std::cout);
  print_v(d_vd_out, n_group, std::cout);
    
  size_t n_keys = n_group;
  size_t n_vals = n_group;
    
  std::cout<<"multi-col count group-by:\n";
  std::cout<<"indices of grouped-by, new keys end position: "<<n_keys<<";\n";
  std::cout<<"indices of grouped-by, new vals end position: "<<n_vals<<";\n";
    
  std::cout<<"grouped-by keys:\n";
  print_v(d_keys, n_keys, std::cout);

  std::cout<<"grouped-by vals:\n";
  print_v(d_vals, n_vals, std::cout);
}

void test_gb_min_api_2(const std::vector<int>& vc1,
                       const std::vector<int>& vi1,
                       const std::vector<double>& vd1)
{
  Vector<int> dc1 = vc1;
  Vector<int> di1 = vi1;
  Vector<double> dd1 = vd1;

  size_t sz = dc1.size();
  assert( sz == di1.size() );
  assert( sz == dd1.size() );
    
  Vector<IndexT> d_indx(sz, 0);
  Vector<IndexT> d_keys(sz, 0);
  Vector<IndexT> d_vals(sz, 0);

  size_t ncols = 3;
  size_t& nrows = sz;

  Vector<void*> d_cols(ncols, nullptr);
  Vector<int>   d_types(ncols, 0);

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

  Vector<double> d_outd(sz, 0);

  c_agg.dtype = GDF_FLOAT64;
  ///c_agg.data = dd1.data().get();
  c_agg.size = nrows;

  c_vout.dtype = GDF_FLOAT64;
  c_vout.data = d_outd.data().get();
  c_vout.size = nrows;

  size_t n_group = 0;
  int flag_sorted = 0;

  std::vector<double> v_col{2., 4., 5., 7., 11., 3.};
  Vector<double> d_col = v_col;

  std::cout<<"aggregate = min on column:\n";
  print_v(d_col, std::cout);

  c_agg.dtype = GDF_FLOAT64;
  c_agg.data = d_col.data().get();

  //input
  //{
  gdf_context ctxt{0, GDF_SORT, 0};
  std::vector<gdf_column*> v_pcols(ncols);
  for(int i = 0; i < ncols; ++i)
    {
      v_pcols[i] = &v_gdf_cols[i];
    }
  gdf_column** cols = &v_pcols[0];//pointer semantic (2);
  //}

  //output:
  //{
  Vector<int32_t> d_vc_out(nrows);
  Vector<int32_t> d_vi_out(nrows);
  Vector<double> d_vd_out(nrows);
    
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
  for(int i=0; i<ncols; ++i)
    h_cols_out[i] = &v_gdf_cols_out[i];//
  
  gdf_column** cols_out = &h_cols_out[0];//pointer semantics (2)

  d_keys.assign(nrows, 0);
  gdf_column c_indx;
  c_indx.data = d_keys.data().get();
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
  std::cout<<"selected rows:\n";
  print_v(d_vc_out, n_group, std::cout);
  print_v(d_vi_out, n_group, std::cout);
  print_v(d_vd_out, n_group, std::cout);
    
  size_t n_keys = n_group;
  size_t n_vals = n_group;
    
  std::cout<<"multi-col generic group-by (min):\n";
  std::cout<<"indices of grouped-by, new keys end position: "<<n_keys<<";\n";
  std::cout<<"indices of grouped-by, new vals end position: "<<n_vals<<";\n";
    
  std::cout<<"grouped-by keys:\n";
  print_v(d_keys, n_keys, std::cout);

  std::cout<<"grouped-by vals:\n";
  print_v(d_outd, n_vals, std::cout);
}



void test_gb_max_api_2(const std::vector<int>& vc1,
                       const std::vector<int>& vi1,
                       const std::vector<double>& vd1)
{
  Vector<int> dc1 = vc1;
  Vector<int> di1 = vi1;
  Vector<double> dd1 = vd1;

  size_t sz = dc1.size();
  assert( sz == di1.size() );
  assert( sz == dd1.size() );
    
  Vector<IndexT> d_indx(sz, 0);
  Vector<IndexT> d_keys(sz, 0);
  Vector<IndexT> d_vals(sz, 0);

  size_t ncols = 3;
  size_t& nrows = sz;

  Vector<void*> d_cols(ncols, nullptr);
  Vector<int>   d_types(ncols, 0);

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

  Vector<double> d_outd(sz, 0);

  c_agg.dtype = GDF_FLOAT64;
  ///c_agg.data = dd1.data().get();
  c_agg.size = nrows;

  c_vout.dtype = GDF_FLOAT64;
  c_vout.data = d_outd.data().get();
  c_vout.size = nrows;

  size_t n_group = 0;
  int flag_sorted = 0;

  std::vector<double> v_col{2., 4., 5., 7., 11., 3.};
  Vector<double> d_col = v_col;

  std::cout<<"aggregate = max on column:\n";
  print_v(d_col, std::cout);

  c_agg.dtype = GDF_FLOAT64;
  c_agg.data = d_col.data().get();

  //input
  //{
  gdf_context ctxt{0, GDF_SORT, 0};
  std::vector<gdf_column*> v_pcols(ncols);
  for(int i = 0; i < ncols; ++i)
    {
      v_pcols[i] = &v_gdf_cols[i];
    }
  gdf_column** cols = &v_pcols[0];//pointer semantic (2);
  //}

  //output:
  //{
  Vector<int32_t> d_vc_out(nrows);
  Vector<int32_t> d_vi_out(nrows);
  Vector<double> d_vd_out(nrows);
    
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
  for(int i=0; i<ncols; ++i)
    h_cols_out[i] = &v_gdf_cols_out[i];//
  
  gdf_column** cols_out = &h_cols_out[0];//pointer semantics (2)

  d_keys.assign(nrows, 0);
  gdf_column c_indx;
  c_indx.data = d_keys.data().get();
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
  std::cout<<"selected rows:\n";
  print_v(d_vc_out, n_group, std::cout);
  print_v(d_vi_out, n_group, std::cout);
  print_v(d_vd_out, n_group, std::cout);
    
  size_t n_keys = n_group;
  size_t n_vals = n_group;
    
  std::cout<<"multi-col generic group-by (max):\n";
  std::cout<<"indices of grouped-by, new keys end position: "<<n_keys<<";\n";
  std::cout<<"indices of grouped-by, new vals end position: "<<n_vals<<";\n";
    
  std::cout<<"grouped-by keys:\n";
  print_v(d_keys, n_keys, std::cout);

  std::cout<<"grouped-by vals:\n";
  print_v(d_outd, n_vals, std::cout);
}




void test_gb_avg_api_2(const std::vector<int>& vc1,
                       const std::vector<int>& vi1,
                       const std::vector<double>& vd1)
{
  Vector<int> dc1 = vc1;
  Vector<int> di1 = vi1;
  Vector<double> dd1 = vd1;

  size_t sz = dc1.size();
  assert( sz == di1.size() );
  assert( sz == dd1.size() );
    
  Vector<IndexT> d_indx(sz, 0);
  Vector<IndexT> d_keys(sz, 0);
  Vector<IndexT> d_vals(sz, 0);

  size_t ncols = 3;
  size_t& nrows = sz;

  Vector<void*> d_cols(ncols, nullptr);
  Vector<int>   d_types(ncols, 0);

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

  Vector<double> d_outd(sz, 0);

  c_agg.dtype = GDF_FLOAT64;
  c_agg.data = dd1.data().get();
  c_agg.size = nrows;

  c_vout.dtype = GDF_FLOAT64;
  c_vout.data = d_outd.data().get();
  c_vout.size = nrows;

  size_t n_group = 0;
  int flag_sorted = 0;

  std::cout<<"aggregate = avg on column:\n";
  print_v(dd1, std::cout);

  //input
  //{
  gdf_context ctxt{0, GDF_SORT, 0};
  std::vector<gdf_column*> v_pcols(ncols);
  for(int i = 0; i < ncols; ++i)
    {
      v_pcols[i] = &v_gdf_cols[i];
    }
  gdf_column** cols = &v_pcols[0];//pointer semantic (2);
  //}

  //output:
  //{
  Vector<int32_t> d_vc_out(nrows);
  Vector<int32_t> d_vi_out(nrows);
  Vector<double> d_vd_out(nrows);
    
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
  for(int i=0; i<ncols; ++i)
    h_cols_out[i] = &v_gdf_cols_out[i];//
  
  gdf_column** cols_out = &h_cols_out[0];//pointer semantics (2)

  d_keys.assign(nrows, 0);
  gdf_column c_indx;
  c_indx.data = d_keys.data().get();
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
  std::cout<<"selected rows:\n";
  print_v(d_vc_out, n_group, std::cout);
  print_v(d_vi_out, n_group, std::cout);
  print_v(d_vd_out, n_group, std::cout);
    
  size_t n_keys = n_group;
  size_t n_vals = n_group;
    
  std::cout<<"multi-col generic group-by (average):\n";
  std::cout<<"indices of grouped-by, new keys end position: "<<n_keys<<";\n";
  std::cout<<"indices of grouped-by, new vals end position: "<<n_vals<<";\n";
    
  std::cout<<"grouped-by keys:\n";
  print_v(d_keys, n_keys, std::cout);

  std::cout<<"grouped-by vals:\n";
  print_v(d_outd, n_vals, std::cout);
}

int main(void)
{
  {
    //okay:
    //
    std::vector<int> vc1{1,1,1};
    std::vector<int> vi1{1,1,0};
    std::vector<double> vd1{12., 11., 17.};

    Vector<int> dc1 = vc1;
    Vector<int> di1 = vi1;
    Vector<double> dd1 = vd1;

    size_t nrows = dc1.size();
    assert( nrows == di1.size() );
    assert( nrows == dd1.size() );
    
    Vector<int> dv(nrows, 0);
    ///multi_col_order_by(nrows, tv1, dv);//okay

    size_t ncols = 3;
 
    std::vector<void*> v_cols{static_cast<void*>(dc1.data().get()),
    	                      static_cast<void*>(di1.data().get()),
    	                      static_cast<void*>(dd1.data().get())};
    std::vector<int> v_types{static_cast<int>(GDF_INT32),
    	                     static_cast<int>(GDF_INT32),
    	                     static_cast<int>(GDF_FLOAT64)};

    
    Vector<void*> d_cols(ncols, nullptr);
    Vector<int>   d_types(ncols, 0);
    Vector<size_t> d_indx(nrows, 0);
    
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

    gdf_column* h_columns = &v_gdf_cols[0];
    void** d_col_data = d_cols.data().get();
    int* d_col_types = d_types.data().get();
    size_t* ptr_dv = d_indx.data().get();

    gdf_order_by(nrows, h_columns, ncols, d_col_data, d_col_types, ptr_dv);
    
  
    std::cout<<"multisort order:\n";
    print_v(d_indx, std::cout);

    //should return:
    //multisort order:
    //2,1,0,
  }
  {
    //okay:
    //
    std::vector<int> vc1{1,1,1,1,1,1};
    std::vector<int> vi1{1,3,3,5,5,0};
    std::vector<double> vd1{12., 13., 13., 17., 17., 17};

    std::cout<<"multi-column group-by experiments:\n";

    test_gb_count_api_2(vc1, vi1, vd1);//okay

    test_gb_sum_api_2(vc1, vi1, vd1);//okay

    test_gb_avg_api_2(vc1, vi1, vd1);//okay
    
    test_gb_min_api_2(vc1, vi1, vd1);//okay
    
    test_gb_max_api_2(vc1, vi1, vd1);//okay
    //}
  }

  {
    std::cout<<"Filtering experiments:\n";
    f_test_multi_filter();
  }
  
  std::cout << "Done!" << std::endl;
  return 0;
}
