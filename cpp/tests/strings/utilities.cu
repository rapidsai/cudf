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

#include "./utilities.h"

#include <cudf/strings/strings_column_factories.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>

#include <cstring>
#include <thrust/execution_policy.h>
#include <thrust/equal.h>

#include <gmock/gmock.h>

namespace cudf {
namespace test {

//
std::unique_ptr<cudf::column> create_strings_column( const std::vector<const char*>& h_strings )
{
    cudf::size_type memsize = 0;
    for( auto itr=h_strings.begin(); itr!=h_strings.end(); ++itr )
        memsize += *itr ? (cudf::size_type)strlen(*itr) : 0;
    if( memsize==0 && h_strings.size() )
        memsize = 1; // prevent vectors from being null in all empty-string case
    cudf::size_type count = (cudf::size_type)h_strings.size();
    thrust::host_vector<char> h_buffer(memsize);
    thrust::device_vector<char> d_buffer(memsize);
    thrust::host_vector<thrust::pair<const char*,size_t> > strings(count);
    cudf::size_type offset = 0;
    for( cudf::size_type idx=0; idx < count; ++idx )
    {
        const char* str = h_strings[idx];
        if( !str )
            strings[idx] = thrust::pair<const char*,size_t>{nullptr,0};
        else
        {
            cudf::size_type length = (cudf::size_type)strlen(str);
            memcpy( h_buffer.data() + offset, str, length );
            strings[idx] = thrust::pair<const char*,size_t>{d_buffer.data().get()+offset,(size_t)length};
            offset += length;
        }
    }
    rmm::device_vector<thrust::pair<const char*,size_t>> d_strings(strings);
    cudaMemcpy( d_buffer.data().get(), h_buffer.data(), memsize, cudaMemcpyHostToDevice );
    return cudf::make_strings_column( d_strings );
}

struct compare_strings_fn
{
    __device__ bool operator()(int lidx, int ridx)
    {
        if( (d_lhs.nullable() && d_lhs.is_null(lidx)) ||
            (d_rhs.nullable() && d_rhs.is_null(ridx)) )
            return d_lhs.is_null(lidx)==d_rhs.is_null(ridx);
        cudf::strings::string_view lstr = d_lhs.element<cudf::strings::string_view>(lidx);
        cudf::strings::string_view rstr = d_rhs.element<cudf::strings::string_view>(ridx);
        return lstr.compare(rstr)==0;
    }
    column_device_view d_lhs;
    column_device_view d_rhs;
};

//
void expect_strings_columns_equal(cudf::strings_column_view lhs, cudf::strings_column_view rhs)
{
  EXPECT_EQ(lhs.size(), rhs.size());
  EXPECT_EQ(lhs.null_count(), rhs.null_count());

  // this almost works
  //auto d_lhs = cudf::table_device_view::create(table_view{{lhs.parent()}});
  //auto d_rhs = cudf::table_device_view::create(table_view{{rhs.parent()}});
  //EXPECT_TRUE(
  //    thrust::equal(thrust::device, thrust::make_counting_iterator(0),
  //                  thrust::make_counting_iterator(lhs.size()),
  //                  thrust::make_counting_iterator(0),
  //                  cudf::exp::row_equality_comparator<true>{*d_lhs, *d_rhs}));
  //CUDA_TRY(cudaDeviceSynchronize());

  auto col_lhs = column_device_view::create(lhs.parent());
  auto col_rhs = column_device_view::create(rhs.parent());

  EXPECT_TRUE(
      thrust::equal(thrust::device, thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>((int)lhs.size()),
                    thrust::make_counting_iterator<int>(0),
                    compare_strings_fn{*col_lhs,*col_rhs}));
}

}  // namespace test
}  // namespace cudf
