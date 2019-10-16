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

#include <cudf/column/column_factories.hpp>
#include <tests/utilities/column_utilities.cuh>

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
    thrust::host_vector<thrust::pair<const char*,size_type> > strings(count);
    cudf::size_type offset = 0;
    for( cudf::size_type idx=0; idx < count; ++idx )
    {
        const char* str = h_strings[idx];
        if( !str )
            strings[idx] = thrust::pair<const char*,size_type>{nullptr,0};
        else
        {
            cudf::size_type length = (cudf::size_type)strlen(str);
            memcpy( h_buffer.data() + offset, str, length );
            strings[idx] = thrust::pair<const char*,size_type>{d_buffer.data().get()+offset,length};
            offset += length;
        }
    }
    rmm::device_vector<thrust::pair<const char*,size_type>> d_strings(strings);
    cudaMemcpy( d_buffer.data().get(), h_buffer.data(), memsize, cudaMemcpyHostToDevice );
    return cudf::make_strings_column( d_strings );
}

void expect_strings_equal(cudf::column_view strings_column, const std::vector<const char*>& h_expected )
{
    auto results_view = cudf::strings_column_view(strings_column);
    auto d_expected = cudf::test::create_strings_column(h_expected);
    auto expected_view = cudf::strings_column_view(d_expected->view());
    cudf::test::expect_columns_equal(results_view.parent(), d_expected->view());
}

void expect_strings_empty(cudf::column_view strings_column)
{
    EXPECT_EQ(STRING, strings_column.type().id());
    EXPECT_EQ(0,strings_column.size());
    EXPECT_EQ(0,strings_column.null_count());
    EXPECT_EQ(0,strings_column.num_children());
}

}  // namespace test
}  // namespace cudf
