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

#include <cudf/strings/strings_column_factories.hpp>
#include "./utilities.h"

#include <cstring>

namespace cudf {
namespace test {

//
std::unique_ptr<cudf::column> create_strings_column( const std::vector<const char*>& h_strings )
{
    cudf::size_type memsize = 0;
    for( auto itr=h_strings.begin(); itr!=h_strings.end(); ++itr )
        memsize += *itr ? (cudf::size_type)strlen(*itr) : 0;
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

}  // namespace test
}  // namespace cudf
