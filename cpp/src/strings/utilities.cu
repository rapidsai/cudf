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

#include <cstring>
#include <cudf/column/column_device_view.cuh>
#include <utilities/error_utils.hpp>
#include "./utilities.h"

#include <rmm/rmm.h>

namespace cudf 
{
namespace strings
{

// Used to build a temporary string_view object from a single host string.
// It will create a single piece of device memory that includes
// the string_view instance and the string data.
std::unique_ptr<cudf::string_view, std::function<void(cudf::string_view*)>>
    string_from_host( const char* str, cudaStream_t stream )
{
    if( !str )
        return nullptr;
    size_type length = (size_type)std::strlen(str);
    size_type bytes = sizeof(cudf::string_view) + length;

    char* d_data;
    RMM_TRY(RMM_ALLOC( &d_data, bytes, stream ));
    char* d_str = d_data + sizeof(cudf::string_view);
    cudf::string_view tmp{d_str,length};
    std::vector<char> h_data(bytes);
    memcpy( h_data.data(), &tmp, sizeof(cudf::string_view) );
    memcpy( h_data.data() + sizeof(cudf::string_view), str, length );
    CUDA_TRY(cudaMemcpyAsync( d_data, h_data.data(), bytes,
                              cudaMemcpyHostToDevice, stream ));
    CUDA_TRY(cudaStreamSynchronize(stream));
    auto deleter = [](cudf::string_view* sv) { RMM_FREE(sv,0); };
    return std::unique_ptr<cudf::string_view,
        decltype(deleter)>{reinterpret_cast<cudf::string_view*>(d_data),deleter};
}

rmm::device_buffer create_string_array_from_column(
    strings_column_handler handler,
    cudaStream_t stream )
{
    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(handler.parent_column(),stream);
    auto d_column = *strings_column;

    auto count = handler.size();
    rmm::device_buffer buffer( count*sizeof(cudf::string_view), stream, handler.memory_resource() );
    cudf::string_view* d_strings = reinterpret_cast<cudf::string_view*>(buffer.data());
    thrust::for_each_n( execpol->on(stream), 
        thrust::make_counting_iterator<size_type>(0), count,
        [d_column, d_strings] __device__ (size_type idx) {
            if( d_column.nullable() && d_column.is_null(idx) )
                d_strings[idx] = cudf::string_view(nullptr,0);
            else
                d_strings[idx] = d_column.element<cudf::string_view>(idx);
        });

    return buffer;
}

} // namespace strings
} // namespace cudf