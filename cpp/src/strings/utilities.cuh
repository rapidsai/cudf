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

#include <cuda_runtime.h>
#include <cudf/strings/string_view.cuh>
#include <cudf/column/column_view.hpp>

#include <cstring>
#include <thrust/scan.h>

namespace cudf
{
namespace strings
{
namespace detail
{

/**
 * @brief This utility will copy the argument string's data into
 * the provided buffer.
 *
 * @param buffer Device buffer to copy to.
 * @param d_string String to copy.
 * @return Points to the end of the buffer after the copy.
 */
__device__ inline char* copy_string( char* buffer, const string_view& d_string )
{
    memcpy( buffer, d_string.data(), d_string.size_bytes() );
    return buffer + d_string.size_bytes();
}

/**
 * @brief Create an offsets column to be a child of a strings column.
 * This will set the offsets values by executing scan on the provided
 * Iterator.
 *
 * @tparam Iterator Used as input to scan to set the offset values.
 * @param begin The beginning of the input sequence
 * @param end The end of the input sequence
 * @param mr Memory resource to use.
 * @stream Stream to use for any kernel calls.
 * @return offsets child column for strings column
 */
template <typename Iterator>
std::unique_ptr<column> make_offsets( Iterator begin, Iterator end,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0)
{
    CUDF_EXPECTS(begin <= end, "Invalid iterator range");
    auto count = thrust::distance(begin, end);
    auto offsets_column = make_numeric_column(
          data_type{INT32}, count + 1, mask_state::UNALLOCATED, stream, mr);
    auto offsets_view = offsets_column->mutable_view();
    auto d_offsets = offsets_view.template data<int32_t>();
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream), begin, end,
                           d_offsets+1);
    CUDA_TRY(cudaMemsetAsync(d_offsets, 0, sizeof(int32_t), stream));
    return offsets_column;
}

} // namespace detail
} // namespace strings
} // namespace cudf
