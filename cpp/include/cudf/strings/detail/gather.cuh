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
#pragma once

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/valid_if.cuh>
#include "./utilities.hpp"
#include "./utilities.cuh"

namespace cudf
{
namespace strings
{
namespace detail
{

/**
 * @brief Returns a new strings column using the specified indices to select
 * elements from the `strings` column.
 *
 * ```
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * map = [0, 2]
 * s2 = gather( s1, map )
 * s2 is ["a", "c"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param begin Start of index iterator.
 * @param end End of index iterator.
 * @param ignore_out_of_bounds If true, indices outside the column's range are ignored.
 * @param mr Resource for allocating device memory.
 * @param stream CUDA stream to use kernels in this method.
 * @return New strings column containing the gather strings only
 */
template<typename MapIterator>
std::unique_ptr<cudf::column> gather( strings_column_view const& strings,
                                      MapIterator begin, MapIterator end,
                                      bool ignore_out_of_bounds,
                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                      cudaStream_t stream=0 )
{
    auto strings_count = std::distance(begin, end);
    if( strings_count == 0 )
        return std::make_unique<column>( data_type{STRING}, 0,
                                         rmm::device_buffer{0,stream,mr},
                                         rmm::device_buffer{0,stream,mr}, 0 );
    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column;

    // create null mask
    rmm::device_buffer null_mask;
    auto valid_mask = cudf::experimental::detail::valid_if( begin, end,
        [d_strings] __device__ (size_type idx) { return !d_strings.is_null(idx); },
        stream, mr );
    auto null_count = valid_mask.second;
    if( null_count > 0 )
        null_mask = valid_mask.first;

    // build offsets column
    auto offsets_transformer = [d_strings] __device__ (size_type idx) {
            size_type bytes = 0;
            if( !d_strings.is_null(idx) ) // handles offset
                bytes = d_strings.element<string_view>(idx).size_bytes();
            return bytes;
        };
    auto offsets_transformer_itr = thrust::make_transform_iterator( begin, offsets_transformer );
    auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                                    offsets_transformer_itr+strings_count,
                                                    mr, stream);
    auto offsets_view = offsets_column->view();
    auto d_offsets = offsets_view.template data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
    auto chars_column = create_chars_child_column( strings_count, null_count, bytes, mr, stream );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.template data<char>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count,
        [d_strings, begin, d_offsets, d_chars] __device__(size_type idx){
            size_type index = begin[idx];
            if( d_strings.is_null(index) )
                return;
            string_view d_str = d_strings.element<string_view>(index);
            memcpy(d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes() );
        });

    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}

} // namespace detail
} // namespace strings
} // namespace cudf
