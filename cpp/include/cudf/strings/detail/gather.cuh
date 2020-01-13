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

#include <cudf/detail/copy.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/detail/utilities.cuh>

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
 * Caller must update the validity mask in the output column.
 *
 * ```
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * map = [0, 2]
 * s2 = gather<true>( s1, map.begin(), map.end() )
 * s2 is ["a", "c"]
 * ```
 *
 * @tparam NullifyOutOfBounds If true, indices outside the column's range are nullified.
 * @tparam MapIterator Iterator for retrieving integer indices of the column.
 *
 * @param strings Strings instance for this operation.
 * @param begin Start of index iterator.
 * @param end End of index iterator.
 * @param mr Resource for allocating device memory.
 * @param stream CUDA stream to use kernels in this method.
 * @return New strings column containing the gathered strings.
 */
template<bool NullifyOutOfBounds, typename MapIterator>
std::unique_ptr<cudf::column> gather( strings_column_view const& strings,
                                      MapIterator begin, MapIterator end,
                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                      cudaStream_t stream=0 )
{
    auto output_count = std::distance(begin, end);
    auto strings_count = strings.size();
    if( output_count == 0 || strings_count == 0 )
        return make_empty_strings_column(mr,stream);

    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column;

    mask_state mstate = mask_state::UNINITIALIZED;
    if( NullifyOutOfBounds ) {
      mstate = ALL_NULL;
    }
    // create null mask -- caller must update this

    auto null_mask  = [&]{
       if(strings.parent().nullable() or NullifyOutOfBounds)
          return create_null_mask(output_count, mstate, stream, mr);
       else
          return rmm::device_buffer{};
    }();

    // build offsets column
    auto offsets_transformer = [d_strings, strings_count] __device__ (size_type idx) {
            if( NullifyOutOfBounds && ((idx<0) || (idx >= strings_count)) )
                return 0;
             if( d_strings.is_null(idx) )
                return 0;
            return d_strings.element<string_view>(idx).size_bytes();
        };
    auto offsets_transformer_itr = thrust::make_transform_iterator( begin, offsets_transformer );
    auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                                    offsets_transformer_itr+output_count,
                                                    mr, stream);
    auto offsets_view = offsets_column->view();
    auto d_offsets = offsets_view.template data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[output_count];
    auto chars_column = create_chars_child_column( output_count, 0, bytes, mr, stream );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.template data<char>();
    // fill in chars
    auto gather_chars = [d_strings, begin, strings_count, d_offsets, d_chars] __device__(size_type idx){
            auto index = begin[idx];
            if( NullifyOutOfBounds && ((index<0) || (index >= strings_count)) )
                return;
            if( d_strings.is_null(index) )
                return;
            string_view d_str = d_strings.element<string_view>(index);
            memcpy(d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes() );
        };
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), output_count, gather_chars);

    return make_strings_column(output_count, std::move(offsets_column), std::move(chars_column),
                               UNKNOWN_NULL_COUNT, std::move(null_mask), stream, mr);
}

/**
 * @brief Returns a new strings column using the specified indices to select
 * elements from the `strings` column.
 *
 * Caller must update the validity mask in the output column.
 *
 * ```
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * map = [0, 2]
 * s2 = gather( s1, map.begin(), map.end(), true )
 * s2 is ["a", "c"]
 * ```
 *
 * @tparam MapIterator Iterator for retrieving integer indices of the column.
 *
 * @param strings Strings instance for this operation.
 * @param begin Start of index iterator.
 * @param end End of index iterator.
 * @param nullify_out_of_bounds If true, indices outside the column's range are nullified.
 * @param mr Resource for allocating device memory.
 * @param stream CUDA stream to use kernels in this method.
 * @return New strings column containing the gathered strings.
 */
template<typename MapIterator>
std::unique_ptr<cudf::column> gather( strings_column_view const& strings,
                                      MapIterator begin, MapIterator end,
                                      bool nullify_out_of_bounds,
                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                      cudaStream_t stream=0 )
{
    if( nullify_out_of_bounds )
        return gather<true>( strings, begin, end, mr, stream );
    return gather<false>( strings, begin, end, mr, stream );
}


} // namespace detail
} // namespace strings
} // namespace cudf
