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
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/detail/utilities.cuh>

namespace cudf
{
namespace strings
{
namespace detail
{

/**
 * @brief Returns a new strings column using the specified Filter to select
 * strings from the lhs column or the rhs column.
 *
 * ```
 * output[i] = filter_fn(i) ? lhs(i) : rhs(i)
 * ```
 *
 * @tparam Filter Functor that takes an index and returns a boolean.
 *
 * @param lhs Strings instance to copy if Filter is true
 * @param rhs Strings instance to copy if Filter is false
 * @param filter_fn Called to determine which column to retrieve an entry for a specific row.
 * @param mr Resource for allocating device memory.
 * @param stream CUDA stream to use kernels in this method.
 * @return New strings column.
 */
template<typename Filter>
std::unique_ptr<cudf::column> copy_if_else( strings_column_view const& lhs,
                                            strings_column_view const& rhs,
                                            Filter filter_fn,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                            cudaStream_t stream=0 )
{
    CUDF_EXPECTS(lhs.size() == rhs.size(), "Both columns must be the same size");
    auto strings_count = lhs.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);

    auto execpol = rmm::exec_policy(stream);
    auto lhs_column = column_device_view::create(lhs.parent(),stream);
    auto d_lhs = *lhs_column;
    auto rhs_column = column_device_view::create(rhs.parent(),stream);
    auto d_rhs = *rhs_column;
    // create null mask
    rmm::device_buffer null_mask;
    size_type null_count = 0;
    if( lhs.has_nulls() || rhs.has_nulls() )
    {
        auto valid_mask = cudf::experimental::detail::valid_if(
            thrust::make_counting_iterator<size_type>(0),
            thrust::make_counting_iterator<size_type>(strings_count),
            [d_lhs, d_rhs, filter_fn] __device__ (size_type idx) { return filter_fn(idx) ? d_lhs.is_valid(idx) : d_rhs.is_valid(idx); },
            stream, mr );
        null_mask = valid_mask.first;
        null_count = valid_mask.second;
    }

    // build offsets column
    auto offsets_transformer = [d_lhs, d_rhs, filter_fn] __device__ (size_type idx) {
            bool bfilter = filter_fn(idx);
            size_type bytes = 0;
            if( bfilter ? d_lhs.is_valid(idx) : d_rhs.is_valid(idx) )
               bytes = bfilter ? d_lhs.element<string_view>(idx).size_bytes() :
                                 d_rhs.element<string_view>(idx).size_bytes();
            return bytes;
        };
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<size_type>(0), offsets_transformer );
    auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                                    offsets_transformer_itr+strings_count,
                                                    mr, stream);
    auto d_offsets = offsets_column->view().template data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
    auto chars_column = create_chars_child_column( strings_count, null_count, bytes, mr, stream );
    auto d_chars = chars_column->mutable_view().template data<char>();
    // fill in chars
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count,
        [d_lhs, d_rhs, filter_fn, d_offsets, d_chars] __device__(size_type idx){
            auto bfilter = filter_fn(idx);
            if( bfilter ? d_lhs.is_null(idx) : d_rhs.is_null(idx) )
               return;
            string_view d_str = bfilter ? d_lhs.element<string_view>(idx) : d_rhs.element<string_view>(idx);
            memcpy(d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes() );
        });

    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}


} // namespace detail
} // namespace strings
} // namespace cudf
