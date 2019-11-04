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

#include <bitmask/legacy/valid_if.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include "./utilities.hpp"
#include "./utilities.cuh"

#include <algorithm>
#include <numeric>
#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform_scan.h>
#include <thrust/logical.h>

namespace cudf
{
namespace strings
{
namespace detail
{

//
std::unique_ptr<cudf::column> concatenate( const std::vector<strings_column_view>& strings_columns,
                                           const std::string separator = "",
                                           const char* narep=nullptr,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                           cudaStream_t stream=0 )
{
    auto num_columns = strings_columns.size();
    if( num_columns==1 ) // single column returns a copy
        return std::make_unique<column>(strings_columns[0].parent(),stream,mr);

    auto first_column = column_device_view::create(strings_columns[0].parent(),stream);
    auto strings_count = first_column->size();
    if( !std::all_of(strings_columns.begin(),strings_columns.end(),
        [strings_count] (strings_column_view view) { return strings_count==view.size(); }) )
    {
        CUDF_FAIL( "concatenate requires all columns have an equal number of rows");
    }
    if( strings_count == 0 )
        return detail::make_empty_strings_column(mr,stream);

    auto execpol = rmm::exec_policy(stream);
    auto separator_ptr = detail::string_from_host(separator.c_str(), stream);
    auto d_separator = *separator_ptr;
    auto narep_ptr = detail::string_from_host(narep, stream);
    string_view d_narep(nullptr,0);
    if( narep_ptr )
        d_narep = *narep_ptr;

    // Create device views from the strings columns.
    //
    // First calculate the size of memory needed to hold the
    // column_device_views. This is done by calling extent()
    // for each of the column_views of the strings_columns.
    size_t views_size_bytes =
        std::accumulate(strings_columns.begin(), strings_columns.end(), 0,
            [](size_type init, strings_column_view col) {
                return init + column_device_view::extent(col.parent());
            });
    // Allocate the device memory to be used in the device methods.
    // We need to pass this down when creating the column_device_views
    // so they can be resolved to point to any child objects.
    rmm::device_buffer d_columns_memory{views_size_bytes,stream,mr};
    column_device_view* d_columns = reinterpret_cast<column_device_view*>(d_columns_memory.data());
    column_device_view* d_column = d_columns; // point to the first one
    // A buffer of CPU memory is created to hold the column_device_view
    // objects and then copied to device memory at the d_columns pointer.
    // But each column_device_view instance may have child objects which
    // require setting an internal device pointer before being copied from
    // CPU to device.
    {
        std::vector<int8_t> h_buffer(views_size_bytes);
        column_device_view* h_column = reinterpret_cast<column_device_view*>(h_buffer.data());
        // The beginning of the memory must be the fixed-sized column_device_view
        // objects in order for d_columns to be used as array. Therefore, any
        // child data is assigned to the end of this array.
        int8_t* h_end = reinterpret_cast<int8_t*>(h_column + num_columns);
        int8_t* d_end = reinterpret_cast<int8_t*>(d_column + num_columns);
        // Create the column_device_view from each column within the CPU memory
        // array. Any column child data should be copied into h_end and any
        // internal pointers should be set using d_end.
        for( auto itr = strings_columns.begin(); itr != strings_columns.end(); ++itr )
        {
            auto column = itr->parent();
            // convert the column_view into column_device_view
            new(h_column) column_device_view(column,reinterpret_cast<ptrdiff_t>(h_end),reinterpret_cast<ptrdiff_t>(d_end));
            h_column++; // next element in array
            // point to the next chunk of memory for use of the children of the next column
            auto col_child_data_size = (column_device_view::extent(column) - sizeof(column_device_view));
            h_end += col_child_data_size;
            d_end += col_child_data_size;
        }
        CUDA_TRY(cudaMemcpyAsync(d_columns, h_buffer.data(),
                                 views_size_bytes, cudaMemcpyDefault, stream));
        CUDA_TRY(cudaStreamSynchronize(stream)); // h_buffer is about to be destroyed
    }

    // create resulting null mask
    auto valid_mask = detail::make_null_mask(strings_count,
        [d_columns, num_columns, d_narep] __device__ (size_type idx) {
            bool null_element = thrust::any_of( thrust::seq, d_columns, d_columns+num_columns,
                [idx] (column_device_view col) { return col.is_null(idx);});
            return( !null_element || !d_narep.is_null() );
        },
        mr, stream );
    rmm::device_buffer null_mask = valid_mask.first;
    auto null_count = valid_mask.second;

    // build offsets column by computing sizes of each string in the output
    auto offsets_transformer = [d_columns, num_columns, d_separator, d_narep] __device__ (size_type idx) {
            size_type bytes = 0;
            for( size_type col_idx=0; col_idx < num_columns; ++col_idx )
            {
                auto d_column = d_columns[col_idx];
                if( d_column.is_null(idx) )
                {
                    if( d_narep.is_null() )
                        return 0; // null entry in result
                    bytes += d_narep.size_bytes();
                }
                else
                    bytes += d_column.element<string_view>(idx).size_bytes();
                // separator only in between elements
                if( col_idx+1 < num_columns )
                    bytes += d_separator.size_bytes();
            }
            return bytes;
        };
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<size_type>(0), offsets_transformer );
    auto offsets_column = detail::make_offsets_child_column(offsets_transformer_itr,
                                               offsets_transformer_itr+strings_count,
                                               mr, stream);
    auto offsets_view = offsets_column->view();
    auto d_results_offsets = offsets_view.data<int32_t>();

    // create the chars column
    size_type bytes = thrust::device_pointer_cast(d_results_offsets)[strings_count];
    auto chars_column = strings::detail::create_chars_child_column( strings_count, null_count, bytes, mr, stream );
    // fill the chars column
    auto chars_view = chars_column->mutable_view();
    auto d_results_chars = chars_view.data<char>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count,
        [d_columns, num_columns, d_separator, d_narep, d_results_offsets, d_results_chars] __device__(size_type idx){
            bool null_element = thrust::any_of( thrust::seq, d_columns, d_columns+num_columns,
                    [idx] (column_device_view col) { return col.is_null(idx);});
            if( null_element && d_narep.is_null() )
                return; // do not write to buffer at all if any element is null
            size_type offset = d_results_offsets[idx];
            char* d_buffer = d_results_chars + offset;
            for( size_type col_idx=0; col_idx < num_columns; ++col_idx )
            {
                auto d_column = d_columns[col_idx];
                if( d_column.nullable() && d_column.is_null(idx) )
                    d_buffer = detail::copy_string(d_buffer, d_narep);
                else
                {
                    string_view d_str = d_column.element<string_view>(idx);
                    d_buffer = detail::copy_string(d_buffer, d_str);
                }
                // separator only in between elements
                if( col_idx+1 < num_columns )
                    d_buffer = detail::copy_string(d_buffer, d_separator);
            }
        });

    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}

//
std::unique_ptr<cudf::column> join_strings( strings_column_view strings,
                                            const std::string separator = "",
                                            const char* narep=nullptr,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                            cudaStream_t stream=0 )
{
    auto strings_count = strings.size();
    if( strings_count == 0 )
        return detail::make_empty_strings_column(mr,stream);

    auto execpol = rmm::exec_policy(stream);
    auto separator_ptr = detail::string_from_host(separator.c_str(), stream);
    auto d_separator = *separator_ptr;
    auto narep_ptr = detail::string_from_host(narep, stream);
    string_view d_narep(nullptr,0);
    if( narep_ptr )
        d_narep = *narep_ptr;

    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;
    auto d_offsets = strings.offsets().data<int32_t>();

    // create an offsets array for building the output memory layout
    rmm::device_vector<size_type> output_offsets(strings_count+1);
    auto d_output_offsets = output_offsets.data().get();
    // using inclusive-scan to compute last entry which is the total size
    thrust::transform_inclusive_scan( execpol->on(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count),
        d_output_offsets + 1,
        [d_column, d_separator, d_narep] __device__ (size_type idx) {
            size_type bytes = 0;
            if( d_column.is_null(idx) )
            {
                if( d_narep.is_null() )
                    return 0; // skip nulls
                bytes += d_narep.size_bytes();
            }
            else
                bytes += d_column.element<string_view>(idx).size_bytes();
            if( (idx+1) < d_column.size() )
                bytes += d_separator.size_bytes();
            return bytes;
        },
        thrust::plus<size_type>());
    CUDA_TRY(cudaMemsetAsync(d_output_offsets, 0, sizeof(size_type), stream));
    // total size is the last entry
    size_type bytes = output_offsets.back();

    // build offsets column (only 1 string so 2 offset entries)
    auto offsets_column = make_numeric_column( data_type{INT32}, 2, mask_state::UNALLOCATED,
                                               stream, mr );
    auto offsets_view = offsets_column->mutable_view();
    // set the first entry to 0 and the last entry to bytes
    int32_t new_offsets[] = {0, bytes};
    CUDA_TRY(cudaMemcpyAsync(offsets_view.data<int32_t>(), new_offsets,
                             sizeof(new_offsets), cudaMemcpyHostToDevice,stream));

    // build null mask
    // only one entry so it is either all valid or all null
    size_type null_count = 0;
    rmm::device_buffer null_mask; // init to null null-mask
    if( strings.null_count()==strings_count )
    {
        null_mask = create_null_mask(1,cudf::ALL_NULL,stream,mr);
        null_count = 1;
    }
    auto chars_column = detail::create_chars_child_column( strings_count, null_count, bytes, mr, stream );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<char>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count,
        [d_column, d_separator, d_narep, d_output_offsets, d_chars] __device__(size_type idx){
            size_type offset = d_output_offsets[idx];
            char* d_buffer = d_chars + offset;
            if( d_column.is_null(idx) )
            {
                if( d_narep.is_null() )
                    return; // do not write to buffer if element is null (including separator)
                d_buffer = detail::copy_string(d_buffer, d_narep);
            }
            else
            {
                string_view d_str = d_column.element<string_view>(idx);
                d_buffer = detail::copy_string(d_buffer, d_str);
            }
            if( (idx+1) < d_column.size() )
                d_buffer = detail::copy_string(d_buffer, d_separator);
        });

    return make_strings_column(1, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}

} // namespace detail

// APIs

std::unique_ptr<cudf::column> concatenate( const std::vector<strings_column_view>& strings_columns,
                                           const std::string separator,
                                           const char* narep,
                                           rmm::mr::device_memory_resource* mr)
{
    return detail::concatenate(strings_columns, separator, narep, mr);
}

std::unique_ptr<cudf::column> join_strings( strings_column_view strings,
                                            const std::string separator,
                                            const char* narep,
                                            rmm::mr::device_memory_resource* mr )
{
    return detail::join_strings(strings, separator, narep, mr);
}

} // namespace strings
} // namespace cudf
