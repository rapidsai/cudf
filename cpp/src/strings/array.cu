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

#include <bitmask/valid_if.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/strings_column_handler.hpp>
#include <cudf/strings/string_view.cuh>
#include "./utilities.h"

#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace cudf 
{
namespace strings
{

// new strings column from subset of this strings instance
std::unique_ptr<cudf::column> sublist( strings_column_handler handler,
                                       size_type start, size_type end,
                                       size_type step, cudaStream_t stream )
{
    if( step <= 0 )
        step = 1;
    size_type count = handler.size();
    if( end < 0 || end > count )
        end = count;
    if( start < 0 || start > end )
        throw std::invalid_argument("invalid start parameter");
    count = (end - start)/step +1;
    //
    auto execpol = rmm::exec_policy(stream);
    // build indices
    thrust::device_vector<size_type> indices(count);
    thrust::sequence( execpol->on(stream), indices.begin(), indices.end(), start, step );
    // create a column_view as a wrapper of these indices
    column_view indices_view( data_type{INT32}, count, indices.data().get(), nullptr, 0 );
    // build a new strings column from the indices
    return gather(handler, indices_view);
}

// return new strings column with strings from this instance as specified by the indices
std::unique_ptr<cudf::column> gather( strings_column_handler handler,
                                      column_view gather_map, cudaStream_t stream )
{
    size_type count = gather_map.size();
    auto d_indices = gather_map.data<int32_t>();

    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(handler.parent_column(),stream);
    auto d_column = *strings_column;
    auto d_offsets = handler.offsets_column().data<int32_t>();

    // build offsets column
    auto offsets_column = make_numeric_column( data_type{INT32}, count, mask_state::UNALLOCATED,
                                               stream, handler.memory_resource() );
    auto offsets_view = offsets_column->mutable_view();
    auto d_new_offsets = offsets_view.data<int32_t>();
    // create new offsets array
    thrust::transform_inclusive_scan( execpol->on(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(count),
        d_new_offsets,
        [d_column, d_offsets, d_indices] __device__ (size_type idx) {
            size_type index = d_indices[idx];
            if( d_column.nullable() && d_column.is_null(index) )
                return 0;
            size_type offset = index ? d_offsets[index-1] : 0;
            return d_offsets[index] - offset;
        },
        thrust::plus<int32_t>());

    // build null mask
    auto valid_mask = valid_if( static_cast<const bit_mask_t*>(nullptr),
        [d_column, d_indices] __device__ (size_type idx) {
            return !d_column.nullable() || !d_column.is_null(d_indices[idx]);
        },
        count, stream );
    auto null_count = valid_mask.second;
    auto null_size = gdf_valid_allocation_size(count);
    rmm::device_buffer null_mask(valid_mask.first,null_size); // does deep copy
    RMM_TRY( RMM_FREE(valid_mask.first,stream) ); // TODO valid_if to return device_buffer in future

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_new_offsets)[count-1]; // this may not be stream friendly
    auto chars_column = make_numeric_column( data_type{INT8}, bytes, mask_state::UNALLOCATED,
                                             stream, handler.memory_resource() );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<int8_t>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), count,
        [d_column, d_indices, d_new_offsets, d_chars] __device__(size_type idx){
            // place individual strings
            if( d_column.nullable() && d_column.is_null(idx) )
                return;
            string_view d_str = d_column.element<string_view>(d_indices[idx]);
            size_type offset = (idx ? d_new_offsets[idx-1] : 0);
            memcpy(d_chars + offset, d_str.data(), d_str.size() );
        });

  // build children vector
  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(offsets_column));
  children.emplace_back(std::move(chars_column));

  return std::make_unique<column>(
        data_type{STRING}, 0, rmm::device_buffer{0,stream,handler.memory_resource()},
        null_mask, null_count,
        std::move(children));
}

// return sorted version of the given strings column
std::unique_ptr<cudf::column> sort( strings_column_handler handler,
                                    strings_column_handler::sort_type stype,
                                    bool ascending, bool nullfirst, cudaStream_t stream )
{
    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(handler.parent_column(), stream);
    auto d_column = *strings_column;

    // lets sort indices
    size_type count = handler.size();
    thrust::device_vector<size_type> indices(count);
    thrust::sequence( execpol->on(stream), indices.begin(), indices.end() );
    thrust::sort( execpol->on(stream), indices.begin(), indices.end(),
        [d_column, stype, ascending, nullfirst] __device__ (size_type lhs, size_type rhs) {
            bool lhs_null{d_column.nullable() && d_column.is_null(lhs)};
            bool rhs_null{d_column.nullable() && d_column.is_null(rhs)};
            if( lhs_null || rhs_null )
                return (nullfirst ? !rhs_null : !lhs_null);
            string_view lhs_str = d_column.element<string_view>(lhs);
            string_view rhs_str = d_column.element<string_view>(rhs);
            int cmp = lhs_str.compare(rhs_str);
            return (ascending ? (cmp<0) : (cmp>0));
        });

    // create a column_view as a wrapper of these indices
    column_view indices_view( data_type{INT32}, count, indices.data().get(), nullptr, 0 );
    // now build a new strings column from the indices
    return gather( handler, indices_view );
}

//
// s1 = ['a','b,'c','d']
// s2 = ['e','f']
// pos = [1,3]  -- must be the same length as s2
// s3 = s1.scatter(s2,pos)
// ['a','e','c','f']
//
std::unique_ptr<cudf::column> scatter( strings_column_handler handler,
                                       strings_column_handler strings,
                                       cudf::column_view scatter_map,
                                       cudaStream_t stream )
{
    size_type elements = strings.size();
    CUDF_EXPECTS( elements==scatter_map.size(), "number of strings must match map size" );
    size_type count = handler.size();
    auto d_indices = scatter_map.data<int32_t>();
    auto execpol = rmm::exec_policy(stream);

    //
    rmm::device_buffer buffer = create_string_array_from_column(handler,stream);
    cudf::string_view* d_strings = reinterpret_cast<cudf::string_view*>(buffer.data());
    rmm::device_buffer map_buffer = create_string_array_from_column(strings,stream);
    cudf::string_view* d_map_strings = reinterpret_cast<cudf::string_view*>(map_buffer.data());
    thrust::scatter( execpol->on(stream), d_map_strings, d_map_strings+elements, d_indices, d_strings );

    // build offsets column
    auto offsets_column = make_numeric_column( data_type{INT32}, count, mask_state::UNALLOCATED,
                                               stream, handler.memory_resource() );
    auto offsets_view = offsets_column->mutable_view();
    auto d_offsets = offsets_view.data<int32_t>();
    // create new offsets array
    thrust::transform_inclusive_scan( execpol->on(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(count),
        d_offsets,
        [d_strings, d_offsets] __device__ (size_type idx) {
            return d_strings[idx].size();
        },
        thrust::plus<int32_t>());

    // build null mask
    auto valid_mask = valid_if( static_cast<const bit_mask_t*>(nullptr),
        [d_strings] __device__ (size_type idx) { return !d_strings[idx].is_null(); },
        count, stream );
    auto null_count = valid_mask.second;
    auto null_size = gdf_valid_allocation_size(count);
    rmm::device_buffer null_mask(valid_mask.first,null_size); // does deep copy
    RMM_TRY( RMM_FREE(valid_mask.first,stream) ); // TODO valid_if to return device_buffer in future

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[count-1]; // this may not be stream friendly
    auto chars_column = make_numeric_column( data_type{INT8}, bytes, mask_state::UNALLOCATED,
                                             stream, handler.memory_resource() );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<int8_t>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), count,
        [d_strings, d_offsets, d_chars] __device__(size_type idx){
            cudf::string_view d_str = d_strings[idx];
            if( d_str.is_null() )
                return;
            size_type offset = (idx ? d_offsets[idx-1] : 0);
            memcpy(d_chars + offset, d_str.data(), d_str.size() );
        });

  // build children vector
  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(offsets_column));
  children.emplace_back(std::move(chars_column));

  return std::make_unique<column>(
        data_type{STRING}, 0, rmm::device_buffer{0,stream,handler.memory_resource()},
        null_mask, null_count,
        std::move(children));
}

//
// s1 = ['a','b,'c','d']
// pos = [1,3]
// s3 = s1.scatter('e',pos,2)
// ['a','e','c','e']
//
std::unique_ptr<cudf::column> scatter( strings_column_handler handler,
                                       const char* string,
                                       cudf::column_view scatter_map,
                                       cudaStream_t stream )
{
//    size_type count = size();
//    size_type elements = scatter_map.size();
//    auto execpol = rmm::exec_policy(0);
//    // copy string to device
//    auto d_string = string_from_host(string);
//    cudf::string_view* d_replace = *d_string;
//    // create result output array
//    rmm::device_vector<custring_view*> results(count,nullptr);
//    auto d_results = results.data().get();
//    custring_view_array d_strings = pImpl->getStringsPtr();
//    thrust::copy( execpol->on(0), d_strings, d_strings+count, d_results );
//    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), elements,
//        [d_pos, count, d_repl, d_results] __device__ (unsigned int idx) {
//            int pos = d_pos[idx];
//            if( (pos >= 0) && (pos < count) )
//                d_results[pos] = d_repl;
//        });
//    // build resulting instance
//    NVStrings* rtn = new NVStrings(count);
//    NVStrings_init_from_custrings(rtn->pImpl, d_results, count);
//    if( !bdevmem )
//        RMM_FREE((void*)d_pos,0);
//    RMM_FREE((void*)d_repl,0);
    return nullptr;
}


} // namespace strings
} // namespace cudf