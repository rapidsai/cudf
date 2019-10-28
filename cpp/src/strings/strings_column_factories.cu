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
#include <cudf/column/column.hpp>
#include <cudf/legacy/functions.h>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include "./utilities.hpp"
#include "./utilities.cuh"

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/for_each.h>


namespace cudf {

// Create a strings-type column from vector of pointer/size pairs
std::unique_ptr<column> make_strings_column(
    const rmm::device_vector<thrust::pair<const char*,size_type>>& strings,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr)
{
    size_type strings_count = strings.size();
    // maybe a separate factory for creating null strings-column
    CUDF_EXPECTS(strings_count > 0, "must specify at least one pair");

    auto execpol = rmm::exec_policy(stream);
    auto d_strings = strings.data().get();

    // check total size is not too large for cudf column
    size_t bytes = thrust::transform_reduce( execpol->on(stream),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(strings_count),
        [d_strings] __device__ (size_t idx) {
            auto item = d_strings[idx];
            return (item.first!=nullptr) ? item.second : 0;
        },
        0, thrust::plus<size_t>());
    CUDF_EXPECTS( bytes < std::numeric_limits<size_type>::max(), "total size of strings is too large for cudf column" );

    // build offsets column from the strings sizes
    auto offsets_transformer = [d_strings] __device__ (size_type idx) {
            thrust::pair<const char*,size_type> item = d_strings[idx];
            return ( item.first!=nullptr ? static_cast<int32_t>(item.second) : 0 );
        };
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<size_type>(0), offsets_transformer );
    auto offsets_column = strings::detail::make_offsets_child_column(offsets_transformer_itr,
                                               offsets_transformer_itr+strings_count,
                                               mr, stream);
    auto offsets_view = offsets_column->view();
    auto d_offsets = offsets_view.data<int32_t>();

    // create null mask
    auto valid_mask = strings::detail::make_null_mask(strings_count,
        [d_strings] __device__ (size_type idx) { return d_strings[idx].first!=nullptr; },
        mr, stream);
    auto null_count = valid_mask.second;
    rmm::device_buffer null_mask = valid_mask.first;

    // build chars column
    auto chars_column = strings::detail::create_chars_child_column( strings_count, null_count, bytes, mr, stream );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<char>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count,
          [d_strings, d_offsets, d_chars] __device__(size_type idx){
              // place individual strings
              auto item = d_strings[idx];
              if( item.first!=nullptr )
                  memcpy(d_chars + d_offsets[idx], item.first, item.second );
          });

    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}

// Create a strings-type column from device vector of chars and vector of offsets.
std::unique_ptr<column> make_strings_column(
    const rmm::device_vector<char>& strings,
    const rmm::device_vector<size_type>& offsets,
    const rmm::device_vector<bitmask_type>& valid_mask,
    size_type null_count,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr )
{
    size_type num_strings = offsets.size()-1;
    CUDF_EXPECTS( num_strings > 0, "strings count must be greater than 0");
    CUDF_EXPECTS( null_count < num_strings, "null strings column not yet supported");
    if( null_count > 0 ) {
        CUDF_EXPECTS( !valid_mask.empty(), "Cannot have null elements without a null mask." );
    }

    auto execpol = rmm::exec_policy(stream);
    size_type bytes = offsets.back();
    CUDF_EXPECTS( bytes >=0, "invalid offsets vector");

    // build offsets column -- this is the number of strings + 1
    auto offsets_column = make_numeric_column( data_type{INT32}, num_strings+1, mask_state::UNALLOCATED, stream, mr );
    auto offsets_view = offsets_column->mutable_view();
    CUDA_TRY(cudaMemcpyAsync( offsets_view.data<int32_t>(), offsets.data().get(),
                              (num_strings+1)*sizeof(int32_t),
                              cudaMemcpyDeviceToDevice, stream ));
    // build null bitmask
    rmm::device_buffer null_mask{
        valid_mask.data().get(),
        valid_mask.size() * sizeof(bitmask_type)}; // Or this works too: sizeof(typename std::remove_reference_t<decltype(valid_mask)>::value_type)
                                                   // Following give the incorrect value of 8 instead of 4 because of smart references:
                                                   // sizeof(valid_mask[0]), sizeof(decltype(valid_mask.front()))

    // build chars column
    auto chars_column = strings::detail::create_chars_child_column( num_strings, null_count, bytes, mr, stream );
    auto chars_view = chars_column->mutable_view();
    CUDA_TRY(cudaMemcpyAsync( chars_view.data<char>(), strings.data().get(), bytes,
                              cudaMemcpyDeviceToDevice, stream ));

    return make_strings_column(num_strings, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}

// Create strings column from host vectors
std::unique_ptr<column> make_strings_column(
    const std::vector<char>& strings, const std::vector<size_type>& offsets,
    const std::vector<bitmask_type>& null_mask, size_type null_count,
    cudaStream_t stream, rmm::mr::device_memory_resource* mr) {
  rmm::device_vector<char> d_strings{strings};
  rmm::device_vector<size_type> d_offsets{offsets};
  rmm::device_vector<bitmask_type> d_null_mask{null_mask};

  return make_strings_column(d_strings, d_offsets, d_null_mask, null_count,
                             stream, mr);
}

//
std::unique_ptr<column> make_strings_column(
    size_type num_strings,
    std::unique_ptr<column> offsets_column,
    std::unique_ptr<column> chars_column,
    size_type null_count,
    rmm::device_buffer&& null_mask,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr)
{
    if( null_count > 0 )
        CUDF_EXPECTS( null_mask.size() > 0, "Column with nulls must be nullable.");
    CUDF_EXPECTS( num_strings == offsets_column->size()-1, "Invalid offsets column size for strings column." );
    CUDF_EXPECTS( offsets_column->null_count()==0, "Offsets column should not contain nulls");
    CUDF_EXPECTS( chars_column->null_count()==0, "Chars column should not contain nulls");

    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(offsets_column));
    children.emplace_back(std::move(chars_column));
    return std::make_unique<column>(
        data_type{STRING}, num_strings, rmm::device_buffer{0,stream,mr},
        null_mask, null_count,
        std::move(children));
}

}  // namespace cudf
