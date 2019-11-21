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
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/detail/utilities.cuh>

namespace cudf
{
namespace strings
{
namespace detail
{

/**
 * @brief Creates a new strings column as if an in-place scatter from the source
 * `strings` column was performed on that column.
 *
 * The values in `scatter_map` must be in the range [0,target.size()).
 *
 * If the same index appears more than once in `scatter_map` the result is undefined for that index.
 *
 * This operation basically pre-fills the output column with elements from `target`. 
 * Then, for each value `i` in range `[0,values.size())`, the `source[i]` element is
 * assigned to `output[scatter_map[i]]`.
 *
 * The caller must update the null mask.
 *
 * ```
 * t = ["a", "b", "c", "d"]
 * s = ["e", "f"]
 * map = [1, 3]
 * r = scatter( s, t, map )
 * r is now ["a", "e", "c", "f"]
 * ```
 *
 * @param source Strings used for scattering into target.
 * @param scatter_map The 0-based index values to retrieve from the source parameter.
 * @param target Strings to place source strings specified by the scatter_map indices.
 * @param mr Resource for allocating device memory.
 * @param stream CUDA stream to use kernels in this method.
 * @return New strings column.
 */
template<typename MapIterator>
std::unique_ptr<cudf::column> scatter( strings_column_view const& source,
                                       MapIterator scatter_map,
                                       strings_column_view const& target,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                       cudaStream_t stream=0 )
{
    auto strings_count = target.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);

    auto execpol = rmm::exec_policy(stream);

    // create null mask -- caller must update this
    rmm::device_buffer null_mask;
    if( target.has_nulls() )
        null_mask = copy_bitmask(target.parent(),stream,mr);
    else if( source.has_nulls() ) // make nullable
        null_mask = create_null_mask( strings_count, UNINITIALIZED, stream,mr );

    // create strings vectors
    rmm::device_vector<string_view> source_vector = create_string_vector_from_column(source,stream);
    rmm::device_vector<string_view> target_vector = create_string_vector_from_column(target,stream);
    // do the scatter
    thrust::scatter( execpol->on(stream),
                     source_vector.begin(), source_vector.end(),
                     scatter_map, target_vector.begin() );

    // build offsets column
    auto offsets_column = child_offsets_from_string_vector(target_vector,mr,stream);
    auto d_offsets = offsets_column->view().data<int32_t>();
    // build chars column
    auto chars_column = child_chars_from_string_vector(target_vector,d_offsets,0,mr,stream    );

    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               UNKNOWN_NULL_COUNT, std::move(null_mask), stream, mr);
}


} // namespace detail
} // namespace strings
} // namespace cudf
