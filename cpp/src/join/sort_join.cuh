/*
 * Copyright (c) 2017, NVIDIA CORPORATION.
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

/* Sort-based join using thrust */

#include "cudf/functions.h"
#include "cudf/types.h"
#include "join_types.h"
#include "full_join.cuh"

#include <utility>
#include <thrust/transform_scan.h>

template <typename index_type>
struct JoinBounds {
    thrust::device_vector<index_type> lower;
    thrust::device_vector<index_type> upper;
};

//Compute the lower and upper bound
//Looking for l in r
template<typename T,
    typename index_type,
    typename size_type>
JoinBounds<index_type>
compute_join_bounds(
        T const * const l, size_type l_count,
        T const * const r, size_type r_count) {
    JoinBounds<index_type> bounds;
    bounds.lower.resize(l_count);
    bounds.upper.resize(l_count);
    thrust::lower_bound(
            thrust::device,
            r, r + r_count,
            l, l + l_count,
            bounds.lower.begin(),
            thrust::less<T>());
    thrust::upper_bound(
            thrust::device,
            r, r + r_count,
            l, l + l_count,
            bounds.upper.begin(),
            thrust::less<T>());
    return bounds;
}

template <typename index_type, JoinType join_type>
struct Diff {
    __device__ index_type operator()(thrust::tuple<const index_type, const index_type> t) {
        return (thrust::get<0>(t) - thrust::get<1>(t)) + (thrust::get<0>(t) == thrust::get<1>(t));
    }
};

template <typename index_type>
struct Diff<index_type, JoinType::INNER_JOIN> {
    __device__ index_type operator()(thrust::tuple<const index_type, const index_type> t) {
        return thrust::get<0>(t) - thrust::get<1>(t);
    }
};

template <typename index_type>
struct JoinConditionalAdd {
    index_type none;

    __host__ __device__
    JoinConditionalAdd(const index_type NoneValue) : none(NoneValue) {}

    __device__ index_type operator()(
            index_type r_ptr,
            thrust::tuple<const index_type, const index_type> lower_upper) {
        if (thrust::get<0>(lower_upper) == thrust::get<1>(lower_upper)) {
            return none;
        } else {
            return r_ptr + thrust::get<0>(lower_upper);
        }
    }
};

template<JoinType join_type,
    typename index_type>
thrust::device_vector<index_type>
scan_join_bounds(const JoinBounds<index_type>& bounds) {
    thrust::device_vector<index_type> scanned_sizes(bounds.lower.size() + 1, 0);
    thrust::transform_inclusive_scan(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(bounds.upper.begin(), bounds.lower.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(  bounds.upper.end(),   bounds.lower.end())),
            scanned_sizes.begin() + 1,
            Diff<index_type, join_type>(),
            thrust::plus<index_type>());
    return scanned_sizes;
}

template <typename index_type>
void
create_load_balanced_tuple(const thrust::device_vector<index_type>& scanned_sizes,
        index_type * const seg, index_type * const rank, const index_type segment_length) {
    thrust::upper_bound(
            thrust::device,
            scanned_sizes.begin(),
            scanned_sizes.end(),
            thrust::make_counting_iterator(static_cast<index_type>(0)),
            thrust::make_counting_iterator(static_cast<index_type>(segment_length)),
            seg);
    thrust::transform(
            thrust::device,
            seg,
            seg + segment_length,
            thrust::make_constant_iterator(static_cast<index_type>(1)),
            seg,
            thrust::minus<index_type>());
    thrust::transform(
            thrust::device,
            thrust::make_counting_iterator(static_cast<index_type>(0)),
            thrust::make_counting_iterator(static_cast<index_type>(segment_length)),
            thrust::make_permutation_iterator(scanned_sizes.begin(), seg),
            rank,
            thrust::minus<index_type>());
}

template<JoinType join_type,
    typename index_type>
std::pair<gdf_column, gdf_column>
compute_joined_indices(const JoinBounds<index_type>& bounds,
        gdf_column * const leftcol, gdf_column * const rightcol,
        thrust::device_vector<index_type>& scanned_sizes) {
    index_type join_size = scanned_sizes[scanned_sizes.size() - 1];
    scanned_sizes.resize(scanned_sizes.size() - 1);

    index_type * l_ptr;
    index_type * r_ptr;
    cudaMalloc(&l_ptr, join_size*sizeof(index_type));
    cudaMalloc(&r_ptr, join_size*sizeof(index_type));
    create_load_balanced_tuple(scanned_sizes, l_ptr, r_ptr, join_size);
    if (join_type == JoinType::INNER_JOIN) {
        thrust::transform(
                thrust::device,
                r_ptr,
                r_ptr + join_size,
                thrust::make_permutation_iterator(bounds.lower.begin(), l_ptr),
                r_ptr,
                thrust::plus<index_type>());
    } else {
        thrust::transform(
                thrust::device,
                r_ptr,
                r_ptr + join_size,
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::make_permutation_iterator(bounds.lower.begin(), l_ptr),
                    thrust::make_permutation_iterator(bounds.upper.begin(), l_ptr))),
                r_ptr,
                JoinConditionalAdd<index_type>(static_cast<index_type>(JoinNoneValue)));
    }
    gdf_size_type final_join_size = static_cast<gdf_size_type>(join_size);
    if (join_type == JoinType::FULL_JOIN) {
        gdf_size_type join_column_capacity = final_join_size;
        append_full_join_indices(
                &l_ptr, &r_ptr,
                &join_column_capacity,
                &final_join_size, rightcol->size);
    }
    gdf_column output_l, output_r;
    gdf_column_view(&output_l, l_ptr, nullptr, final_join_size, GDF_INT32);
    gdf_column_view(&output_r, r_ptr, nullptr, final_join_size, GDF_INT32);
    return std::make_pair(output_l, output_r);
}

template<JoinType join_type,
    typename column_type,
    typename index_type>
gdf_error sort_join_typed(
        gdf_column * const output_l,
        gdf_column * const output_r,
        gdf_column * const leftcol,
        gdf_column * const rightcol,
        bool flip_results = false) {
    JoinBounds<index_type> bounds =
        compute_join_bounds<column_type, index_type>(
                static_cast<column_type*>(leftcol->data), leftcol->size,
                static_cast<column_type*>(rightcol->data), rightcol->size);
    thrust::device_vector<index_type> scanned_sizes =
        scan_join_bounds<join_type, index_type>(bounds);
    std::pair<gdf_column, gdf_column> join_result =
        compute_joined_indices<join_type, index_type>(bounds, leftcol, rightcol, scanned_sizes);
    *output_l = join_result.first;
    *output_r = join_result.second;
    if (flip_results) {
        *output_l = join_result.first;
        *output_r = join_result.second;
    }
    return GDF_SUCCESS;
}

template<JoinType join_type,
    typename index_type>
gdf_error compute_sort_join(
        gdf_column * const output_l,
        gdf_column * const output_r,
        gdf_column * const lcol,
        gdf_column * const rcol,
        bool flip = false) {
    int column_width_bytes{0};
    gdf_error gdf_status = get_column_byte_width(lcol, &column_width_bytes);

    if(GDF_SUCCESS != gdf_status)
        return gdf_status;

    switch (column_width_bytes) {
        case 1 : return sort_join_typed<join_type,  int8_t, index_type>(output_l, output_r, lcol, rcol, flip);
        case 2 : return sort_join_typed<join_type, int16_t, index_type>(output_l, output_r, lcol, rcol, flip);
        case 4 : return sort_join_typed<join_type, int32_t, index_type>(output_l, output_r, lcol, rcol, flip);
        case 8 : return sort_join_typed<join_type, int64_t, index_type>(output_l, output_r, lcol, rcol, flip);
        default: return GDF_UNSUPPORTED_DTYPE;
    }
}
