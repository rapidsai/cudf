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

#ifndef SORT_JOIN_CUH
#define SORT_JOIN_CUH

#include <cudf/legacy/functions.h>
#include <cudf/types.h>
#include "join_compute_api.h"
#include "full_join.cuh"

#include <utility>
#include <thrust/binary_search.h>
#include <thrust/transform_scan.h>

template <typename index_type>
struct JoinBounds {
    rmm::device_vector<index_type> lower;
    rmm::device_vector<index_type> upper;
};

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Computes the lower and upper bound searches of left column values
 * in the right column
 *
 * @Param l The left column to be joined
 * @Param l_count The size of the left column
 * @Param r The right column to be joined
 * @Param r_count The size of the right column
 * @tparam T Type of column data to be joined
 * @tparam index_type Output type for the index calculation
 * @tparam size_type Type for size specification of the columns
 *
 * @Returns JoinBounds struct containing the lower and upper bound search results
 */
/* ----------------------------------------------------------------------------*/
template<typename T,
    typename index_type,
    typename size_type>
JoinBounds<index_type>
compute_join_bounds(
        T const * const l, size_type l_count,
        T const * const r, size_type r_count,
        cudaStream_t stream) {
    JoinBounds<index_type> bounds;
    bounds.lower.resize(l_count);
    bounds.upper.resize(l_count);
    thrust::lower_bound(
            rmm::exec_policy(stream)->on(stream),
            r, r + r_count,
            l, l + l_count,
            bounds.lower.begin(),
            thrust::less<T>());
    thrust::upper_bound(
            rmm::exec_policy(stream)->on(stream),
            r, r + r_count,
            l, l + l_count,
            bounds.upper.begin(),
            thrust::less<T>());
    return bounds;
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Functor for computing the difference in a tuple
 *
 * If the join_type is inner then the difference in the tuple values are
 * returned otherwise an extra check ensures that the difference is at least
 * one.
 *
 */
/* ----------------------------------------------------------------------------*/
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

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Functor for taking care of non inner join index calculations
 *
 * If the upper and lower bound index values of the left table are the same
 * then 'none' is returned otherwise the returned index is the sum of the
 * lower bound search result and r_ptr
 *
 */
/* ----------------------------------------------------------------------------*/
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

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Scans the difference between the upper and lower bound search results
 * from compute_join_bounds call
 * @Param bounds Struct containing the upper and lower bound search results
 * @tparam join_type Type of join to be performed (INNER, LEFT, FULL)
 * @tparam index_type Output type for the index calculation
 *
 * @Returns thrust::device_vector containing the scanned differences
 */
/* ----------------------------------------------------------------------------*/
template<JoinType join_type,
    typename index_type>
rmm::device_vector<index_type>
scan_join_bounds(const JoinBounds<index_type>& bounds, cudaStream_t stream) {
    rmm::device_vector<index_type> scanned_sizes(bounds.lower.size() + 1, 0);
    thrust::transform_inclusive_scan(
            rmm::exec_policy(stream)->on(stream),
            thrust::make_zip_iterator(thrust::make_tuple(bounds.upper.begin(), bounds.lower.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(  bounds.upper.end(),   bounds.lower.end())),
            scanned_sizes.begin() + 1,
            Diff<index_type, join_type>(),
            thrust::plus<index_type>());
    return scanned_sizes;
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Creates two outputs segment and rank.
 * Segment contains values i repeated scanned_sizes[i] times where i ranges from
 * 0 to the length of scanned_sizes. Rank denotes the repeatition rank of each
 * element in Segment.
 * @Param scanned_sizes Array containin the number of time an index has to be repeated
 * @Param seg Array containing repeated indices
 * @Param rank Array containing repeation rank of seg
 * @Param segment_length Last value in scanned_sizes
 * @tparam index_type Output type for the index calculation
 *
 */
/* ----------------------------------------------------------------------------*/
template <typename index_type>
void
create_load_balanced_tuple(const rmm::device_vector<index_type>& scanned_sizes,
        index_type * const seg, index_type * const rank, const index_type segment_length,
        cudaStream_t stream) {
    thrust::upper_bound(
            rmm::exec_policy(stream)->on(stream),
            scanned_sizes.begin(),
            scanned_sizes.end(),
            thrust::make_counting_iterator(static_cast<index_type>(0)),
            thrust::make_counting_iterator(static_cast<index_type>(segment_length)),
            seg);
    thrust::transform(
            rmm::exec_policy(stream)->on(stream),
            seg,
            seg + segment_length,
            thrust::make_constant_iterator(static_cast<index_type>(1)),
            seg,
            thrust::minus<index_type>());
    thrust::transform(
            rmm::exec_policy(stream)->on(stream),
            thrust::make_counting_iterator(static_cast<index_type>(0)),
            thrust::make_counting_iterator(static_cast<index_type>(segment_length)),
            thrust::make_permutation_iterator(scanned_sizes.begin(), seg),
            rank,
            thrust::minus<index_type>());
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Computes the joined indices of the left and right columns
 * @Param bounds Struct containing the upper and lower bound search results
 * @Param leftcol The left column to be joined
 * @Param rightcol The right column to be joined
 * @Param scanned_sizes The scanned result of the difference between upper and lower bound search results
 * @tparam join_type Type of join to be performed (INNER, LEFT, FULL)
 * @tparam index_type Output type for the index calculation
 *
 * @Returns Pair of gdf columns containing the join result indices
 */
/* ----------------------------------------------------------------------------*/
template<JoinType join_type,
    typename index_type>
gdf_error
compute_joined_indices(const JoinBounds<index_type>& bounds,
        gdf_column * const leftcol, gdf_column * const rightcol,
        rmm::device_vector<index_type>& scanned_sizes,
        std::pair<gdf_column, gdf_column>& join_result,
        cudaStream_t stream) {
    index_type join_size = scanned_sizes[scanned_sizes.size() - 1];
    scanned_sizes.resize(scanned_sizes.size() - 1);

    index_type * l_ptr;
    index_type * r_ptr;
    RMM_TRY( RMM_ALLOC((void**)&l_ptr, join_size*sizeof(index_type), stream));
    RMM_TRY( RMM_ALLOC((void**)&r_ptr, join_size*sizeof(index_type), stream));
    create_load_balanced_tuple(scanned_sizes, l_ptr, r_ptr, join_size,
            stream);
    CHECK_CUDA(stream);
    if (join_type == JoinType::INNER_JOIN) {
        thrust::transform(
                rmm::exec_policy(stream)->on(stream),
                r_ptr,
                r_ptr + join_size,
                thrust::make_permutation_iterator(bounds.lower.begin(), l_ptr),
                r_ptr,
                thrust::plus<index_type>());
    } else {
        thrust::transform(
                rmm::exec_policy(stream)->on(stream),
                r_ptr,
                r_ptr + join_size,
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::make_permutation_iterator(bounds.lower.begin(), l_ptr),
                    thrust::make_permutation_iterator(bounds.upper.begin(), l_ptr))),
                r_ptr,
                JoinConditionalAdd<index_type>(static_cast<index_type>(JoinNoneValue)));
    }
    CHECK_CUDA(stream);
    cudf::size_type final_join_size = static_cast<cudf::size_type>(join_size);
    if (join_type == JoinType::FULL_JOIN) {
        cudf::size_type join_column_capacity = final_join_size;
        gdf_error err = append_full_join_indices(
                &l_ptr, &r_ptr,
                join_column_capacity,
                final_join_size, rightcol->size,
                stream);
        if (GDF_SUCCESS != err) return err;
    }
    gdf_column output_l, output_r;
    gdf_column_view(&output_l, l_ptr, nullptr, final_join_size, GDF_INT32);
    gdf_column_view(&output_r, r_ptr, nullptr, final_join_size, GDF_INT32);
    join_result.first = output_l;
    join_result.second = output_r;
    return GDF_SUCCESS;
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Sort based join call
 *
 * @Param output_l The left index output of join
 * @Param output_r The right index output of join
 * @Param leftcol The left column to be joined
 * @Param rightcol The right column to be joined
 * @Param flip_indices Flag that indicates whether the left and right tables have been
 * flipped, meaning the output indices should also be flipped.
 * @tparam join_type The type of join to be performed
 * @tparam column_type The datatype of the join input columns
 * @tparam index_type The datatype used for the output indices
 *
 * @Returns Upon successful computation, returns GDF_SUCCESS. Otherwise returns appropriate error code 
 */
/* ----------------------------------------------------------------------------*/
template<JoinType join_type,
    typename column_type,
    typename index_type>
gdf_error sort_join_typed(
        gdf_column * const output_l,
        gdf_column * const output_r,
        gdf_column * const leftcol,
        gdf_column * const rightcol,
        bool flip_results = false) {
    cudaStream_t stream = 0;
    JoinBounds<index_type> bounds =
        compute_join_bounds<column_type, index_type>(
                static_cast<column_type*>(leftcol->data), leftcol->size,
                static_cast<column_type*>(rightcol->data), rightcol->size,
                stream);
    CHECK_CUDA(stream);
    rmm::device_vector<index_type> scanned_sizes =
        scan_join_bounds<join_type, index_type>(bounds, stream);
    CHECK_CUDA(stream);
    std::pair<gdf_column, gdf_column> join_result;
    gdf_error err = compute_joined_indices<join_type, index_type>(
            bounds, leftcol, rightcol,
            scanned_sizes, join_result,
            stream);
    if (GDF_SUCCESS != err) return err;
    *output_l = join_result.first;
    *output_r = join_result.second;
    if (flip_results) {
        *output_l = join_result.first;
        *output_r = join_result.second;
    }
    return GDF_SUCCESS;
}

/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis  Struct wrapper around typed sort based join call
 *
 * @Param output_l The left index output of join
 * @Param output_r The right index output of join
 * @Param lcol The left column to be joined
 * @Param rcol The right column to be joined
 * @Param flip Flag that indicates whether the left and right tables have been
 * flipped, meaning the output indices should also be flipped.
 * @tparam join_type The type of join to be performed
 * @tparam index_type The datatype used for the output indices
 *
 * @Returns Upon successful computation, returns GDF_SUCCESS. Otherwise returns appropriate error code 
 */
template<JoinType join_type,
    typename index_type>
struct compute_sort_join {
    template <typename column_type>
    gdf_error operator()(
        gdf_column * const output_l,
        gdf_column * const output_r,
        gdf_column * const lcol,
        gdf_column * const rcol,
        bool flip = false) {
        using T = typename std::decay<decltype(cudf::detail::unwrap(column_type{}) )>::type;
        return sort_join_typed<join_type, T, index_type>(output_l, output_r, lcol, rcol, flip);
    }
};

#endif
