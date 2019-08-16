/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Christian Cordova Estrada <christianc@blazingdb.com>
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

#include <cudf/gis.hpp>
#include "point_in_polygon.hpp"

#include <bitmask/legacy/BitMask.cuh>
#include <bitmask/legacy/bitmask_ops.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/cuda_utils.hpp>

#include <type_traits>

namespace {

 /** 
 * @brief Compute the orientation of point p3 relative to the vector from point p1 to p2
 *
 * P1 * 
 *    |\
 *    | \
 * P2 *--*P3
 *
 * Orientation is calculated as if a 2D vector from P1 to P2 (e.g. U = P2 - P1) 
 * were rotated to orient toward P3. It is positive for clockwise rotation, 
 * negative for counter-clockwise rotation, and 0 if the points are collinear.
 *
 * @param[in] p1_x: Longitude of the first point p1
 * @param[in] p1_y: Latitude of the first point p1
 * @param[in] p2_x: Longitude of the second point p2
 * @param[in] p2_y: Latitude of the second point p2
 * @param[in] p3_x: Longitude of the third point p3
 * @param[in] p3_y: Latitude of the third point p3
 *
 * @returns positive if it's clockwise, negative if is counter-clockwise and 0 if p1, p2 and p3 are colinear
 */
template <typename T>
__device__ T orientation( const T & p1_x, const T & p1_y,
                          const T & p2_x, const T & p2_y,
                          const T p3_x, const T & p3_y )
{
    return ( (p2_y - p1_y) * (p3_x - p2_x) - (p2_x - p1_x) * (p3_y - p2_y) );
}

/** 
 * @brief Determine whether or not coordinates (query points) are completely inside a static polygon.
 *
 * Note: The polygon must not have holes or intersect with itself, but it is not
 * required to be convex.
 *
 * Each thread check if a query point is inside the polygon.
 *
 * First check if the query point longitude is between two contiguous sides of the polygon
 * If so, then check the orientations. If the orientation is clockwise then it increases by one,
 * otherwise it decreases by one at the count variable.
 *
 * For instance the "q" query point is between longitudes of Po and P1, also P2 and P3
 * In both "q" has a clockwise orientation, so the count value increases in 2.
 * The P polygon {P0, P1, P2, P3, P0}.
 *
 *  Po --------------- P1
 *  |                  |
 *  |       *q         |
 *  |                  |
 *  P3 --------------- P2
 *
 * For a more complex polygon P. The "q" query point longitude is between the longitudes of:
 * Po to P1: q has clockwise orientation, "count" increases by one (count = 1)
 * P2 to P3: q has counter-clockwise orientation, "count" decreases by one (count = 0)
 * P4 to P5: q has counter-clockwise orientation, "count" decreases by one (count = -1)
 * P6 to P7: q has clockwise orientation, "count" increases by one (count = 0)
 * Finally: count = 0, so "q" is outside of P polygon.
 *
 * Po --------------------- P1
 * |                        |
 * |                        |
 * |         P3 ----------- P2
 * |         |
 * |         |     * q
 * |         |
 * |         P4 ----------- P5
 * |                        |
 * |                        |
 * P7 --------------------- P6
 *
 * 
 * Degenerate polygon P {P0, P1, P2, P3, P4, P5, P6, P7, P0} intersects with itself.
 * Query point q must be outside, but is considered inside.
 *
 * P0 ------------------- P1
 * |                      |
 * |                      |
 * |         P4 ------------------- P5
 * |         |    .q      |          |
 * |         |            |          |
 * |         P3 -------- P2          |
 * |                                 |
 * |                                 |
 * P7 ----------------------------- P6
 * 
 * You must verify that the polygon does not intersect with itself as the before case.
 * 
 * @param[in] poly_lats: Pointer to latitudes of a polygon
 * @param[in] poly_lons: Pointer to longitudes of a polygon
 * @param[in] point_lats: Pointer to latitudes of query points
 * @param[in] point_lons: Pointer to longitudes of query points
 * @param[in] poly_size: Size of polygon
 * @param[in] point_size: Total number of query points
 * @param[out] point_is_in_polygon: Pointer indicating if the i-th query point is inside or not
 *
 * @returns
 */
 template <typename T>
 __global__ void point_in_polygon_kernel( const T* const __restrict__ poly_lats,
                                          const T* const __restrict__ poly_lons,
                                          const T* const __restrict__ point_lats,
                                          const T* const __restrict__ point_lons,
                                          gdf_size_type poly_size,
                                          gdf_size_type points_size,
                                          cudf::bool8* const __restrict__ point_is_in_polygon )
{
    gdf_index_type start_idx = blockIdx.x * blockDim.x + threadIdx.x;
 
    for(gdf_index_type idx = start_idx; idx < points_size; idx += blockDim.x * gridDim.x)
    {
        T point_lat = point_lats[idx];
        T point_lon = point_lons[idx];
        gdf_size_type count = 0;
 
        for(gdf_index_type poly_side = 0; poly_side < poly_size - 1; poly_side++) 
        {
            if(poly_lons[poly_side] <= point_lon && point_lon < poly_lons[poly_side + 1])
            {
                T orientation_ = orientation(poly_lons[poly_side], poly_lats[poly_side], poly_lons[poly_side + 1], poly_lats[poly_side + 1], point_lon, point_lat);
                
                if (orientation_ > 0) count++;
                else if (orientation_ < 0) count--;

            }
            else if (point_lon <= poly_lons[poly_side] && poly_lons[poly_side + 1] < point_lon) 
            {
                T orientation_ = orientation(poly_lons[poly_side], poly_lats[poly_side], poly_lons[poly_side + 1], poly_lats[poly_side + 1], point_lon, point_lat);

                if (orientation_ > 0) count++;
                else if (orientation_ < 0) count--;
            }
        }
        point_is_in_polygon[idx] = cudf::bool8{(count != 0) && (count % 2 == 0)};
    }
}

struct point_in_polygon_functor {
    template <typename col_type>
    static constexpr bool is_supported()
    {
        return std::is_arithmetic<col_type>::value;
    }

    template <typename col_type, std::enable_if_t< is_supported<col_type>() >* = nullptr>
    gdf_column operator()( gdf_column const & d_poly_lats, gdf_column const & d_poly_lons,
                           gdf_column const & d_point_lats, gdf_column const & d_point_lons,
                           cudaStream_t stream = 0 )
    {
        gdf_column d_point_is_in_polygon;
        cudf::bool8* data;
        RMM_TRY( RMM_ALLOC(&data, d_point_lats.size * sizeof(cudf::bool8), 0) );
        gdf_column_view(&d_point_is_in_polygon, data, nullptr, d_point_lats.size, GDF_BOOL8);

        gdf_size_type min_grid_size = 0, block_size = 0;
        CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, point_in_polygon_kernel<col_type>) );

        cudf::util::cuda::grid_config_1d grid{d_point_lats.size, block_size, 1};

        point_in_polygon_kernel<col_type> <<< grid.num_blocks, block_size >>> ( static_cast<col_type*>(d_poly_lats.data), 
                static_cast<col_type*>(d_poly_lons.data), static_cast<col_type*>(d_point_lats.data), static_cast<col_type*>(d_point_lons.data), 
                d_poly_lats.size, d_point_lats.size, static_cast<cudf::bool8*>(d_point_is_in_polygon.data) );

        CHECK_STREAM(stream);
                
        return d_point_is_in_polygon;
    }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    gdf_column operator()( gdf_column const & d_poly_lats, gdf_column const & d_poly_lons,
                           gdf_column const & d_point_lats, gdf_column const & d_point_lons,
                           cudaStream_t stream = 0 )
    {
        CUDF_FAIL("Non-arithmetic operation is not supported");
    }
};
    
}   //  namespace

namespace cudf {
namespace gis {

gdf_column point_in_polygon( gdf_column const & polygon_lats, gdf_column const & polygon_lons,
                             gdf_column const & point_lats, gdf_column const & point_lons, 
                             cudaStream_t stream )
{       
    CUDF_EXPECTS(polygon_lats.data != nullptr && polygon_lons.data != nullptr, "polygon data cannot be empty");
    CUDF_EXPECTS(point_lats.data != nullptr && point_lons.data != nullptr, "query point data cannot be empty");
    CUDF_EXPECTS(polygon_lats.size == polygon_lons.size, "polygon size mismatch");
    CUDF_EXPECTS(point_lats.size == point_lons.size, "query points size mismatch");
    CUDF_EXPECTS(polygon_lats.dtype == polygon_lons.dtype, "polygon type mismatch");
    CUDF_EXPECTS(polygon_lats.dtype == point_lats.dtype, "query point / polygon type mismatch");
    CUDF_EXPECTS(point_lats.dtype == point_lons.dtype, "query points type mismatch");
    CUDF_EXPECTS(polygon_lons.null_count == 0 && polygon_lats.null_count == 0, "polygon should not contain nulls");

    gdf_column inside_polygon = cudf::type_dispatcher( polygon_lats.dtype,
                                                point_in_polygon_functor(),
                                                polygon_lats,
                                                polygon_lons,
                                                point_lats,
                                                point_lons,
                                                stream );
    
    if ( point_lats.null_count == 0 && point_lons.null_count == 0 ) inside_polygon.null_count = 0;
    else {
        auto error_copy_bit_mask = bit_mask::copy_bit_mask( reinterpret_cast<bit_mask::bit_mask_t*>(inside_polygon.valid),
                                                            reinterpret_cast<bit_mask::bit_mask_t*>(point_lats.valid), 
                                                            point_lats.size, cudaMemcpyDeviceToDevice );

        gdf_size_type null_count;
        auto err = apply_bitmask_to_bitmask( null_count, inside_polygon.valid,
                                             inside_polygon.valid, point_lons.valid, 
                                             stream, inside_polygon.size );

        inside_polygon.null_count = null_count;
    }

    return inside_polygon;
}
}   // namespace gis

gdf_column point_in_polygon( gdf_column const & polygon_lats, gdf_column const & polygon_lons,
                             gdf_column const & query_point_lats, gdf_column const & query_point_lons )
{
    return gis::point_in_polygon(polygon_lats, polygon_lons, query_point_lats, query_point_lons);
}

}   // namespace cudf
