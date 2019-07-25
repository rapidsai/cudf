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

#ifndef POINT_IN_POLYGON_HPP
#define POINT_IN_POLYGON_HPP

#include <cudf/cudf.h>

namespace cudf {
namespace gis {

/** 
 * @brief Determine whether or not coordinates (query points) are completely inside a static polygon
 * 
 * Note: The polygon must not have holes or intersect with itself, but it is not
 * required to be convex.
 * 
 * The polygon is defined by a set of coordinates (latitudes and longitudes),
 * where the first and last coordinates must have the same value (closed).
 * 
 * This function supports clockwise and counter-clockwise polygons.
 * 
 * If a query point is colinear with two contiguous polygon coordinates
 * then this query point isn't inside.
 * 
 * polygon_latitudes and polygon_longitudes must have equal size.
 * 
 * point_latitudes and point_longitudes must have equal size.
 * 
 * All input params must have equal datatypes (for numeric operations).
 *
 * @param[in] polygon_latitudes: column with latitudes of a polygon
 * @param[in] polygon_longitudes: column with longitudes of a polygon
 * @param[in] query_point_latitudes: column with latitudes of query points
 * @param[in] query_point_longitudes: column with longitudes of query points
 * @param stream Optional CUDA stream on which to execute kernels
 *
 * @returns gdf_column of type GDF_BOOL8 indicating whether the i-th query point is inside (true) or not (false)
 */
gdf_column point_in_polygon( gdf_column const & polygon_latitudes,
                             gdf_column const & polygon_longitudes,
                             gdf_column const & query_point_latitudes,
                             gdf_column const & query_point_longitudes,
                             cudaStream_t stream = 0 );

}  // namespace gis
}  // namespace cudf

#endif
