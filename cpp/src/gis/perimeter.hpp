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

#ifndef PERIMETER_HPP
#define PERIMETER_HPP

#include "cudf.h"

namespace cudf {
namespace gis {

/**
 * @brief Compute the perimeter of polygons on the Earth surface using Haversine formula.
 * 
 * Note: The polygon must not have holes.
 * 
 * The polygon is defined by a set of coordinates (latitudes and longitudes),
 * where the first and last coordinates must have the same value (closed).
 * 
 * polygon_latitudes and polygon_longitudes must have equal size to 'num_polygons'.
 *
 * All latitudes and longitudes must have equal datatypes (for numeric operations).
 * 
 * @param[in] polygons_latitudes[]: set of columns with latitudes of polygons
 * @param[in] polygons_longitudes[]: set of columns with longitudes of polygons
 * @param[in] num_polygons: Number of polygons
 *
 * @returns gdf_column of perimeters with size 'num_polygons'
 */
gdf_column perimeter( gdf_column* polygons_latitudes[],
					  gdf_column* polygons_longitudes[],
					  gdf_size_type const & num_polygons,
					  cudaStream_t stream = 0 );
}  // namespace gis
}  // namespace cudf

#endif
