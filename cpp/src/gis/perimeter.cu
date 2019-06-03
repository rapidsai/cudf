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

 #include "gis.hpp"
 #include "perimeter.hpp"
 #include "rmm/rmm.h"
 #include "utilities/type_dispatcher.hpp"
 #include <utilities/cuda_utils.hpp>

 #include <thrust/reduce.h>
 #include <type_traits>
 #include <vector>
 
namespace {

__constant__ double DEG_TO_RAD = 0.01745329251994329;
__constant__ double EARTH_RADIUS = 6372797.560856;

/*
* @brief Compute the geodesic distance between two coordinates with Haversine formula.
*
* @param[in] current_latitude: Latitude of the first coordinate
* @param[in] current_longitude: Longitude of the first coordinate
* @param[in] next_latitude: Latitude of the second coordinate
* @param[in] next_longitude: Longitude of the second coordinate
*
* @returns the geodesic distance between two coordinates
*/
template <typename T>
__device__ T haversine_formula( const T current_latitude, const T current_longitude,
                                const T next_latitude, const T next_longitude )
{
    T lat_variation = (next_latitude - current_latitude) * DEG_TO_RAD;
    T lon_variation = (next_longitude - current_longitude) * DEG_TO_RAD;
    T partial_1 = sin(lat_variation * 0.5);
    partial_1 *= partial_1;
    T partial_2 = sin(lon_variation * 0.5);
    partial_2 *= partial_2;
    T tmp = cos(current_latitude * DEG_TO_RAD) * cos(next_latitude * DEG_TO_RAD);
    return ( 2.0 * EARTH_RADIUS * asin(sqrt(partial_1 + tmp * partial_2)) );
}

/** 
* @brief Compute the perimeter of polygons on the Earth surface using Haversine formula.
*
* @param[in] poly_concat_lats: Pointer to latitudes for multiple polygons
* @param[in] poly_concat_lons: Pointer to longitudes for multiple polygons
* @param[in] num_polygons: # polygons
* @param[in] offset_polygon_values: Accumulated histogram that contain size of polygons
* @param[out] perimeter_output: Pointer indicating perimeters of polygons
*
* @returns
*/
template <typename T>
__global__ void perimeters_kernel( const T* const __restrict__ concat_lats, 
                                   const T* const __restrict__ concat_lons,
                                   const gdf_size_type* const __restrict__ offset_values,
                                   const gdf_size_type num_polygons,
                                   T* const __restrict__ perimeter_output )
{
    gdf_index_type start_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (gdf_index_type idx = start_idx; idx < num_polygons; idx += blockDim.x * gridDim.x)
    {   
        perimeter_output[idx] = 0;
        for (int side_idx = offset_values[idx]; side_idx < offset_values[idx + 1] - 1; side_idx++) 
        {
            perimeter_output[idx] += haversine_formula(concat_lats[side_idx], concat_lons[side_idx], concat_lats[side_idx + 1], concat_lons[side_idx + 1]);
        }
    }
}

/** 
*  @brief Compute the perimeter of only one polygon on the Earth surface using Haversine formula.
*
* @param[in] pol_lats: Pointer to latitudes of only one polygon
* @param[in] pol_lons: Pointer to longitudes of only one polygon
* @param[in] size_polygon: Size polygon
* @param[out] lenghts_polygon: Pointer indicating the length for each side of polygon
*
* @returns
*/
template <typename T>
__global__ void perimeter_for_one_polygon_kernel( const T* const __restrict__ pol_lats, 
                                                  const T* const __restrict__ pol_lons,
                                                  const gdf_size_type size_polygon,
                                                  T* const __restrict__ lenghts_polygon )
{
    gdf_index_type start_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (gdf_index_type idx = start_idx; idx < size_polygon - 1; idx += blockDim.x * gridDim.x)
    {   
        lenghts_polygon[idx] = haversine_formula(pol_lats[idx], pol_lons[idx], pol_lats[idx + 1], pol_lons[idx + 1]);
    }
}

struct perimeter_functor {
    template <typename col_type>
    static constexpr bool is_supported() { return std::is_arithmetic<col_type>::value; }

    template <typename col_type, std::enable_if_t< is_supported<col_type>() >* = nullptr>
    gdf_column operator()( gdf_column* polygons_latitudes[], gdf_column* polygons_longitudes[],
                           gdf_size_type* const offset_size_polygons, gdf_size_type const num_polygons, 
                           gdf_size_type const total_points_of_all_polygons, cudaStream_t stream = 0 )
    {
        // Preparing the output perimeter
        gdf_column perimeter_output;
        col_type* buffer_perimeter;
        RMM_TRY( RMM_ALLOC((void**)&buffer_perimeter, sizeof(col_type) * num_polygons, 0) );
       
        gdf_size_type min_grid_size = 0, block_size = 0;

        // Only a polygon
        if (num_polygons == 1)
        {
            // Store lenghts of polygon
            rmm::device_vector<col_type> lenghts_polygon(polygons_latitudes[0]->size);

            // Launch the optimized Kernel
            CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, perimeter_for_one_polygon_kernel<col_type>) );
            cudf::util::cuda::grid_config_1d grid{polygons_latitudes[0]->size, block_size, 1};

            perimeter_for_one_polygon_kernel<col_type> <<< grid.num_blocks, block_size >>> ( static_cast<col_type*>(polygons_latitudes[0]->data),
                        static_cast<col_type*>(polygons_longitudes[0]->data), polygons_latitudes[0]->size, lenghts_polygon.data().get() );

            // Add all lenghts
            col_type perimeter = thrust::reduce(lenghts_polygon.begin(), lenghts_polygon.end());
            CUDA_TRY( cudaMemcpy(buffer_perimeter, &perimeter, sizeof(col_type) * num_polygons, cudaMemcpyHostToDevice) );
            gdf_column_view(&perimeter_output, buffer_perimeter, nullptr, num_polygons, GDF_FLOAT64);

        } else
        {
            gdf_column_view(&perimeter_output, buffer_perimeter, nullptr, num_polygons, GDF_FLOAT64);

            // Concatenate latitudes of the polygons to one column (same for longitudes)
            gdf_column* concat_latitudes = new gdf_column;
            gdf_column* concat_longitudes = new gdf_column;

            col_type* buffer_all_latitudes, *buffer_all_longitudes;
            RMM_TRY( RMM_ALLOC((void**)&buffer_all_latitudes, sizeof(col_type) * total_points_of_all_polygons, 0) );
            RMM_TRY( RMM_ALLOC((void**)&buffer_all_longitudes, sizeof(col_type) * total_points_of_all_polygons, 0) );

            gdf_column_view(concat_latitudes, buffer_all_latitudes, nullptr, total_points_of_all_polygons, GDF_FLOAT64);
            gdf_column_view(concat_longitudes, buffer_all_longitudes, nullptr, total_points_of_all_polygons, GDF_FLOAT64);

            gdf_column_concat(concat_latitudes, polygons_latitudes, num_polygons);
            gdf_column_concat(concat_longitudes, polygons_longitudes, num_polygons);

            // Launch the optimized Kernel
            CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, perimeters_kernel<col_type>) );
            cudf::util::cuda::grid_config_1d grid{concat_latitudes->size, block_size, 1};

            perimeters_kernel<col_type> <<< grid.num_blocks, block_size >>> ( static_cast<col_type*>(concat_latitudes->data),
                        static_cast<col_type*>(concat_longitudes->data), offset_size_polygons,
                        num_polygons, static_cast<col_type*>(perimeter_output.data) );
        }
        
        CHECK_STREAM(stream);

        return perimeter_output;
    }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    gdf_column operator()( gdf_column* polygons_latitudes[], gdf_column* polygons_longitudes[],
                           gdf_size_type* const vector_offset_data, gdf_size_type const num_polygons, 
                           gdf_size_type const offset_values, cudaStream_t stream = 0 )
    {
        CUDF_FAIL("Non-arithmetic operation is not supported");
    }
};
    
}   //  namespace
 
namespace cudf {
namespace gis {

gdf_column perimeter( gdf_column* polygons_latitudes[], gdf_column* polygons_longitudes[],
                      gdf_size_type const & num_polygons, cudaStream_t stream )
{
    CUDA_TRY( cudaStreamCreate(&stream) );

    // Expects data be in good format 
    for (gdf_index_type polygon_i = 0; polygon_i < num_polygons; ++polygon_i)
    {
        CUDF_EXPECTS(polygons_latitudes[polygon_i]->data != nullptr && polygons_longitudes[polygon_i]->data != nullptr, "polygon data cannot be empty");
        CUDF_EXPECTS(polygons_latitudes[polygon_i]->size == polygons_longitudes[polygon_i]->size, "polygon size mismatch");
        CUDF_EXPECTS(polygons_latitudes[polygon_i]->dtype == polygons_longitudes[polygon_i]->dtype, "polygon type mismatch");
        CUDF_EXPECTS(polygons_latitudes[polygon_i]->null_count == 0 && polygons_longitudes[polygon_i]->null_count == 0, "polygon should not contain nulls");
    }

    // Vector with the sizes of polygons
    gdf_size_type offset_size_polygons = 0;
    std::vector<gdf_size_type> host_offset_values(1, offset_size_polygons);
    
    for (gdf_size_type polygon_i = 0; polygon_i < num_polygons; ++polygon_i)
    {
        offset_size_polygons += polygons_latitudes[polygon_i]->size;
        host_offset_values.push_back(offset_size_polygons);
    }
    rmm::device_vector<gdf_size_type> vector_offset_data = host_offset_values;

    gdf_column perimeter_output = cudf::type_dispatcher( polygons_latitudes[0]->dtype ,
                                                    perimeter_functor(),
                                                    polygons_latitudes,
                                                    polygons_longitudes,
                                                    vector_offset_data.data().get(),
                                                    num_polygons,
                                                    offset_size_polygons,
                                                    stream );
    return perimeter_output;
}
}   // namespace gis

gdf_column perimeter(gdf_column* polygons_latitudes[], gdf_column* polygons_longitudes[], gdf_size_type const & num_polygons)
{
    return gis::perimeter(polygons_latitudes, polygons_longitudes, num_polygons);
}

}   // namespace cudf
