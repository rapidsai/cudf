#include "gis.hpp"
#include "point_in_polygon.hpp"

#include <bitmask/BitMask.cuh>
#include "bitmask/bitmask_ops.h"
#include "utilities/type_dispatcher.hpp"

#include <type_traits>

namespace {

    /** 
 * @brief Compute the orientation of p3 from two others points p1 and p2
 *
 * @param[in] p1_x: Longitude of the first point p1
 * @param[in] p1_y: Latitude of the first point p1
 * @param[in] p2_x: Longitude of the second point p2
 * @param[in] p2_y: Latitude of the second point p2
 * @param[in] p3_x: Longitude of the third point p3
 * @param[in] p3_y: Latitude of the third point p3
 *
 * @returns positive if it's clockwise, negative if is counter-clockwise and 0 if is colinear
 */
template <typename T>
__device__ T orientation(const T p1_x, const T p1_y, const T p2_x, const T p2_y, const T p3_x, const T p3_y)
{
    return ((p2_y - p1_y) * (p3_x - p2_x) - (p2_x - p1_x) * (p3_y - p2_y));
}

/** 
 * @brief Find if coordinates (query points) are completely inside or not in a specific polygon
 *
 * @param[in] poly_lats: Pointer to latitudes of a polygon
 * @param[in] poly_lons: Pointer to longitudes of a polygon
 * @param[in] point_lats: Pointer to latitudes of many query points
 * @param[in] point_lons: Pointer to longitudes of many query points
 * @param[in] poly_size: Size of polygon (first coordinate = last coordinate) must be closed
 * @param[in] point_size: Total number of query points
 * @param[out] point_is_in_polygon: Pointer indicating if the i-th query point is inside or not with {1, 0}
 *
 * @returns
 */
 template <typename T>
 __global__ void point_in_polygon_kernel(const T* poly_lats,
                                    const T* poly_lons,
                                    const T* point_lats,
                                    const T* point_lons,
                                    gdf_size_type poly_size,
                                    gdf_size_type point_size,
                                    int8_t* point_is_in_polygon)
{
    gdf_index_type start_idx = blockIdx.x * blockDim.x + threadIdx.x;
 
    for(gdf_index_type idx = start_idx; idx < point_size; idx += blockDim.x * gridDim.x)
    {
        T point_lat = point_lats[start_idx];
        T point_lon = point_lons[start_idx];
        gdf_size_type count = 0;
 
        for(gdf_index_type poly_idx = 0; poly_idx < poly_size - 1; poly_idx++) 
        {
            if(poly_lons[poly_idx] <= point_lon && point_lon < poly_lons[poly_idx + 1])
            {
                if (orientation(poly_lons[poly_idx], poly_lats[poly_idx], poly_lons[poly_idx + 1], poly_lats[poly_idx + 1], point_lon, point_lat) > 0)
                {
                    count++;
                }
            }
            else if (point_lon <= poly_lons[poly_idx] && poly_lons[poly_idx + 1] < point_lon) 
            {
                if (orientation(poly_lons[poly_idx], poly_lats[poly_idx], poly_lons[poly_idx + 1], poly_lats[poly_idx + 1], point_lon, point_lat) > 0)
                {
                    count++;
                }
            }
        }
        if ((count > 0) && (count % 2 == 0)) point_is_in_polygon[start_idx] = 1;
        else point_is_in_polygon[start_idx] = 0;
    }
}

struct point_in_polygon_functor {
    template <typename col_type>
    static constexpr bool is_supported()
    {
        return std::is_arithmetic<col_type>::value ;
    }
 
    template <typename col_type, typename std::enable_if< is_supported<col_type>() >::type* = nullptr>
    void operator()(const gdf_column* d_poly_lats,
                    const gdf_column* d_poly_lons,
                    const gdf_column* d_point_lats,
                    const gdf_column* d_point_lons,
                    gdf_column* d_point_is_in_polygon)
    {

        gdf_size_type min_grid_size = 0, block_size = 0;
        CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, point_in_polygon_kernel<col_type>) );

        point_in_polygon_kernel<col_type> <<< min_grid_size, block_size >>> (static_cast<col_type*>(d_poly_lats->data), 
                static_cast<col_type*>(d_poly_lons->data), static_cast<col_type*>(d_point_lats->data), static_cast<col_type*>(d_point_lons->data), 
                d_poly_lats->size, d_point_lats->size, static_cast<int8_t*>(d_point_is_in_polygon->data) );
    }

    template <typename col_type, typename std::enable_if< !is_supported<col_type>() >::type* = nullptr>
    void operator()(const gdf_column* d_poly_lats,
        const gdf_column* d_poly_lons,
        const gdf_column* d_point_lats,
        const gdf_column* d_point_lons,
        gdf_column* d_point_is_in_polygon)
    {
        CUDF_FAIL("Non-arithmetic operation is not supported");
    }
};
    
}   //  namespace

namespace cudf {
namespace gis {

gdf_column* gdf_point_in_polygon(const gdf_column* polygon_lats, const gdf_column* polygon_lons, const gdf_column* point_lats, const gdf_column* point_lons)
{
    cudaStream_t stream;
    CUDA_TRY( cudaStreamCreate(&stream) );
        
    CUDF_EXPECTS(polygon_lats != nullptr && polygon_lons != nullptr, "polygon data cannot be empty");
    CUDF_EXPECTS(point_lats != nullptr && point_lons != nullptr, "query points are empty");
    CUDF_EXPECTS(polygon_lats->size == polygon_lons->size, "polygon size doesn't match");
    CUDF_EXPECTS(point_lats->size == point_lons->size, "query points size doesn't match");
    CUDF_EXPECTS(polygon_lats->dtype == polygon_lons->dtype, "polygon type doesn't match");
    CUDF_EXPECTS(polygon_lats->dtype == point_lats->dtype, "type of query points and polygon doesn't match");
    CUDF_EXPECTS(point_lats->dtype == point_lons->dtype, "query points type doesn't match");
    CUDF_EXPECTS(polygon_lons->null_count == 0 && polygon_lats->null_count == 0, "polygon should not contain nulls");

    gdf_column* inside_polygon = new gdf_column;
    int8_t* data;
    RMM_TRY(RMM_ALLOC(&data, point_lats->size * sizeof(int8_t), 0));
    gdf_column_view(inside_polygon, data, nullptr, point_lats->size, GDF_INT8);

    cudf::type_dispatcher(polygon_lats->dtype,
                        point_in_polygon_functor(),
                        polygon_lats,
                        polygon_lons,
                        point_lats,
                        point_lons,
                        inside_polygon);
    
    if (point_lats->null_count == 0 && point_lons->null_count == 0) inside_polygon->null_count = 0;
    else {
        auto error_copy_bit_mask = bit_mask::copy_bit_mask( reinterpret_cast<bit_mask::bit_mask_t*>(inside_polygon->valid),
        reinterpret_cast<bit_mask::bit_mask_t*>(point_lats->valid), point_lats->size, cudaMemcpyDeviceToDevice );

        gdf_size_type null_count;
        auto err = apply_bitmask_to_bitmask(null_count, inside_polygon->valid, inside_polygon->valid, point_lons->valid, stream, inside_polygon->size);
        inside_polygon->null_count = null_count;
    }

    CUDA_TRY( cudaStreamDestroy(stream) );

    return inside_polygon;
}
}   // namespace gis

gdf_column* gdf_point_in_polygon(const gdf_column* polygon_lats, const gdf_column* polygon_lons, const gdf_column* point_lats, const gdf_column* point_lons)
{
    return gis::gdf_point_in_polygon(polygon_lats, polygon_lons, point_lats, point_lons);;
}

}   // namespace cudf