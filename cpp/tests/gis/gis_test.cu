#include "gtest/gtest.h"
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/column_wrapper.cuh>
#include "src/gis/gis_dispatcher.cu"
#include "data/real_polygon.hpp"
#include <vector>

template <typename T>
struct GISTest : public GdfTest 
{
    std::vector<T> polygon_lats, polygon_lons, point_lats, point_lons;
    gdf_column *gdf_column_poly_lats, *gdf_column_poly_lons, *gdf_column_point_lats, *gdf_column_point_lons;

    // From host vectors initializer
    void create_input(const std::initializer_list<T> &polygon_lats_list,
        const std::initializer_list<T> &polygon_lons_list,
        const std::initializer_list<T> &point_lats_list,
        const std::initializer_list<T> &point_lons_list,
        bool print = false)
    {
        polygon_lats = polygon_lats_list;
        polygon_lons = polygon_lons_list;
        point_lats = point_lats_list;
        point_lons = point_lons_list;

        EXPECT_EQ( polygon_lats.size(), polygon_lons.size() ) << "TEST: Polygon size doesn't match.";
        EXPECT_EQ( point_lats.size(), point_lons.size() ) << "TEST: Points size doesn't match." ;

        if(print)
        {
            std::cout << "\nSize of polygon: " << polygon_lats.size() << std::endl;
            std::cout << "Number of points: " << point_lats.size() << std::endl;
        }
    }

    // Aditional host function
    T orientation(T p1_x, T p1_y, T p2_x, T p2_y, T p3_x, T p3_y)
    {
	    return ((p2_y - p1_y) * (p3_x - p2_x) - (p2_x - p1_x) * (p3_y - p2_y));
    }

    std::vector<int8_t> compute_reference_pip(bool print = false)
    {   
        size_t total_points = point_lats.size();
        std::vector<int8_t> h_inside_polygon(total_points, -1);

        for (size_t id_point = 0; id_point < total_points; ++id_point)
        {
            T point_lat = point_lats[id_point];
            T point_lon = point_lons[id_point];
            int count = 0;

            for (size_t poly_idx = 0; poly_idx < polygon_lats.size() - 1; ++poly_idx)
            {
                if(polygon_lons[poly_idx] <= point_lon && point_lon < polygon_lons[poly_idx + 1])
                {
                    if (orientation(polygon_lons[poly_idx], polygon_lats[poly_idx], polygon_lons[poly_idx + 1], polygon_lats[poly_idx + 1], point_lon, point_lat) > 0)
                    {
                        count++;
                    }
                }
                else if (point_lon <= polygon_lons[poly_idx] && polygon_lons[poly_idx + 1] < point_lon) 
                {
                    if (orientation(polygon_lons[poly_idx], polygon_lats[poly_idx], polygon_lons[poly_idx + 1], polygon_lats[poly_idx + 1], point_lon, point_lat) > 0)
                    {
                        count++;
                    }
                }
            }

            if ((count > 0) && (count % 2 == 0)) h_inside_polygon[id_point] = 1;
		    else h_inside_polygon[id_point] = 0;
        }

        if(print)
        {
            std::cout << "\nReference result: " << std::endl;
            print_vector(h_inside_polygon);
            std::cout << std::endl;;
        }

        return h_inside_polygon;
    }

    std::vector<int8_t> compute_gdf_pip(bool print = false)
    {
        // column_wrapper for tests
        cudf::test::column_wrapper<T> polygon_lat_wrapp{polygon_lats};
        cudf::test::column_wrapper<T> polygon_lon_wrapp{polygon_lons};
        cudf::test::column_wrapper<T> point_lat_wrapp{point_lats};
        cudf::test::column_wrapper<T> point_lon_wrapp{point_lons};

        gdf_column_poly_lats = polygon_lat_wrapp.get();
        gdf_column_poly_lons = polygon_lon_wrapp.get();
        gdf_column_point_lats = point_lat_wrapp.get();
        gdf_column_point_lons = point_lon_wrapp.get();

        gdf_column* inside_polygon_column = gdf_point_in_polygon(gdf_column_poly_lats, gdf_column_poly_lons, gdf_column_point_lats, gdf_column_point_lons);

        std::vector<int8_t> host_inside_polygon(point_lats.size());

        EXPECT_EQ(cudaMemcpy(host_inside_polygon.data(), inside_polygon_column->data, inside_polygon_column->size * sizeof(int8_t), cudaMemcpyDeviceToHost), cudaSuccess);

        if(print)
        {
            std::cout << "\nGDF result: " << std::endl;
            print_vector(host_inside_polygon);
            std::cout << std::endl;;
        }
    
        return host_inside_polygon;
    }

    // TODO: Function to check the range for latitude and longitude  
};

using Types = testing::Types<double>;

TYPED_TEST_CASE(GISTest, Types);

TYPED_TEST(GISTest, InsidePolygon)
{
    // Latitudes polygon, longitudes polygon, latitudes of query points, longitudes of query points, print = false
    this->create_input({0.0, 1.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 1.0, 0.0}, {0.4, 0.5, 0.2, 0.6, 0.32, 0.78}, {0.2, 0.6, 0.5, 0.8, 0.41, 0.63}, false);

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip(false);
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip(false);

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    // Compare the GDF and reference solutions
    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
        
    }   
}

TYPED_TEST(GISTest, OutsidePolygon)
{
    this->create_input({0.0, 1.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 1.0, 0.0}, {-0.4, -0.5, -0.2, 1.25, 5.36}, {-0.2, 0.6, 0.5, 0.22, 8.21}, false);

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip(false);
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip(false);

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    // Compare the GDF and reference solutions
    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

// Query points inside Callao (polygon real) extension
TYPED_TEST(GISTest, RealPolygonInside)
{
    this->create_input(real_polygon_lats, real_polygon_lons,  {-12.003602, -12.029122, -11.955576 }, {-77.11652, -77.108792, -77.126560}, false);

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip(false);
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip(false);

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    // Compare the GDF and reference solutions
    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);  
    }   
}

// Query points outside Callao (polygon real) extension
TYPED_TEST(GISTest, RealPolygonOutside)
{
    this->create_input(real_polygon_lats, real_polygon_lons,  {-11.971293, -12.061337, -11.438918 }, {-77.106183, -77.062727, -73.642309}, false);

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip(false);
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip(false);

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    // Compare the GDF and reference solutions
    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);   
    }   
}
