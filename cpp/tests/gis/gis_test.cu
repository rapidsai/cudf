#include <bitmask/BitMask.cuh>
#include "gtest/gtest.h"
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>  // GdfTest
#include <cudf.h>     // create_gdf_column
#include <vector>

template <typename T>
struct GISTest : public GdfTest 
{
    std::vector<T> polygon_lats;
    std::vector<T> polygon_lons;
    std::vector<T> point_lats;
    std::vector<T> point_lons;
    std::vector<int32_t> inside_polygon;
    size_t total_points;

    gdf_col_pointer gdf_col_ptr_polygon_lats;
    gdf_col_pointer gdf_col_ptr_polygon_lons;
    gdf_col_pointer gdf_col_ptr_point_lats;
    gdf_col_pointer gdf_col_ptr_point_lons;
    gdf_col_pointer gdf_col_ptr_inside_points;

    gdf_column* gdf_raw_polygon_lats;
    gdf_column* gdf_raw_polygon_lons;
    gdf_column* gdf_raw_point_lats;
    gdf_column* gdf_raw_point_lons;
    gdf_column* gdf_raw_inside_polygon;

    // TODO: comment params
    void create_input(const std::initializer_list<T> &column_polygon_lats_list,
        const std::initializer_list<T> &column_polygon_lons_list,
        const std::initializer_list<T> &column_point_lats_list,
        const std::initializer_list<T> &column_point_lons_list,
        bool print = false)
    {
        polygon_lats = column_polygon_lats_list;
        polygon_lons = column_polygon_lons_list;
        point_lats = column_point_lats_list;
        point_lons = column_point_lons_list;

        if (polygon_lats.size() != polygon_lons.size())
        {
            std::cerr << "Polygon size doesn't match." << std::endl;
            return;
        }
        
        if (point_lats.size() != point_lons.size())
        {
            std::cerr << "Points size doesn't match." << std::endl;
            return;
        }

        total_points = point_lats.size();
        inside_polygon.resize(total_points, -1);

        gdf_col_ptr_polygon_lats = create_gdf_column(polygon_lats);
        gdf_col_ptr_polygon_lons = create_gdf_column(polygon_lons);
        gdf_col_ptr_point_lats = create_gdf_column(point_lats);
        gdf_col_ptr_point_lons = create_gdf_column(point_lons);
        gdf_col_ptr_inside_points = create_gdf_column(inside_polygon);

        gdf_raw_polygon_lats = gdf_col_ptr_polygon_lats.get();
        gdf_raw_polygon_lons = gdf_col_ptr_polygon_lons.get();
        gdf_raw_point_lats = gdf_col_ptr_point_lats.get();
        gdf_raw_point_lons = gdf_col_ptr_point_lons.get();
        gdf_raw_inside_polygon = gdf_col_ptr_inside_points.get();

        if(print)
        {
            std::cout << "\nSize of polygon: " << polygon_lats.size() << std::endl;
            std::cout << "Number of points: " << point_lats.size() << std::endl;
        }
    }

    // TODO: Implement  pip host
    std::vector<int32_t> compute_reference_pip(bool print = false)
    {
        // todo: pip host
        std::vector<int32_t> h_inside_polygon(total_points, 0);//default 0


        if(print)
        {
            std::cout << "\nReference result: " << std::endl;
            print_vector(h_inside_polygon);
            std::cout << std::endl;;
        }

        return h_inside_polygon;
    }

    std::vector<int32_t> compute_gdf_pip(bool print = false)
    {
        gdf_point_in_polygon(gdf_raw_polygon_lats, gdf_raw_polygon_lons, gdf_raw_point_lats, gdf_raw_point_lons, gdf_raw_inside_polygon);
      
        size_t output_size = gdf_raw_point_lats->size;
        std::vector<int32_t> host_inside_polygon(output_size);
      
        EXPECT_EQ(cudaMemcpy(host_inside_polygon.data(), gdf_raw_inside_polygon->data, output_size * sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);

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


using Types = testing::Types<float, double>;

TYPED_TEST_CASE(GISTest, Types);

TYPED_TEST(GISTest, InsidePolygon)
{
    // Latitudes polygon, longitudes polygon, latitudes of query points, longitudes of query points, print = false
    this->create_input({0.0, 1.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 1.0, 0.0}, {0.4, 0.5, 0.2, 0.6}, {0.2, 0.6, 0.5, 0.8}, false);

    std::vector<int32_t> reference_pip_result = this->compute_reference_pip(true);
    std::vector<int32_t> gdf_pip_result = this->compute_gdf_pip(true);

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match";

    // Compare the GDF and reference solutions
    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

TYPED_TEST(GISTest, OutsidePolygon)
{
    this->create_input({0.0, 1.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 1.0, 0.0}, {-0.4, -0.5, -0.2}, {0.2, 0.6, 0.5}, false);

    std::vector<int32_t> reference_pip_result = this->compute_reference_pip(true);
    std::vector<int32_t> gdf_pip_result = this->compute_gdf_pip(true);

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match";

    // Compare the GDF and reference solutions
    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

TYPED_TEST(GISTest, EmptyPolygon)
{

}