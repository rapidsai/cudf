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
#include "gtest/gtest.h"

#include <tests/groupby/groupby_test_helpers.cuh>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/column_wrapper.cuh>
#include <vector>

template <typename T>
struct PIPTest : public GdfTest 
{
    std::vector<T> polygon_lats, polygon_lons, point_lats, point_lons;

    // To validate input data
    void validation( const std::vector<T> & polygon_lats_v,
                     const std::vector<T> & polygon_lons_v,
                     const std::vector<T> & point_lats_v,
                     const std::vector<T> & point_lons_v,
                     bool print = false )
    {
        size_t min_polygon_size_accepted = 4;
        
        EXPECT_LT( min_polygon_size_accepted, polygon_lats_v.size() ) << "TEST: Unbuilt polygon";
        EXPECT_EQ( polygon_lats_v.size(), polygon_lons_v.size() ) << "TEST: Polygon size doesn't match.";
        EXPECT_EQ( point_lats_v.size(), point_lons_v.size() ) << "TEST: Points size doesn't match." ;
        
        // Polygon must be closed
        size_t size_polygon = polygon_lats_v.size();
        EXPECT_EQ( polygon_lats_v[0], polygon_lats_v[size_polygon - 1] ) << "TEST: Latitudes of polygon must be closed.";
        EXPECT_EQ( polygon_lats_v[0], polygon_lats_v[size_polygon- 1] ) << "TEST: Longitudes of polygon must be closed.";

        if(print)
        {
            std::cout << "\nSize of polygon: " << polygon_lats_v.size() << std::endl;
            std::cout << "Number of points: " << point_lats_v.size() << std::endl;
        }
    }

    void create_random_data( const std::initializer_list<T> & polygon_lats_list,
                             const std::initializer_list<T> & polygon_lons_list,
                             const int size_points, bool print = false )
    {
        polygon_lats = polygon_lats_list;
        polygon_lons = polygon_lons_list;

        const T min_value_lats = -90, max_value_lats = 90;
        const T min_value_lons = -180, max_value_lons = 180;

        for (int i = 0; i < size_points; ++i) {
            RandomValues<T>random_lat(min_value_lats, max_value_lats);
            point_lats.push_back(random_lat());
            RandomValues<T>random_lon(min_value_lons, max_value_lons);
            point_lons.push_back(random_lon());
        }

        validation(polygon_lats, polygon_lons, point_lats, point_lons, print);
    }

    void create_input( const std::initializer_list<T> & polygon_lats_list,
                       const std::initializer_list<T> & polygon_lons_list,
                       const std::initializer_list<T> & point_lats_list,
                       const std::initializer_list<T> & point_lons_list,
                       bool print = false )
    {
        polygon_lats = polygon_lats_list;
        polygon_lons = polygon_lons_list;
        point_lats = point_lats_list;
        point_lons = point_lons_list;

        validation(polygon_lats, polygon_lons, point_lats, point_lons, print);
    }

    // Host function to check the orientation of point p3 relative to the vector from point p1 to p2
    T orientation(T p1_x, T p1_y, T p2_x, T p2_y, T p3_x, T p3_y)
    {
	    return ((p2_y - p1_y) * (p3_x - p2_x) - (p2_x - p1_x) * (p3_y - p2_y));
    }

    // Host point in polygon function
    std::vector<int8_t> compute_reference_pip()
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
                    } else {
                        count--;
                    }
                }
                else if (point_lon <= polygon_lons[poly_idx] && polygon_lons[poly_idx + 1] < point_lon) 
                {
                    if (orientation(polygon_lons[poly_idx], polygon_lats[poly_idx], polygon_lons[poly_idx + 1], polygon_lats[poly_idx + 1], point_lon, point_lat) > 0)
                    {
                        count++;
                    } else {
                        count--;
                    }
                }
            }
            h_inside_polygon[id_point] = (count > 0) ? 1 : 0;
        }

        return h_inside_polygon;
    }

    std::vector<int8_t> compute_gdf_pip()
    {
        cudf::test::column_wrapper<T> polygon_lat_wrapp{polygon_lats}, polygon_lon_wrapp{polygon_lons};
        cudf::test::column_wrapper<T> point_lat_wrapp{point_lats}, point_lon_wrapp{point_lons};

        gdf_column inside_poly_col = cudf::point_in_polygon( *(polygon_lat_wrapp.get()), *(polygon_lon_wrapp.get()), 
                                                             *(point_lat_wrapp.get()), *(point_lon_wrapp.get()) );

        std::vector<int8_t> host_inside_poly(point_lats.size());

        EXPECT_EQ(cudaMemcpy(host_inside_poly.data(), inside_poly_col.data, inside_poly_col.size * sizeof(int8_t), cudaMemcpyDeviceToHost), cudaSuccess);
    
        return host_inside_poly;
    }
};

// Geographical data are real numeric (int8_t is lesser than the min_max longitude value)
typedef testing::Types<int16_t, int32_t, int64_t, float, double> NumericTypes;

TYPED_TEST_CASE(PIPTest, NumericTypes);

// All points should be inside the polygon with clockwise orientation
TYPED_TEST(PIPTest, InsidePolygonClockwise)
{
    // Latitudes polygon, longitudes polygon, latitudes of query points, longitudes of query points, print = false
    this->create_input({-10, -10, 10, 10, -10}, {10, -10, -10, 10, 10}, {4, 5, 2, 6, 8, -4, -2}, {2, 6, 5, 8, -7, -3, 7});

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    // Compare the GDF and reference solutions
    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);   
    }
}

// All points should be outside the polygon with clockwise orientation
TYPED_TEST(PIPTest, OutsidePolygonClockwise)
{
    this->create_input({-10, 10, 10, -10, -10}, {-10, -10, 10, 10, -10}, {-14, 15, 12, -18, 5}, {2, 6, 4, 3, 18});

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);   
    }
}

// All points should be inside the polygon with counter-clockwise orientation
TYPED_TEST(PIPTest, InsidePolygonCounterClockwise)
{
    this->create_input({-10, -10, 10, 10, -10}, {-10, 10, 10, -10, -10}, {4, 5, 2, 6, 2, 8}, {2, 6, 5, 8, 1, 3});

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

// All points should be outside the polygon with counter-clockwise orientation
TYPED_TEST(PIPTest, OutsidePolygonCounterClockwise)
{
    this->create_input({-10, 10, 10, -10, -10}, {10, 10, -10, -10, 10},{-14, 15, 12, -18, 5}, {2, 6, 4, 3, 18});

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);   
    }
}

// Some points should be inside and other outside the polygon with clockwise orientation
TYPED_TEST(PIPTest, MixedUpPointsInPolygonClockwise)
{
    this->create_input({-10, 10, 10, -10, -10}, {-10, -10, 10, 10, -10}, {4, 15, -2, 6, 2, -28}, {2, 6, 5, 11, 9, 3});

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);   
    }
}

// Some points should be inside and other outside the polygon with counter-clockwise orientation
TYPED_TEST(PIPTest, MixedUpPointsInPolygonCounterClockwise)
{
    this->create_input({-10, -10, 10, 10, -10}, {-10, 10, 10, -10, -10}, {4, 15, -2, 6, 2, -28}, {2, 6, 5, 11, 9, 3});

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

// Some random points should be inside and other outside the polygon with clockwise orientation
TYPED_TEST(PIPTest, RandomPointsMixedUpInPolygonClockwise)
{
    const int N_RANDOM_SIZE = 1000;
    this->create_random_data( {-10, 25, 25, 10, 10, -10, -10, -25, -25, -10},
                              {-50, -30, 20, 40, 5, 5, 40, 20, -30, -50}, 
                              N_RANDOM_SIZE );

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

// Some random points should be inside and other outside the polygon with counter-clockwise orientation
TYPED_TEST(PIPTest, RandomPointsMixedUpInPolygonCounterClockwise)
{
    const int N_RANDOM_SIZE = 1000;
    this->create_random_data( {-10, -25, -25, -10, -10, 10, 10, 25, 25, -10},
                              {-50, -30, 20, 40, 5, 5, 40, 20, -30, -50},
                              N_RANDOM_SIZE );

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}
