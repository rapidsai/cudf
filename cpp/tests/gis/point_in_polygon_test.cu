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
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/column_wrapper.cuh>
#include <vector>

template <typename T>
struct PIPTest : public GdfTest 
{
    std::vector<T> polygon_lats, polygon_lons, point_lats, point_lons;

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

        size_t min_polygon_size_accepted = 4;
        
        EXPECT_LT( min_polygon_size_accepted, polygon_lats.size() ) << "TEST: Unbuilt polygon";
        EXPECT_EQ( polygon_lats.size(), polygon_lons.size() ) << "TEST: Polygon size doesn't match.";
        EXPECT_EQ( point_lats.size(), point_lons.size() ) << "TEST: Points size doesn't match." ;
        
        // Polygon must be closed
        size_t size_polygon = polygon_lats.size();
        EXPECT_EQ( polygon_lats[0], polygon_lats[size_polygon - 1] ) << "TEST: Latitudes of polygon must be closed.";
        EXPECT_EQ( polygon_lats[0], polygon_lats[size_polygon- 1] ) << "TEST: Longitudes of polygon must be closed.";

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

    // pip host function
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

        return h_inside_polygon;
    }

    std::vector<int8_t> compute_gdf_pip()
    {
        cudf::test::column_wrapper<T> polygon_lat_wrapp{polygon_lats};
        cudf::test::column_wrapper<T> polygon_lon_wrapp{polygon_lons};
        cudf::test::column_wrapper<T> point_lat_wrapp{point_lats};
        cudf::test::column_wrapper<T> point_lon_wrapp{point_lons};

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

TYPED_TEST(PIPTest, InsidePolygonClockwise)
{
    // Latitudes polygon, longitudes polygon, latitudes of query points, longitudes of query points, print = false
    this->create_input({-10, -10, 10, 10, -10}, {10, -10, -10, 10, 10}, {4, 5, 2, 6, 2, 8}, {2, 6, 5, 8, 1, 3}, false);

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    // Compare the GDF and reference solutions
    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);   
    }
}

TYPED_TEST(PIPTest, OutsidePolygonClockwise)
{
    this->create_input({-10, 10, 10, -10, -10}, {-10, -10, 10, 10, -10}, {-14, 15, 12, -18, 5}, {2, 6, 4, 3, 18}, false);

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);   
    }
}

TYPED_TEST(PIPTest, InsidePolygonCounterClockwise)
{
    this->create_input({-10, -10, 10, 10, -10}, {-10, 10, 10, -10, -10}, {4, 5, 2, 6, 2, 8}, {2, 6, 5, 8, 1, 3}, false);

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

TYPED_TEST(PIPTest, OutsidePolygonCounterClockwise)
{
    this->create_input({-10, 10, 10, -10, -10}, {10, 10, -10, -10, 10},{-14, 15, 12, -18, 5}, {2, 6, 4, 3, 18}, false);

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);   
    }
}