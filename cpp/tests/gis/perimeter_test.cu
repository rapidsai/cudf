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
 
 // useful GIS constants
const double DEG_TO_RAD = 0.01745329251994329;
const double EARTH_RADIUS = 6372797.560856; // in meters

template <typename T>
struct PerimeterTest : public GdfTest 
{
    // To validate input data
    void validation(std::vector<T> poly_concat_lats[], std::vector<T> poly_concat_lons[], int n_polygons)
    {
        size_t min_polygon_size_accepted = 4;

        for (int i = 0; i < n_polygons; ++i)
        {
            EXPECT_LE( min_polygon_size_accepted, poly_concat_lats[i].size() ) << "TEST: Unbuilt polygon";
            EXPECT_LE( min_polygon_size_accepted, poly_concat_lons[i].size() ) << "TEST: Unbuilt polygon";

            // Polygon must be closed
            size_t size_polygon_i = poly_concat_lats[i].size();
            EXPECT_EQ( poly_concat_lats[i][0], poly_concat_lats[i][size_polygon_i - 1] ) << "TEST: Latitudes of polygon must be closed.";
            EXPECT_EQ( poly_concat_lons[i][0], poly_concat_lons[i][size_polygon_i - 1] ) << "TEST: Longitudes of polygon must be closed.";
        }
    }

    // Initialize random polygons with a specific range
    void initialize_random( std::vector<T> poly_concat_lats[], std::vector<T>  poly_concat_lons[], 
                            const int & n_polygons, const int & size_polygon, const T & min_value_lats,
                            const T & max_value_lats, const T & min_value_lons, const T & max_value_lons,
                            bool print = false )
    {
        for (int i = 0; i < n_polygons; ++i)
        {
            poly_concat_lats[i].resize(size_polygon);
            poly_concat_lons[i].resize(size_polygon);
        }

        for (int i = 0; i < n_polygons; ++i)
        {
            for (int j = 0; j < size_polygon; ++j)
            {
                RandomValues<T>random_lat(min_value_lats, max_value_lats);
                poly_concat_lats[i][j] = random_lat();
                RandomValues<T>random_lon(min_value_lons, max_value_lons);
                poly_concat_lons[i][j] = random_lon();
            }
        }

        // First and last points of each polygon must be equal
        for (int i = 0; i < n_polygons; ++i)
        {
            poly_concat_lats[i][size_polygon - 1] = poly_concat_lats[i][0];
            poly_concat_lons[i][size_polygon - 1] = poly_concat_lons[i][0];
        }
    }

    T haversine_formula (const T current_latitude, const T current_longitude,
                         const T next_latitude, const T next_longitude)
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

    std::vector<T> compute_reference_perimeter(std::vector<T> poly_concat_lats[], std::vector<T> poly_concat_lons[], int n_polygons)
    {
        validation(poly_concat_lats, poly_concat_lons, n_polygons);
        std::vector<T> host_perimeter(n_polygons);
        
        for (int poly_i = 0; poly_i < n_polygons; ++poly_i)
        {
            int size_poly_i = poly_concat_lats[poly_i].size();
            host_perimeter[poly_i] = 0.0;
            for (int side_i = 0; side_i < size_poly_i - 1; ++side_i)
            {
                host_perimeter[poly_i] += haversine_formula(poly_concat_lats[poly_i][side_i], poly_concat_lons[poly_i][side_i],
                    poly_concat_lats[poly_i][side_i + 1], poly_concat_lons[poly_i][side_i + 1]);
            }
        }
        
        return host_perimeter;
    }

    std::vector<T> compute_gdf_perimeter(std::vector<T> poly_concat_lats[], std::vector<T> poly_concat_lons[], int n_polygons)
    {
        std::vector<T> host_perimeter(n_polygons);

        std::vector<gdf_column*> ptr_to_concat_lats, ptr_to_concat_lons;
        cudf::test::column_wrapper<T>* wrapper_lats[n_polygons];
        cudf::test::column_wrapper<T>* wrapper_lons[n_polygons];

        for (int i = 0; i < n_polygons; ++i)
        {
            wrapper_lats[i] = new cudf::test::column_wrapper<T>(poly_concat_lats[i]);
            wrapper_lons[i] = new cudf::test::column_wrapper<T>(poly_concat_lons[i]);

            ptr_to_concat_lats.push_back(wrapper_lats[i]->get());
            ptr_to_concat_lons.push_back(wrapper_lons[i]->get());
        }
        
        gdf_column gdf_perimeter = cudf::perimeter(ptr_to_concat_lats.data(), ptr_to_concat_lons.data(), n_polygons);
        EXPECT_EQ(cudaMemcpy(host_perimeter.data(), gdf_perimeter.data, n_polygons * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);

        return host_perimeter;
    } 
};

using FloatingTypes = testing::Types<double>;

TYPED_TEST_CASE(PerimeterTest, FloatingTypes);

TYPED_TEST(PerimeterTest, SinglePolygon)
{
    std::vector<double> poly_latitudes {-10, 10, 10, -10, -10}, poly_longitudes {-10, -10, 10, 10, -10};
    int num_polygons = 1;

    std::vector<double> poly_concat_lats[num_polygons], poly_concat_lons[num_polygons];
    poly_concat_lats[0] = poly_latitudes, poly_concat_lons[0] = poly_longitudes;

    std::vector<double> reference_perimeter = this->compute_reference_perimeter(poly_concat_lats, poly_concat_lons, num_polygons);
    std::vector<double> gdf_perimeter = this->compute_gdf_perimeter(poly_concat_lats, poly_concat_lons, num_polygons);

    ASSERT_EQ(reference_perimeter.size(), gdf_perimeter.size()) << "Size of gdf result doesn't match with reference result";

    // Compare the GDF and reference solutions with a small error of compute
    for (size_t i = 0; i < reference_perimeter.size(); ++i) {
        EXPECT_LT(reference_perimeter[i] - gdf_perimeter[i], 0.00001);   
    }
}

/* The second polygon is degenerate, all points are the same
 * There is not problem with this case
 */
TYPED_TEST(PerimeterTest, MultiplePolygonWithADegeneratePolygon)
{
    std::vector<double> poly1_lats {-10, 10, -10, -10}, poly1_lons {-10, 0, 10, -10};
    std::vector<double> poly2_lats {0, 0, 0, 0, 0}, poly2_lons {0, 0, 0, 0, 0};
    int num_polygons = 2;

    std::vector<double> poly_concat_lats[num_polygons], poly_concat_lons[num_polygons];
    poly_concat_lats[0] = poly1_lats, poly_concat_lons[0] = poly1_lons;
    poly_concat_lats[1] = poly2_lats, poly_concat_lons[1] = poly2_lons;

    std::vector<double> reference_perimeter = this->compute_reference_perimeter(poly_concat_lats, poly_concat_lons, num_polygons);
    std::vector<double> gdf_perimeter = this->compute_gdf_perimeter(poly_concat_lats, poly_concat_lons, num_polygons);

    ASSERT_EQ(reference_perimeter.size(), gdf_perimeter.size()) << "Size of gdf result doesn't match with reference result";

    for (size_t i = 0; i < reference_perimeter.size(); ++i) {
        EXPECT_LT(reference_perimeter[i] - gdf_perimeter[i], 0.00001);  
    }
}

// Polygons generate manually
TYPED_TEST(PerimeterTest, MultiplePolygons)
{
    std::vector<double> poly1_lats {-10, 10, -10, -10}, poly1_lons {-10, 0, 10, -10};
    std::vector<double> poly2_lats {-20, 20, 20, -20, -20}, poly2_lons {-20, -20, 20, 20, -20};
    std::vector<double> poly3_lats {-10, 10, 10, -10, 23, -10}, poly3_lons {-10, -10, 10, 10, 52, -10};
    std::vector<double> poly4_lats {-10, 50, 22, 10, 10, -10, 23, -10}, poly4_lons {-10, 17, -16, -10, 10, 10, 52, -10};
    int num_polygons = 4;

    std::vector<double> poly_concat_lats[num_polygons], poly_concat_lons[num_polygons];
    poly_concat_lats[0] = poly1_lats; poly_concat_lats[1] = poly2_lats; poly_concat_lats[2] = poly3_lats; poly_concat_lats[3] = poly4_lats;
    poly_concat_lons[0] = poly1_lons; poly_concat_lons[1] = poly2_lons; poly_concat_lons[2] = poly3_lons; poly_concat_lons[3] = poly4_lons;
    
    std::vector<double> reference_perimeter = this->compute_reference_perimeter(poly_concat_lats, poly_concat_lons, num_polygons);
    std::vector<double> gdf_perimeter = this->compute_gdf_perimeter(poly_concat_lats, poly_concat_lons, num_polygons);

    ASSERT_EQ(reference_perimeter.size(), gdf_perimeter.size()) << "Size of gdf result doesn't match with reference result";

    for (size_t i = 0; i < reference_perimeter.size(); ++i) {
        EXPECT_LT(reference_perimeter[i] - gdf_perimeter[i], 0.00001);   
    }
}

// Polygons generate randomly. These polygons can intersect with itself
TYPED_TEST(PerimeterTest, RandomPolygons)
{
    const int num_polygons = 1000;
    const int size_polygon = 50;
    std::vector<double> poly_concat_lats[num_polygons], poly_concat_lons[num_polygons];

    this->initialize_random(poly_concat_lats, poly_concat_lons, num_polygons, size_polygon, -90, 90, -180, 180);

    std::vector<double> reference_perimeter = this->compute_reference_perimeter(poly_concat_lats, poly_concat_lons, num_polygons);
    std::vector<double> gdf_perimeter = this->compute_gdf_perimeter(poly_concat_lats, poly_concat_lons, num_polygons);

    ASSERT_EQ(reference_perimeter.size(), gdf_perimeter.size()) << "Size of gdf result doesn't match with reference result";

    for (size_t i = 0; i < reference_perimeter.size(); ++i) {
        EXPECT_LT(reference_perimeter[i] - gdf_perimeter[i], 0.00001);   
    }
}

// A polygon generate randomly
TYPED_TEST(PerimeterTest, BigUniquePolygon)
{
    const int num_polygons = 1;
    const int size_polygon = 10000;
    std::vector<double> poly_concat_lats[num_polygons], poly_concat_lons[num_polygons];

    this->initialize_random(poly_concat_lats, poly_concat_lons, num_polygons, size_polygon, -90, 90, -180, 180);

    std::vector<double> reference_perimeter = this->compute_reference_perimeter(poly_concat_lats, poly_concat_lons, num_polygons);
    std::vector<double> gdf_perimeter = this->compute_gdf_perimeter(poly_concat_lats, poly_concat_lons, num_polygons);

    ASSERT_EQ(reference_perimeter.size(), gdf_perimeter.size()) << "Size of gdf result doesn't match with reference result";

    for (size_t i = 0; i < reference_perimeter.size(); ++i) {
        EXPECT_LT(reference_perimeter[i] - gdf_perimeter[i], 0.00001);   
    }
}

