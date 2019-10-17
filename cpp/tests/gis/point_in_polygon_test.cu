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
#include <gtest/gtest.h>

#include <tests/groupby/without_agg/groupby_test_helpers.cuh>
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
        
        EXPECT_LE( min_polygon_size_accepted, polygon_lats_v.size() ) << "TEST: Unbuilt polygon";
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

    // It Initialize random query points with a specific range
    void initialize_random( const std::initializer_list<T> & polygon_lats_list,
                            const std::initializer_list<T> & polygon_lons_list,
                            const int & size_points, const T & min_value_lats,
                            const T & max_value_lats, const T & min_value_lons,
                            const T & max_value_lons, bool print = false )
    {
        polygon_lats = polygon_lats_list;
        polygon_lons = polygon_lons_list;

        for (int i = 0; i < size_points; ++i) {
            RandomValues<T>random_lat(min_value_lats, max_value_lats);
            point_lats.push_back(random_lat());
            RandomValues<T>random_lon(min_value_lons, max_value_lons);
            point_lons.push_back(random_lon());
        }

        validation(polygon_lats, polygon_lons, point_lats, point_lons, print);
    }

    // It Initialize query points with a initializer list
    void set_initialize( const std::initializer_list<T> & polygon_lats_list,
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
                    T orientation_ = orientation(polygon_lons[poly_idx], polygon_lats[poly_idx], polygon_lons[poly_idx + 1], polygon_lats[poly_idx + 1], point_lon, point_lat);
                
                    if (orientation_ > 0) count++;
                    else if (orientation_ < 0) count--;
                }
                else if (point_lon <= polygon_lons[poly_idx] && polygon_lons[poly_idx + 1] < point_lon) 
                {
                    T orientation_ = orientation(polygon_lons[poly_idx], polygon_lats[poly_idx], polygon_lons[poly_idx + 1], polygon_lats[poly_idx + 1], point_lon, point_lat);
                
                    if (orientation_ > 0) count++;
                    else if (orientation_ < 0) count--;
                }
            }
            h_inside_polygon[id_point] = ( (count != 0) && (count % 2 == 0) ) ? 1 : 0;
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

// Geographical data are numeric
typedef testing::Types<int16_t, int32_t, int64_t, float, double> NumericTypes;

TYPED_TEST_CASE(PIPTest, NumericTypes);

// All points should be inside the polygon with clockwise orientation.
TYPED_TEST(PIPTest, InsidePolygonClockwise)
{
    const int n_random_points = 1000;
    
    // static polygon and random points (range latitudes: [-9 , 9], range longitudes: [-9, 9]) 
    this->initialize_random( {-10, -10, 10, 10, -10}, {10, -10, -10, 10, 10}, n_random_points, -9, 9, -9, 9 );

    std::vector<int8_t> reference_pip_result(n_random_points, 1);
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    // Compare the GDF and reference solutions
    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);   
    }
}

// All points should be outside the polygon with clockwise orientation.
TYPED_TEST(PIPTest, OutsidePolygonClockwise)
{
    const int n_random_points = 1000;
    this->initialize_random( {-10, -10, 10, 10, -10}, {10, -10, -10, 10, 10}, n_random_points, -90, -11, -180, -11 );

    std::vector<int8_t> reference_pip_result(n_random_points, 0);
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);   
    }
}

// All points should be inside the polygon with counter-clockwise orientation.
TYPED_TEST(PIPTest, InsidePolygonCounterClockwise)
{
    const int n_random_points = 1000;
    this->initialize_random( {-10, -10, 10, 10, -10}, {-10, 10, 10, -10, -10}, n_random_points, -9, 9, -9, 9 );

    std::vector<int8_t> reference_pip_result(n_random_points, 1);;
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

// All points should be outside the polygon with counter-clockwise orientation.
TYPED_TEST(PIPTest, OutsidePolygonCounterClockwise)
{
    const int n_random_points = 1000;
    this->initialize_random( {-10, -10, 10, 10, -10}, {-10, 10, 10, -10, -10}, n_random_points, -90, -11, -180, -11 );

    std::vector<int8_t> reference_pip_result(n_random_points, 0);
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);   
    }
}

// Some points should be inside and other outside the polygon with clockwise orientation.
TYPED_TEST(PIPTest, MixedUpPointsInPolygonClockwise)
{
    const int n_random_points = 1000;
    this->initialize_random( {-10, 10, 10, -10, -10}, {-10, -10, 10, 10, -10}, n_random_points, -90, 90, -180, 180 );

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);   
    }
}

// Some points should be inside and other outside the polygon with counter-clockwise orientation.
TYPED_TEST(PIPTest, MixedUpPointsInPolygonCounterClockwise)
{
    const int n_random_points = 1000;
    this->initialize_random( {-10, -10, 10, 10, -10}, {-10, 10, 10, -10, -10}, n_random_points, -90, 90, -180, 180 );

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

// Some points should be inside and other outside the bigger polygon with clockwise orientation.
TYPED_TEST(PIPTest, MixedUpPointsInBiggerPolygonClockwise)
{
    const int n_random_points = 1000;
    this->initialize_random( {-35, -15, 5, 25, 35, 20, 30, 40, 15, 5, -10, -5, -25, -35},
                             {5, -30, -40, -35, -20, -5, 10, 25, 40, 15, 5, 35, 20, 5}, 
                             n_random_points, -90, 90, -180, 180 );

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

// Some points should be inside and other outside the bigger polygon with counter-clockwise orientation.
TYPED_TEST(PIPTest, MixedUpPointsInBiggerPolygonCounterClockwise)
{
    const int n_random_points = 1000;
    this->initialize_random( {-35, -25, -5, -10, 5, 15, 40, 30, 20, 35, 25, 5, -15, -35},
                             {5, 20, 35, 5, 15, 40, 25, 10, -5, -20, -35, -40, -30, 5},
                             n_random_points, -90, 90, -180, 180 );

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

/* Case: Degenerate polygon P {P0, P0 ,P0 , ..., P0} (area = 0) all points are equal. 
*  Query points {q1, q2, ...} are random.
*  Result: No points must be inside the polygon.
*/
TYPED_TEST(PIPTest, DegeneratePolygonWithMixedUpPoints)
{
    const int n_random_points = 1000;
    this->initialize_random( {10, 10, 10, 10, 10, 10}, {20, 20, 20, 20, 20, 20}, n_random_points, -90, 90, -180, 180 );

    std::vector<int8_t> reference_pip_result(n_random_points, 0) ;
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}
 
/* Case: Degenerate polygon P {P0, P0 ,P0 , ..., P0} (area = 0) all points are equal. 
*  Query point q {P0} is equal to the degenerate polygon.
*  Result: query point q must be outside the polygon.
*/
TYPED_TEST(PIPTest, DegeneratePolygonWithSamePoint)
{
    this->set_initialize( {10, 10, 10, 10, 10, 10}, {20, 20, 20, 20, 20, 20}, {10}, {20} );

    std::vector<int8_t> reference_pip_result(1, 0);
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

/* Case: Degenerate Polygon P {P0, P1, P2, P0} with colinear coordinates.
*  It is degenerate because it must be a multiline.
*  Query points {q1, q2, q3, ...} are all random.
*
*            P2
*    .q1    /
*          / .q2
*         /
*        /
*       P1       .q3
*      /
*     /
*   P0
*
* Result: query points must be outside the polygon.
*/
TYPED_TEST(PIPTest, ColinearPolygonWithMixedUpPoints)
{
    const int n_random_points = 1000;
    this->initialize_random( {10, 15, 20, 25, 30, 10}, {10, 15, 20, 25, 30, 10}, n_random_points, -90, 90, -180, 180 );

    std::vector<int8_t> reference_pip_result = this->compute_reference_pip();
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

/* Case: Degenerate Polygon P {P0, P1, P2, P0} with colinear coordinates.
*  It is degenerate because it must be a multiline.
*  Two Query points {q1, q2} are colinear with the Polygon.
*
*              P2
*             /
*            .q2
*           /
*          /
*         P1
*        /
*       /
*      .q1
*     /
*    P0
*
* Result: query points must be outside the polygon.
*/
TYPED_TEST(PIPTest, ColinearPolygonWithColinearPoint)
{
    this->set_initialize( {10, 15, 20, 10}, {10, 15, 20, 10}, {12, 18}, {12, 18} );

    std::vector<int8_t> reference_pip_result(2, 0);
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

/* Case: Degenerate polygon P {P0, P1, P2, P3, P0} intersects with itself.
*  Three query points are considered {q1, q2, q3}.
* 
*        P0        P2
*        | \      / |
*        |  \    /  |
*        |   \  /   |
*   .q1  |.q2 \/    |  .q3
*        |    /\    |
*        |   /  \   |
*        |  /    \  |
*        | /      \ |
*        P3        P1
*
* Result: q2 must be inside P while q1 and q3 must be outside P.
** For this polygon there is no problem with the algorithm.
*/
TYPED_TEST(PIPTest, PolygonIntersectsWithItselfPassed)
{
    this->set_initialize( {10, -10, 10, -10, 10}, {-10, 10, 10, -10, -10}, {0, 0, 0}, {-20, -8, 20} );

    std::vector<int8_t> reference_pip_result{0, 1, 0};
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}

/* Case: Degenerate polygon P {P0, P1, P2, P3, P4, P5, P6, P7, P0} intersects with itself.
*  This polygon is more complex than the previous one.
*  Five query points are considered {q1, q2, q3, q4, q5}.
*
* P0 ------------------- P1
* |                .q2   |
* |                      |
* |         P4 ------------------- P5
* |         |    .q3     |          |
* |         |            |   .q4    |
* |         P3 -------- P2          |   .q5
* |   .q1                           |
* |                                 |
* P7 ----------------------------- P6
*
*
* Result: {q1, q2, q4} must be inside and {q3, q5} must be outside P.
** For this more complex polygon threre is a problem with the algorithm
** The algorithm consider q3 as inside which es wrong.
** The real ouptut must be: [1, 1, 0, 1, 0]
** 
** This algorithm was designed to non convex polygon.
** Also, support some degenerate cases as before TESTS.
** Unfortunately the algorithm doesn't work well for polygons that
** that intersects with itself as this TEST demonstrate.
**
** So for the point_in_polygon algorithm the polygon must not has intersections with itself.
**
*/
TYPED_TEST(PIPTest, PolygonIntersectsWithItselfNotPassed)
{
    this->set_initialize( {15, 15, 5, 5, 10, 10, 0, 0, 15}, {0, 10, 10, 5, 5, 15, 15, 0, 0}, {3, 12, 8, 7, 6}, {2, 8, 7, 12, 18} );

    std::vector<int8_t> reference_pip_result{1, 1, 1, 1, 0};
    std::vector<int8_t> gdf_pip_result = this->compute_gdf_pip();

    ASSERT_EQ(reference_pip_result.size(), gdf_pip_result.size()) << "Size of gdf result doesn't match with reference result";

    for(size_t i = 0; i < reference_pip_result.size(); ++i) {
        EXPECT_EQ(reference_pip_result[i], gdf_pip_result[i]);
    }
}
