/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cudf/cudf.h>
#include <cudf/legacy/table.hpp>
#include <cudf/legacy/predicates.hpp>

#include <cudf/utilities/error.hpp>

#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/column_wrapper_factory.hpp>
#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <tests/utilities/legacy/cudf_test_utils.cuh>

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

template <typename T>
struct IsSortedAlphaNum : GdfTest
{
};

using test_types =
  ::testing::Types<cudf::nvstring_category>;

TYPED_TEST_CASE(IsSortedAlphaNum, test_types);

TYPED_TEST(IsSortedAlphaNum, AlphaNumericalTrue)
{
    using T = TypeParam;
    bool found  = false;

    cudf::test::column_wrapper_factory<T> factory;
    column_wrapper <T> col1 = factory.make(4, [](auto row) { return (row); });
    column_wrapper <T> col2 = factory.make(4, [](auto row) { return (100 - row); });
    column_wrapper <int32_t> col3 = column_wrapper<int32_t>({1,1,1,1},[](auto row) { return true; });
    column_wrapper <int32_t> col4 = column_wrapper<int32_t>({4,3,2,1},[](auto row) { return true; });

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());
    cols.push_back(col3.get());
    cols.push_back(col4.get());

    cudf::table table(cols.data(), cols.size());
    EXPECT_EQ(cols.size(), static_cast<unsigned int> (table.num_columns()));
    std::vector <int8_t> const asc_dec {0,1,0,1};

    found = is_sorted (table, asc_dec, false);
    
    EXPECT_EQ(found, true);
}

TYPED_TEST(IsSortedAlphaNum, AlphaNumericalFalse)
{
    using T = TypeParam;
    bool found  = false;

    cudf::test::column_wrapper_factory<T> factory;
    column_wrapper <T> col1 = factory.make(4, [](auto row) { return (1); });
    column_wrapper <T> col2 = factory.make(4, [](auto row) { return (row); });
    column_wrapper <int32_t> col3 = column_wrapper<int32_t>({1,1,1,1},[](auto row) { return true; });
    column_wrapper <int32_t> col4 = column_wrapper<int32_t>({4,3,2,1},[](auto row) { return true; });

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());
    cols.push_back(col3.get());
    cols.push_back(col4.get());
    
    cudf::table table(cols.data(), cols.size());
    EXPECT_EQ(cols.size(), static_cast<unsigned int> (table.num_columns()));
    std::vector <int8_t> const asc_dec {0,1,0,1};
    
    found = is_sorted (table, asc_dec, false);
    
    EXPECT_EQ(found, false);
}


template <typename T>
struct IsSorted : GdfTest
{
};

using test_types_num = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float,
                                        double>;

TYPED_TEST_CASE(IsSorted, test_types_num);

TYPED_TEST(IsSorted, NumericalTrue)
{   
    using T = TypeParam;
    bool found  = false;
    
    column_wrapper <T> col1 = column_wrapper<T>({1,1,1,1},[](auto row) { return true; });
    column_wrapper <T> col2 = column_wrapper<T>({4,3,2,1},[](auto row) { return true; });
    column_wrapper <T> col3 = column_wrapper<T>({1,1,1,1},[](auto row) { return true; }); 
    column_wrapper <T> col4 = column_wrapper<T>({4,3,2,1},[](auto row) { return true; }); 
    
    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());
    cols.push_back(col3.get());
    cols.push_back(col4.get());
    
    cudf::table table(cols.data(), cols.size());
    EXPECT_EQ(cols.size(), static_cast<unsigned int> (table.num_columns()));
    std::vector <int8_t> const asc_dec {0,1,0,1};
    
    found = is_sorted (table, asc_dec, false);
    
    EXPECT_EQ(found, true);
}

TYPED_TEST(IsSorted, NumericalFalse)
{
    using T = TypeParam;
    bool found  = false;

    column_wrapper <T> col1 = column_wrapper<T>({1,1,1,1},[](auto row) { return true; }); 
    column_wrapper <T> col2 = column_wrapper<T>({4,3,2,4},[](auto row) { return true; }); 
    column_wrapper <T> col3 = column_wrapper<T>({1,1,1,1},[](auto row) { return true; }); 
    column_wrapper <T> col4 = column_wrapper<T>({4,3,2,1},[](auto row) { return true; }); 

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());
    cols.push_back(col3.get());
    cols.push_back(col4.get());

    cudf::table table(cols.data(), cols.size());
    EXPECT_EQ(cols.size(), static_cast<unsigned int> (table.num_columns()));
    std::vector <int8_t> const asc_dec {0,1,0,1};

    found = is_sorted (table, asc_dec, false);

    EXPECT_EQ(found, false);
}

TYPED_TEST(IsSorted, NumericalRowAscending)
{
    using T = TypeParam;
    bool found  = false;

    column_wrapper <T> col1 = column_wrapper<T>({1,1,1,1},[](auto row) { return true; }); 
    column_wrapper <T> col2 = column_wrapper<T>({4,3,2,1},[](auto row) { return true; });

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());

    cudf::table table(cols.data(), cols.size());
    EXPECT_EQ(cols.size(), static_cast<unsigned int> (table.num_columns()));
    std::vector <int8_t> const asc_dec {0,1};

    found = is_sorted (table, asc_dec, false);

    EXPECT_EQ(found, true);
}

TYPED_TEST(IsSorted, NumericalRowDecendingFails)
{
    using T = TypeParam;
    bool found  = false;

    column_wrapper <T> col1 = column_wrapper<T>({1,1,1,1},[](auto row) { return true; }); 
    column_wrapper <T> col2 = column_wrapper<T>({1,2,3,4},[](auto row) { return true; }); 

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());

    cudf::table table(cols.data(), cols.size());
    EXPECT_EQ(cols.size(), static_cast<unsigned int> (table.num_columns()));
    std::vector <int8_t> const asc_dec {0,1};

    found = is_sorted (table, asc_dec, false);

    EXPECT_EQ(found, false);
}

TYPED_TEST(IsSorted, NumericalNullSmallest)
{
    using T = TypeParam;
    bool found  = false;
    bool nulls_are_smallest  = true;
    
    column_wrapper <T> col1 = column_wrapper<T>({1,1,1,1},[](auto row) { return true; }); 
    column_wrapper <T> col2 = column_wrapper<T>({4,2,3,4},[](auto row) { return ((row == 0)? false : true); });

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());

    cudf::table table(cols.data(), cols.size());
    EXPECT_EQ(cols.size(), static_cast<unsigned int> (table.num_columns()));
    std::vector <int8_t> const asc_dec {0,0};

    found = is_sorted (table, asc_dec, nulls_are_smallest);

    EXPECT_EQ(found, true);
}

TYPED_TEST(IsSorted, NumericalNullBiggest)
{
    using T = TypeParam;
    bool found  = false;
    bool nulls_are_smallest  = false;

    column_wrapper <T> col1 = column_wrapper<T>({1,1,1,1},[](auto row) { return true; }); 
    column_wrapper <T> col2 = column_wrapper<T>({1,2,3,1},[](auto row) { return ((row == 3)? false : true); });

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());

    cudf::table table(cols.data(), cols.size());
    EXPECT_EQ(cols.size(), static_cast<unsigned int> (table.num_columns()));
    std::vector <int8_t> const asc_dec {0,0};

    found = is_sorted (table, asc_dec, nulls_are_smallest);

    EXPECT_EQ(found, true);
}


TYPED_TEST(IsSorted, EmptyInput)
{
    using T = TypeParam;
    bool found  = false;
    bool nulls_are_smallest  = false;

    cudf::table table{};
    std::vector <int8_t> const asc_dec{};

    found = is_sorted (table, asc_dec, nulls_are_smallest);

    EXPECT_EQ(found, true);
}


TYPED_TEST(IsSorted, ZeroRowswithColumns)
{
    using T = TypeParam;
    bool found  = false;
    bool nulls_are_smallest  = false;

    column_wrapper <T> col1 = column_wrapper<T>({},[](auto row) { return true; });
    column_wrapper <T> col2 = column_wrapper<T>({},[](auto row) { return true; });

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());

    cudf::table table{cols.data(), (int)cols.size()};
    std::cout<<"Number of rows "<<table.num_rows()<<std::endl;
    std::cout<<"Number of cols "<<table.num_columns()<<std::endl;
    std::vector <int8_t> const asc_dec{};

    found = is_sorted (table, asc_dec, nulls_are_smallest);

    EXPECT_EQ(found, true);
}

TYPED_TEST(IsSorted, EmptyColumnOrderingInfoTrue)
{
    using T = TypeParam;
    bool found  = false;

    column_wrapper <T> col1 = column_wrapper<T>({1,1,1,1},[](auto row) { return true; });
    column_wrapper <T> col2 = column_wrapper<T>({1,2,3,4},[](auto row) { return true; });

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());

    cudf::table table(cols.data(), cols.size());
    EXPECT_EQ(cols.size(), static_cast<unsigned int> (table.num_columns()));
    std::vector <int8_t> const asc_dec{};

    found = is_sorted (table, asc_dec, false);

    EXPECT_EQ(found, true);

}

TYPED_TEST(IsSorted, EmptyColumnOrderingInfoFalse)
{
    using T = TypeParam;
    bool found  = false;

    column_wrapper <T> col1 = column_wrapper<T>({1,1,1,1},[](auto row) { return true; });
    column_wrapper <T> col2 = column_wrapper<T>({4,3,2,1},[](auto row) { return true; });

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());

    cudf::table table(cols.data(), cols.size());
    EXPECT_EQ(cols.size(), static_cast<unsigned int> (table.num_columns()));
    std::vector <int8_t> const asc_dec{};

    found = is_sorted (table, asc_dec, false);

    EXPECT_EQ(found, false);
}
