/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
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

#include <gdf/gdf.h>

#include "utils.h"

template <class T>
static inline thrust::device_vector<T>
MakeDeviceVector(const std::initializer_list<T> list) {
    const std::vector<T>     column_data(list);
    thrust::device_vector<T> device_data(column_data);
    return device_data;
}

// This is the main test feature
template <class T>
class ReplaceTest : public testing::Test {
protected:
    thrust::device_ptr<T>
    test(const std::initializer_list<T> &data_list,
         const std::initializer_list<T> &to_replace_list,
         const std::initializer_list<T> &values_list) {
        device_data     = MakeDeviceVector<T>(data_list);
        to_replace_data = MakeDeviceVector<T>(to_replace_list);
        values_data     = MakeDeviceVector<T>(values_list);

        column     = MakeGdfColumn(device_data);
        to_replace = MakeGdfColumn(to_replace_data);
        values     = MakeGdfColumn(values_data);

        const gdf_error status =
          gdf_find_and_replace_all(&column, &to_replace, &values);

        EXPECT_EQ(GDF_SUCCESS, status);

        return thrust::device_ptr<T>(static_cast<T *>(column.data));
    }

    thrust::device_vector<T> device_data;
    thrust::device_vector<T> to_replace_data;
    thrust::device_vector<T> values_data;

    gdf_column column;
    gdf_column to_replace;
    gdf_column values;
};

using Types = testing::
  Types<std::int8_t, std::int16_t, std::int32_t, std::int64_t, float, double>;

TYPED_TEST_CASE(ReplaceTest, Types);

// Simple test, replacing all even values
TYPED_TEST(ReplaceTest, ReplaceEvenPosition) {
    thrust::device_ptr<TypeParam> results =
      this->test({1, 2, 3, 4, 5, 6, 7, 8}, {2, 4, 6, 8}, {0, 2, 4, 6});

    EXPECT_EQ(0, results[1]);
    EXPECT_EQ(2, results[3]);
    EXPECT_EQ(4, results[5]);
    EXPECT_EQ(6, results[7]);
}

// Similar test as ReplaceEvenPosition, but with unordered data
TYPED_TEST(ReplaceTest, Unordered) {
    thrust::device_ptr<TypeParam> results =
      this->test({7, 5, 6, 3, 1, 2, 8, 4}, {2, 6, 4, 8}, {0, 4, 2, 6});

    EXPECT_EQ(4, results[2]);
    EXPECT_EQ(0, results[5]);
    EXPECT_EQ(6, results[6]);
    EXPECT_EQ(2, results[7]);
}

// Testing with Empty Replace
TYPED_TEST(ReplaceTest, EmptyReplace) {
    thrust::device_ptr<TypeParam> results =
      this->test({7, 5, 6, 3, 1, 2, 8, 4}, {}, {});

    EXPECT_EQ(7, results[0]);
    EXPECT_EQ(5, results[1]);
    EXPECT_EQ(6, results[2]);
    EXPECT_EQ(3, results[3]);
    EXPECT_EQ(1, results[4]);
    EXPECT_EQ(2, results[5]);
    EXPECT_EQ(8, results[6]);
    EXPECT_EQ(4, results[7]);
}

// Testing with Nothing To Replace
TYPED_TEST(ReplaceTest, NothingToReplace) {
    thrust::device_ptr<TypeParam> results =
      this->test({7, 5, 6, 3, 1, 2, 8, 4}, {10, 11, 12}, {15, 16, 17});

    EXPECT_EQ(7, results[0]);
    EXPECT_EQ(5, results[1]);
    EXPECT_EQ(6, results[2]);
    EXPECT_EQ(3, results[3]);
    EXPECT_EQ(1, results[4]);
    EXPECT_EQ(2, results[5]);
    EXPECT_EQ(8, results[6]);
    EXPECT_EQ(4, results[7]);
}

// Testing with Empty Data
TYPED_TEST(ReplaceTest, EmptyData) {
    this->test({}, {10, 11, 12}, {15, 16, 17});
}

// Test with much larger data sets
TEST(LargeScaleReplaceTest, LargeScaleReplaceTest) {
    const int DATA_SIZE    = 1000000;
    const int REPLACE_SIZE = 10000;

    srand((unsigned) time(NULL));

    std::vector<std::int32_t> column_data(DATA_SIZE);
    for (int i = 0; i < DATA_SIZE; i++) {
        column_data[i] = rand() % (2 * REPLACE_SIZE);
    }

    std::vector<std::int32_t> from(DATA_SIZE);
    std::vector<std::int32_t> to(DATA_SIZE);
    int                       count = 0;
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < REPLACE_SIZE; j += 7) {
            from[i + j] = count;
            count++;
            to[i + j] = count;
        }
    }

    thrust::device_vector<std::int32_t> device_data(column_data);
    gdf_column                          data_gdf = MakeGdfColumn(device_data);
    thrust::device_vector<std::int32_t> device_from(from);
    gdf_column                          from_gdf = MakeGdfColumn(device_from);
    thrust::device_vector<std::int32_t> device_to(to);
    gdf_column                          to_gdf = MakeGdfColumn(device_to);

    const gdf_error status =
      gdf_find_and_replace_all(&data_gdf, &from_gdf, &to_gdf);

    EXPECT_EQ(GDF_SUCCESS, status);

    std::vector<std::int32_t> replaced_data(DATA_SIZE);
    thrust::copy(device_data.begin(), device_data.end(), replaced_data.begin());

    count = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        if (column_data[i] < REPLACE_SIZE) {
            EXPECT_EQ(column_data[i] + 1, replaced_data[i]);
            if (column_data[i] + 1 != replaced_data[i]) {
                std::cout << "failed at " << i
                          << "  column_data[i]: " << column_data[i]
                          << "  replaced_data[i]: " << replaced_data[i]
                          << std::endl;
                count++;
                if (count > 20) { break; }
            }
        }
    }
}
