/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 William Scott Malpica <william@blazingdb.com>
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

#include <vector>
#include <iostream>

#include "gtest/gtest.h"

#include <cudf.h>
#include <rmm/rmm.h>

#include <cudf/types.h>
#include <cudf/functions.h>
#include "utilities/cudf_utils.h"
#include "utilities/bit_util.cuh"
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>

cudaStream_t stream;

/// Helper class for similar tests
struct GroupByWoAggregationsTest : public GdfTest {

    size_t num_rows = 18;
    std::vector<int16_t> data_col0 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    std::vector<int32_t> data_col1 = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
    std::vector<int64_t> data_col2 = {0, 0, 0, -10000, -10000, -10000, 999999, 999999, 999999, -4412355, -4412355, -4412355, 12345678, 12345678, 12345678, 5, 5, 5};
    std::vector<float> data_col3 = {1000.5, 1000.5, 1000.5, -2.33, -2.33, 1000.5, -3.123, 1000.5, 0, 1, -2.33, 2.33, -2.33, -3.123, 123, -3.123, 123, 0};

    size_t num_masks = gdf_get_num_chars_bitmask(num_rows);
    std::vector<gdf_valid_type> valid_int_col0;
    std::vector<gdf_valid_type> valid_int_col1;
    std::vector<gdf_valid_type> valid_int_col2;
    std::vector<gdf_valid_type> valid_int_col3;


    void initialize_data(){

std::cout<<"Here A"<<std::endl;

        valid_int_col0.resize(num_masks);
        valid_int_col1.resize(num_masks);
        valid_int_col2.resize(num_masks);
        valid_int_col3.resize(num_masks);

        std::cout<<"Here B"<<std::endl;
        for(size_t row = 0; row < num_rows; ++row){
            gdf::util::turn_bit_on(valid_int_col0.data(), row);  // all on

            if (row % 2 == 0)
                gdf::util::turn_bit_on(valid_int_col1.data(), row);  // every other on
            else
                gdf::util::turn_bit_off(valid_int_col1.data(), row);  // every other on

            if (row < num_rows/3)
                gdf::util::turn_bit_on(valid_int_col2.data(), row);  // first third on
            else
                gdf::util::turn_bit_off(valid_int_col2.data(), row);  // first third on

            if (row < num_rows/2)
                gdf::util::turn_bit_off(valid_int_col3.data(), row);  // first half off
            else
                gdf::util::turn_bit_off(valid_int_col3.data(), row);  // first half off
        }
        std::cout<<"Here C"<<std::endl;
    }
};



TEST_F(GroupByWoAggregationsTest, SimpleTest){

    initialize_data();

    std::cout<<"Here D"<<std::endl;

    gdf_col_pointer gdf_col0 = create_gdf_column(data_col1);

    std::vector<int32_t> temp_for_output_data0(num_rows);
    gdf_col_pointer gdf_out_data0 = create_gdf_column(temp_for_output_data0);
    std::vector<int32_t> temp_for_output_indices(num_rows);
    gdf_col_pointer gdf_out_indices = create_gdf_column(temp_for_output_indices);

    std::vector<int32_t> expected_output = {0, 3, 6, 9, 12, 15};
    gdf_col_pointer expected_gdf_out_indices = create_gdf_column(expected_output);
    
    std::cout<<"Here E"<<std::endl;

    int num_data_cols = 1;
    std::vector<gdf_column*> data_cols_in = {gdf_col0.get()};
    int num_groupby_cols = 1;
	std::vector<int> groupby_col_indices = {0};
    std::vector<gdf_column*> data_cols_out = {gdf_out_data0.get()};
    int nulls_are_smallest = 0;

    std::cout<<"Here F"<<std::endl;

    gdf_error err = gdf_group_by_wo_aggregations(num_data_cols,
                           	   	   	   &data_cols_in[0],
									   num_groupby_cols,
									   &groupby_col_indices[0],
									   &data_cols_out[0],
									   gdf_out_indices.get(),
									   nulls_are_smallest);

    
    std::cout<<"expected"<<std::endl;
    print_gdf_column(expected_gdf_out_indices.get());
    std::cout<<"got"<<std::endl;
    print_gdf_column(gdf_out_indices.get());

    std::cout<<"Here G"<<std::endl;
    ASSERT_TRUE(gdf_equal_columns<int32_t>(gdf_out_indices.get(), expected_gdf_out_indices.get()));

}