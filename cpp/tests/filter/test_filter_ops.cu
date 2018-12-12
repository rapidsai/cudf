/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
 *     Copyright 2018 Felipe Aramburu <felipe@blazingdb.com>
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

#include "gtest/gtest.h"
#include <iostream>
#include <cudf.h>
#include <cudf/functions.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <tuple>
#include "helper/utils.cuh"
#include "tests/utilities/cudf_test_fixtures.h"

/*
 ============================================================================
 Description : Compute gpu_comparison and apply_stencil of gdf_columns using Thrust on GPU
 ============================================================================
 */

enum class valids_t {
    HALF_BITS, // init randomly the last half bits
    ALL_BITS_ON, // all bits set to 1
    NULL_BITS // valid is a nullptr
};

template <typename column_t>
struct GpuApplyStencilTest : public GdfTest {

    // contains the input column
    std::vector<column_t> host_vector;

    // contains valids for column
    host_valid_pointer host_valid = nullptr;

    // contains the stencil column
    std::vector<int8_t> host_stencil_vector;

    // contains valids for stencil column
    host_valid_pointer host_stencil_valid = nullptr;

    gdf_col_pointer col;
    gdf_col_pointer stencil;
    gdf_col_pointer output;

    // contains the filtered column
    std::vector<column_t> reference_vector;

    // contains the valids for the filtered column
    host_valid_pointer reference_valid = nullptr;

    /* --------------------------------------------------------------------------*/
    /**
    * @Synopsis  Initializes input columns
    *
    * @Param length The length of the column
    * @Param max_val The maximum value of aggregation column
    * @Param valids_col The type of initialization for valids from col according the valids_t enum
    * @Param valids_stencil The type of initialization for valids from stencil according the valids_t enum
    * @Param print Optionally print column for debugging
    */
    /* ----------------------------------------------------------------------------*/
    void create_input(const size_t length, const size_t max_val,
        valids_t valids_col, valids_t valids_stencil,
        bool print = false) {

        size_t shuffle_seed = rand();

        initialize_values(host_vector, length, max_val, shuffle_seed);
        if ( valids_col != valids_t::NULL_BITS) {
            initialize_valids(host_valid, length, valids_col == valids_t::ALL_BITS_ON);
        }

        col = create_gdf_column(host_vector, host_valid);

        initialize_values(host_stencil_vector, length, 1, shuffle_seed);
        if ( valids_stencil != valids_t::NULL_BITS) {
            initialize_valids(host_stencil_valid, length, valids_stencil == valids_t::ALL_BITS_ON);
        }

        stencil = create_gdf_column(host_stencil_vector, host_stencil_valid);

        std::vector<column_t> zero_vector (length, 0);
        host_valid_pointer output_valid;

        initialize_valids(output_valid, length, true);

        output = create_gdf_column(zero_vector, output_valid);

        allocate_reference_valids(reference_valid, length);

        if(print) {
            std::cout<<"Input:\n";
            print_gdf_column(col.get());

            std::cout<<"Stencil:\n";
            print_gdf_column(stencil.get());

            std::cout<<"\n";
        }
    }

    void print_reference_column() {
        std::cout<<"Reference Output:\n";
        for(size_t i = 0; i < reference_vector.size(); i++){
            std::cout << "(" << std::to_string(reference_vector[i]) << "|" << gdf_is_valid(reference_valid.get(), i) << "), ";
        }
        std::cout<<"\n\n";
    }

    void allocate_reference_valids(host_valid_pointer& valid_ptr, size_t length) {
        auto deleter = [](gdf_valid_type* valid) { delete[] valid; };
        auto n_bytes = get_number_of_bytes_for_valid(length);
        auto valid_bits = new gdf_valid_type[n_bytes];
    
        valid_ptr = host_valid_pointer{ valid_bits, deleter };
    }

    void compute_reference_solution() {

        size_t valid_index=0;
        for (size_t index = 0 ; index < host_vector.size() ; index++) {
            if (host_stencil_vector[index] == 1 && gdf_is_valid(host_stencil_valid.get(), index) ){
                reference_vector.push_back(host_vector[index]);

                if ( gdf_is_valid(host_valid.get(), index) ) {
                    gdf::util::turn_bit_on(reference_valid.get(), valid_index);
                } else {
                    gdf::util::turn_bit_off(reference_valid.get(), valid_index);
                }

                valid_index++;
            }
        }
    }

    gdf_error compute_gdf_result() {
        gdf_error error = gpu_apply_stencil(col.get(), stencil.get(), output.get());
        return error;
    }

    void print_debug() {
        std::cout<<"Output:\n";
        print_gdf_column(output.get());
        std::cout<<"\n";

        print_reference_column();
    }

    void compare_gdf_result() {
        std::vector<column_t> host_result(reference_vector.size());
        
        // Copy result of applying stencil to the host
        EXPECT_EQ(cudaMemcpy(host_result.data(), output.get()->data, output.get()->size * sizeof(column_t), cudaMemcpyDeviceToHost), cudaSuccess);

        // Compare the gpu and reference solutions
        for(size_t i = 0; i < reference_vector.size(); ++i) {
            EXPECT_EQ(reference_vector[i], host_result[i]);
        }

        auto n_bytes = get_number_of_bytes_for_valid(col->size);
        gdf_valid_type* host_ptr = new gdf_valid_type[n_bytes];

        // Copy valids to the host
        EXPECT_EQ(cudaMemcpy(host_ptr, output.get()->valid, n_bytes, cudaMemcpyDeviceToHost), cudaSuccess);

        // Compare the gpu and reference valid arrays
        for(size_t i = 0; i < reference_vector.size(); ++i) {
            EXPECT_EQ( gdf_is_valid(reference_valid.get(), i), gdf_is_valid(host_ptr, i) );
        }
    }
};

typedef ::testing::Types<
    int32_t,
    int64_t,
    float,
    double
  > Implementations;

TYPED_TEST_CASE(GpuApplyStencilTest, Implementations);

//Todo: usage_example
//TYPED_TEST(GpuApplyStencilTest, usage_example) {

TYPED_TEST(GpuApplyStencilTest, all_bits_on_multiple_32) {
    const bool print_result{false};

    this->create_input(32, 100, valids_t::ALL_BITS_ON, valids_t::ALL_BITS_ON, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ(this->output.get()->size, this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(GpuApplyStencilTest, all_bits_on_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(25, 100, valids_t::ALL_BITS_ON, valids_t::ALL_BITS_ON, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ(this->output.get()->size, this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(GpuApplyStencilTest, half_zero_col_multiple_32) {
    const bool print_result{false};

    this->create_input(32, 100, valids_t::HALF_BITS, valids_t::ALL_BITS_ON, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ(this->output.get()->size, this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(GpuApplyStencilTest, half_zero_stencil_multiple_32) {
    const bool print_result{false};

    this->create_input(32, 100, valids_t::ALL_BITS_ON, valids_t::HALF_BITS, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ(this->output.get()->size, this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(GpuApplyStencilTest, half_zero_col_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(25, 100, valids_t::HALF_BITS, valids_t::ALL_BITS_ON, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ(this->output.get()->size, this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(GpuApplyStencilTest, half_zero_stencil_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(25, 100, valids_t::ALL_BITS_ON, valids_t::HALF_BITS, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ(this->output.get()->size, this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(GpuApplyStencilTest, both_half_zero_multiple_of_32) {
    const bool print_result{false};

    this->create_input(32, 100, valids_t::HALF_BITS, valids_t::HALF_BITS, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ(this->output.get()->size, this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(GpuApplyStencilTest, both_half_zero_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(25, 100, valids_t::HALF_BITS, valids_t::HALF_BITS, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ(this->output.get()->size, this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(GpuApplyStencilTest, null_valids_stencil_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(25, 100, valids_t::HALF_BITS, valids_t::NULL_BITS, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ(this->output.get()->size, this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(GpuApplyStencilTest, null_valids_col_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(25, 100, valids_t::NULL_BITS, valids_t::HALF_BITS, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ(this->output.get()->size, this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

TYPED_TEST(GpuApplyStencilTest, both_null_valids_non_multiple_of_32) {
    const bool print_result{false};

    this->create_input(25, 100, valids_t::NULL_BITS, valids_t::NULL_BITS, print_result);

    gdf_error error = this->compute_gdf_result();
    ASSERT_EQ(error, GDF_SUCCESS) << "GPU Apply stencil returned an error code\n";

    this->compute_reference_solution();

    if(print_result) {
        this->print_debug();
    }
    
    ASSERT_EQ(this->output.get()->size, this->reference_vector.size()) << "Size of gdf result does not match reference result\n";

    this->compare_gdf_result();
}

struct FilterOperationsTest : public GdfTest {};

TEST_F(FilterOperationsTest, usage_example) {

    using LeftValueType = int16_t;
    using RightValueType = int16_t;
    int column_size = 10;
    int init_value = 10;
    int max_size = 4;
    gdf_comparison_operator gdf_operator = GDF_EQUALS;

    gdf_column lhs = gen_gdb_column<LeftValueType>(column_size, init_value); // 4, 2, 0
    
    gdf_column rhs = gen_gdb_column<RightValueType>(column_size, 0.01 + max_size - init_value); // 0, 2, 4

    gdf_column output = gen_gdb_column<int8_t>(column_size, 0);

    gdf_error error = gpu_comparison(&lhs, &rhs, &output, gdf_operator);
    EXPECT_TRUE(error == GDF_SUCCESS);

    std::cout << "Left" << std::endl;
    print_column(&lhs);

    std::cout << "Right" << std::endl;
    print_column(&rhs);

    std::cout << "Output" << std::endl;
    print_column(&output);

    check_column_for_comparison_operation<LeftValueType, RightValueType>(&lhs, &rhs, &output, gdf_operator);

    /// lhs.dtype === rhs.dtype
    gpu_apply_stencil(&lhs, &output, &rhs);

    check_column_for_stencil_operation<LeftValueType, RightValueType>(&lhs, &output, &rhs);

    delete_gdf_column(&lhs);
    delete_gdf_column(&rhs);
    delete_gdf_column(&output);
}


template <typename LeftValueType, typename RightValueType>
void test_filterops_using_templates(gdf_comparison_operator gdf_operator = GDF_EQUALS)
{
    //0, ..., 100,
    //100, 10000, 10000, 100000
    for (int column_size = 0; column_size < 10; column_size += 1)
    {
        const int max_size = 8;
        for (int init_value = 0; init_value <= 1; init_value++)
        {
            gdf_column lhs = gen_gdb_column<LeftValueType>(column_size, init_value); // 4, 2, 0
            // lhs.null_count = 2;

            gdf_column rhs = gen_gdb_column<RightValueType>(column_size, 0.01 + max_size - init_value); // 0, 2, 4
            // rhs.null_count = 1;

            gdf_column output = gen_gdb_column<int8_t>(column_size, 0);

            gdf_error error = gpu_comparison(&lhs, &rhs, &output, gdf_operator);
            EXPECT_TRUE(error == GDF_SUCCESS);

            check_column_for_comparison_operation<LeftValueType, RightValueType>(&lhs, &rhs, &output, gdf_operator);

            if (lhs.dtype == rhs.dtype ) {
                gpu_apply_stencil(&lhs, &output, &rhs);
                check_column_for_stencil_operation<LeftValueType, RightValueType>(&lhs, &output, &rhs);
            }

            delete_gdf_column(&lhs);
            delete_gdf_column(&rhs);
            delete_gdf_column(&output);
        }
    }
}

TEST_F(FilterOperationsTest, WithInt8AndOthers)
{
    test_filterops_using_templates<int8_t, int8_t>();
    test_filterops_using_templates<int8_t, int16_t>();
    
    test_filterops_using_templates<int8_t, int32_t>();
    test_filterops_using_templates<int8_t, int64_t>();
    test_filterops_using_templates<int8_t, float>(); 
    test_filterops_using_templates<int8_t, double>();
}

TEST_F(FilterOperationsTest, WithInt16AndOthers)
{
    test_filterops_using_templates<int16_t, int8_t>();
    test_filterops_using_templates<int16_t, int16_t>();
    test_filterops_using_templates<int16_t, int32_t>();
    test_filterops_using_templates<int16_t, int64_t>();
    test_filterops_using_templates<int16_t, float>();
    test_filterops_using_templates<int16_t, double>();
   
}

TEST_F(FilterOperationsTest, WithInt32AndOthers)
{
    test_filterops_using_templates<int32_t, int8_t>();
    test_filterops_using_templates<int32_t, int16_t>();
    test_filterops_using_templates<int32_t, int32_t>();
    test_filterops_using_templates<int32_t, int64_t>();
    test_filterops_using_templates<int32_t, float>();
    test_filterops_using_templates<int32_t, double>();
   
}

TEST_F(FilterOperationsTest, WithInt64AndOthers)
{
    test_filterops_using_templates<int64_t, int8_t>();
    test_filterops_using_templates<int64_t, int16_t>();
    test_filterops_using_templates<int64_t, int32_t>();
    test_filterops_using_templates<int64_t, int64_t>();
    test_filterops_using_templates<int64_t, float>();
    test_filterops_using_templates<int64_t, double>();
   
}

TEST_F(FilterOperationsTest, WithFloat32AndOthers)
{
    test_filterops_using_templates<float, int8_t>();
    test_filterops_using_templates<float, int16_t>();
    test_filterops_using_templates<float, int32_t>();
    test_filterops_using_templates<float, int64_t>();
    test_filterops_using_templates<float, float>();
    test_filterops_using_templates<float, double>();
   
}

TEST_F(FilterOperationsTest, WithFloat64AndOthers)
{
    test_filterops_using_templates<double, int8_t>();
    test_filterops_using_templates<double, int16_t>();
    test_filterops_using_templates<double, int32_t>();
    test_filterops_using_templates<double, int64_t>();
    test_filterops_using_templates<double, float>();
    test_filterops_using_templates<double, double>();
   
}
