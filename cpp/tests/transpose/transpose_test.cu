/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

// #include "test_utils.h"
#include "gtest/gtest.h"
#include <tests/utilities/cudf_test_fixtures.h>
#include "tests/utilities/cudf_test_utils.cuh"

#include <cudf.h>
#include <cudf/functions.h>
#include <utilities/error_utils.hpp>
#include <utilities/cudf_utils.h>

#include <vector>
#include <random>
#include <algorithm>
#include <memory>

/** macro to throw a c++ std::runtime_error */
#define THROW(fmt, ...)                                                 \
    do {                                                                \
        std::string msg;                                                \
        char errMsg[2048];                                              \
        std::sprintf(errMsg, "Exception occured! file=%s line=%d: ",    \
                     __FILE__, __LINE__);                               \
        msg += errMsg;                                                  \
        std::sprintf(errMsg, fmt, ##__VA_ARGS__);                       \
        msg += errMsg;                                                  \
        throw std::runtime_error(msg);                                  \
    } while(0)

/** macro to check for a conditional and assert on failure */
#define ASSERT(check, fmt, ...)                  \
    do {                                         \
        if(!(check))  THROW(fmt, ##__VA_ARGS__); \
    } while(0)

/** check for cuda runtime API errors and assert accordingly */
#define CUDA_CHECK(call)                                \
    do {                                                \
        cudaError_t status = call;                      \
        ASSERT(status == cudaSuccess,                   \
               "FAIL: call='%s'. Reason:%s\n",          \
               #call, cudaGetErrorString(status));      \
    } while(0)

/** performs a device to host copy */
template <typename Type>
void updateHost(Type* hPtr, const Type* dPtr, size_t len) {
    CUDA_CHECK(cudaMemcpy(hPtr, dPtr, len*sizeof(Type),
                          cudaMemcpyDeviceToHost));
}

/*
 * @brief Helper function to compare 2 device n-D arrays with custom comparison
 * @tparam T the data type of the arrays
 * @tparam L the comparator lambda or object function
 * @param expected expected value(s)
 * @param actual actual values
 * @param eq_compare the comparator
 * @return the testing assertion to be later used by ASSERT_TRUE/EXPECT_TRUE
 * @{
 */
template<typename T, typename L>
::testing::AssertionResult devArrMatch(const T* expected, const T* actual,
                                       size_t size, L eq_compare) {
    std::shared_ptr<T> exp_h(new T [size]);
    std::shared_ptr<T> act_h(new T [size]);
    updateHost<T>(exp_h.get(), expected, size);
    updateHost<T>(act_h.get(), actual, size);
    for(size_t i(0);i<size;++i) {
        auto exp = exp_h.get()[i];
        auto act = act_h.get()[i];
        if(!eq_compare(exp, act)) {
            return ::testing::AssertionFailure() << "actual=" << act
                                                 << " != expected=" << exp
                                                 << " @" << i;
        }
    }
    return ::testing::AssertionSuccess();
}

template <typename T>
struct CompareApproxAbs {
    CompareApproxAbs(T eps_): eps(eps_) {}
    bool operator()(const T& a, const T& b) const {
        T diff = abs(abs(a) - abs(b));
        T m = std::max(abs(a), abs(b));
        T ratio = m >= eps? diff / m : diff;
        return (ratio <= eps);
    }
private:
    T eps;
};


template <typename T>
class TransposeTest : public GdfTest {

protected:
    void make_input()
    {
        // generate random numbers for the input vector of vectors
        std::mt19937 rng(1);
        auto generator = [&](){ 
            return rng() % std::numeric_limits<T>::max() + 1;
        };

        in_columns.resize(_ncols);

        for(auto & c : in_columns){
            c.resize(_nrows);
            std::generate(c.begin(), c.end(), generator);
        }

        if (_add_nulls) {
            auto valid_generator = [&](size_t row, size_t col){
                return static_cast<bool>( (row ^ col) % 3 );
            };

            in_gdf_columns = initialize_gdf_columns(in_columns, valid_generator);
        } 
        else {
            in_gdf_columns = initialize_gdf_columns(in_columns);
        }
    }

    void create_gdf_output_buffers()
    {
        out_columns.resize(_nrows);

        for(auto & c : out_columns){
            c.resize(_ncols);
        }

        out_gdf_columns = initialize_gdf_columns(out_columns);
    }
    
    void compute_gdf_result(void)
    {
      std::vector<gdf_column*> in_gdf_column_ptr(in_gdf_columns.size());
      std::vector<gdf_column*> out_gdf_column_ptr(out_gdf_columns.size());

      auto to_raw_ptr = [&](gdf_col_pointer const& col) { return col.get(); };

      std::transform(in_gdf_columns.begin(), in_gdf_columns.end(),
                     in_gdf_column_ptr.begin(), to_raw_ptr);

      std::transform(out_gdf_columns.begin(), out_gdf_columns.end(),
                     out_gdf_column_ptr.begin(), to_raw_ptr);

      gdf_transpose(in_gdf_columns.size(), in_gdf_column_ptr.data(),
                    out_gdf_column_ptr.data());

    }

    void create_reference_output(void)
    {
        // create vec of vec (dim transpose of in_columns)

        ref_data.resize(_nrows);

        for(auto & c : ref_data){
            c.resize(_ncols);
        }

        // copy transposed from in_columns to ref_data
        for(size_t i = 0; i < _nrows; i++)
            for(size_t j = 0; j < _ncols; j++)
                (ref_data[i])[j] = (in_columns[j])[i];
        
        if (_add_nulls) {
            auto valid_generator = [&](size_t row, size_t col){
                return static_cast<bool>( (row ^ col) % 3 );
            };

            ref_gdf_columns = initialize_gdf_columns(ref_data, valid_generator);
        }
        else {
            ref_gdf_columns = initialize_gdf_columns(ref_data);
        }
    }

    void compare_gdf_result(void)
    {
        // new num cols = old _nrows
        for(size_t i = 0; i < _nrows; i++)
        {
            ASSERT_TRUE(
                gdf_equal_columns<T>(ref_gdf_columns[i].get(), out_gdf_columns[i].get())
            );
        }
    }

    void set_params(size_t ncols, size_t nrows, bool add_nulls = false)
    {
        _nrows = nrows; _ncols = ncols; _add_nulls = add_nulls;
    }

    void run_test(void)
    {
        make_input();
        create_gdf_output_buffers();
        compute_gdf_result();
        create_reference_output();
        compare_gdf_result();
    }

    // vector of vector to serve as input to transpose
    std::vector< std::vector<T> > in_columns;
    std::vector< gdf_col_pointer > in_gdf_columns;
    std::vector< std::vector<T> > ref_data;
    std::vector< gdf_col_pointer > ref_gdf_columns;
    std::vector< std::vector<T> > out_columns;
    std::vector< gdf_col_pointer > out_gdf_columns;

    bool _add_nulls = false;
    size_t _ncols = 0;
    size_t _nrows = 0;
};

using TestTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t>;

TYPED_TEST_CASE(TransposeTest, TestTypes);

TYPED_TEST(TransposeTest, SingleValue)
{
    size_t num_cols = 1;
    size_t num_rows = 1;
    this->set_params(num_cols, num_rows);
    this->run_test();
}

TYPED_TEST(TransposeTest, SingleColumn)
{
    size_t num_cols = 1;
    size_t num_rows = 1000;
    this->set_params(num_cols, num_rows);
    this->run_test();
}

TYPED_TEST(TransposeTest, SingleColumnNulls)
{
    size_t num_cols = 1;
    size_t num_rows = 1000;
    this->set_params(num_cols, num_rows, true);
    this->run_test();
}

TYPED_TEST(TransposeTest, Square)
{
    size_t num_cols = 100;
    size_t num_rows = 100;
    this->set_params(num_cols, num_rows);
    this->run_test();
}

TYPED_TEST(TransposeTest, SquareNulls)
{
    size_t num_cols = 100;
    size_t num_rows = 100;
    this->set_params(num_cols, num_rows, true);
    this->run_test();
}

TYPED_TEST(TransposeTest, Slim)
{
    size_t num_cols = 10;
    size_t num_rows = 1000;
    this->set_params(num_cols, num_rows);
    this->run_test();
}

TYPED_TEST(TransposeTest, SlimNulls)
{
    size_t num_cols = 10;
    size_t num_rows = 1000;
    this->set_params(num_cols, num_rows, true);
    this->run_test();
}

TYPED_TEST(TransposeTest, Fat)
{
    size_t num_cols = 1000;
    size_t num_rows = 10;
    this->set_params(num_cols, num_rows);
    this->run_test();
}

TYPED_TEST(TransposeTest, FatNulls)
{
    size_t num_cols = 1000;
    size_t num_rows = 10;
    this->set_params(num_cols, num_rows, true);
    this->run_test();
}
