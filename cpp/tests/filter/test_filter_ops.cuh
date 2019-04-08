/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Eyal Rozenberg <eyalroz@blazingdb.com>
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

#ifndef CUDF_TEST_FILTER_OPS_CUH_
#define CUDF_TEST_FILTER_OPS_CUH_

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/compare_column_wrappers.cuh>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/column_wrapper.cuh>
#include <utilities/fill.cuh>
#include <utilities/bit_util.cuh>
#include <utilities/type_dispatcher.hpp>
#include <utilities/column_utils.hpp>

#include <cudf.h>

#include <gtest/gtest.h>

#include <iostream>
#include <iomanip>
#include <tuple>

#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/valid_vectors.h>

#include <utilities/type_dispatcher.hpp>

#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <random>

using cudf::test::column_wrapper;


enum : bool {
    non_nullable = false,
    nullable = true
};

template <typename E>
void print(
    const column_wrapper<E>& wrapper,
    const std::string& title,
    unsigned min_element_printing_width = 1)
{
    std::cout << title << std::endl;
    wrapper.print(min_element_printing_width);
    return;
}

// Use this to print a column wrapper variable with its identifier as its name
#define self_titled_print(wrapper, min_element_width) print(wrapper, CUDF_STRINGIFY(wrapper), min_element_width)



// Note: A few fixed column_sizes are not sufficient to test gdf_apply_boolean_mask.
// Multiple column_sizes are necessary to cover the large number of cases
struct column_sizes {

    static constexpr const gdf_size_type  short_non_round = 25;
    static constexpr const gdf_size_type  short_round = 32;
    static constexpr const gdf_size_type  long_round = 128 * 1024;
    static constexpr const gdf_size_type  long_non_round = 100007;

//    enum : gdf_size_type {
//        short_non_round = 25,
//        short_round = 32,
//        long_round = 100000, // Not currently used
//        long_non_round = 100007,
//    };
};

enum : gdf_bool { gdf_false = 0, gdf_true = 1 };


template <typename T>
struct uniformly_distributed {
    std::random_device device {};
//    std::mt19937 generator { device() };
//    std::default_random_engine engine { device() };
    std::mt19937 generator { static_cast<long unsigned int>(std::time(0))};
    typename std::conditional_t<
        std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        std::uniform_real_distribution<T> >  distribution;

    uniformly_distributed() = default;
    uniformly_distributed(const uniformly_distributed& other) : device{}, generator{other.generator}, distribution{other.distribution} { }
    uniformly_distributed(T min, T max) : distribution{ min, max } { }

    T operator()(gdf_size_type) { return distribution(generator); }
};

template <typename T, typename U, U Value>
T constant(gdf_size_type) noexcept { return T{Value}; }

// Only works for integral types...
template <typename T, T Value>
T constant(gdf_size_type) noexcept { return Value; }

template <typename T>
T zero(gdf_size_type) noexcept { return T{0}; }


const auto fully_valid = constant<bool, true>;


struct first_half {
    gdf_size_type range_length;
    bool operator()(gdf_size_type i) const noexcept {
        return (i < range_length / 2);
    }
};

template <gdf_size_type CycleLength>
struct first_of_every {
    bool operator()(gdf_size_type i) const noexcept {
        return (i % CycleLength == 0);
    }
};

namespace {


// This functionaly should really have been provided by column_wrapper - but it isn't really.
// It holds its data on the device side.
template <typename element_type>
struct poor_mans_host_side_column {
    gdf_size_type size;
    gdf_dtype_extra_info extra_dtype_info;
    gdf_size_type null_count;
    std::unique_ptr<element_type> data;
    std::unique_ptr<gdf_valid_type> valid; // has size 0 if the column is non-nullable
    // gdf_dtype_extra_info extra_type_info; // <- We should really have this...

//    poor_mans_host_side_column() { size = 0; null_count == 0; data = nullptr; valid == nullptr; }
//    poor_mans_host_side_column(const poor_mans_host_side_column& other) = default;
//    ~poor_mans_host_side_column() { if (data != nullptr) delete data; if (valid != nullptr) delete valid; }
};

template <typename Element>
void print(const poor_mans_host_side_column<Element>& col, const std::string& title, unsigned min_element_printing_width = 1)
{
    std::cout << title << std::endl;
    print_host_side_column_copy<Element>(
        col.size, col.extra_dtype_info, col.data.get(),
        col.size == 0 ? nullptr : col.valid.get(),
        min_element_printing_width);
    return;
}


// This ugly code makes me want to cry! T_T
template <typename element_type>
poor_mans_host_side_column<element_type> compute_expected_output(
    column_wrapper<element_type>& data,
    column_wrapper<gdf_bool    >& stencil)
{
    gdf_size_type input_size =  data.get()->size;
    assert(input_size == stencil.get()->size);
    poor_mans_host_side_column<element_type> result { 0, data.get()->dtype_info, 0, nullptr, nullptr };
    if (input_size == 0) {
        return result;
    }
    auto data_on_host = data.to_host();
    const element_type* data_on_host_data  = std::get<0>(data_on_host).data();
    assert(data != nullptr);
    const gdf_valid_type * data_on_host_validity =
        cudf::is_nullable(*data.get()) ? std::get<1>(data_on_host).data() : nullptr;

    auto stencil_on_host = stencil.to_host();
    const gdf_bool* stencil_on_host_data  = std::get<0>(stencil_on_host).data();
    // Stencils are assumed to have validity indicators
    const gdf_valid_type * stencil_on_host_validity = std::get<1>(stencil_on_host).data();

    auto input_element_is_valid = [&stencil_on_host_data, &stencil_on_host_validity](gdf_size_type i) {
        return stencil_on_host_data[i] and
            (stencil_on_host_validity == nullptr or
                gdf::util::bit_is_set(stencil_on_host_validity, i));
    };

    std::vector<element_type> result_data;
    std::vector<gdf_valid_type> result_validity;
    result_data.reserve(input_size);

    if (data_on_host_validity == nullptr) {
        result.null_count = 0;
        for (gdf_size_type i = 0 ; i < input_size ; i++) {
            if (input_element_is_valid(i)) {
                result_data.push_back(data_on_host_data[i]);
                result.size++;
            }
        }
    }
    else {
        result_validity.resize(gdf_valid_allocation_size(input_size));
        std::fill(result_validity.begin(), result_validity.end(), 0);

        for (gdf_size_type i = 0 ; i < input_size ; i++) {
            if (input_element_is_valid(i)) {
                result_data.push_back(data_on_host_data[i]);
                if (gdf::util::bit_is_set(data_on_host_validity, i))
                {
                    gdf::util::turn_bit_on(result_validity.data(), result.size);
                }
                else { result.null_count++; }
                result.size++;
            }
        }
        if (result.size > 0) {
            result.valid.reset(new gdf_valid_type[gdf_valid_allocation_size(result.size)]);
            std::copy_n(result_validity.cbegin(), gdf_valid_allocation_size(result.size), result.valid.get());
        }
        else {
            result.valid.reset(nullptr);
        }
    }
    if (result.size > 0) {
        result.data.release();
        result.data.reset(new element_type[result.size]);
        std::copy_n(result_data.cbegin(), result.size, result.data.get());
    }
    else {
        result.data.reset(nullptr);
    }
    return result;
}

template<typename E>
void expect_columns_are_equal(
    const cudf::test::column_wrapper<E>&  lhs,
    const std::string&                    lhs_name,
    const poor_mans_host_side_column<E>&  rhs,
    const std::string&                    rhs_name,
    bool                                  print_all_unequal_pairs = false)
{
    auto lhs_gdf_column = *(lhs.get());
    EXPECT_EQ(cudf::validate(lhs_gdf_column), GDF_SUCCESS) << "The " << lhs_name << " column is invalid, cannot compare it to " << rhs_name;
    EXPECT_EQ(lhs_gdf_column.dtype, cudf::gdf_dtype_of<E>());
    auto common_dtype = lhs_gdf_column.dtype;
    EXPECT_TRUE(cudf::detail::extra_type_info_is_compatible(common_dtype, lhs_gdf_column.dtype_info, rhs.extra_dtype_info));
        // TODO: Support equality with different dtype_info values, e.g. different scales
    EXPECT_EQ(lhs_gdf_column.size, rhs.size);
    auto common_size = lhs_gdf_column.size;
    EXPECT_EQ(lhs_gdf_column.null_count, rhs.null_count);
    if (lhs_gdf_column.size != rhs.size or
        (not cudf::detail::extra_type_info_is_compatible(
            common_dtype,
            lhs_gdf_column.dtype_info,
            rhs.extra_dtype_info)
        )
       ) { return; }

    if (common_size == 0) { return; }

    auto lhs_on_host = lhs.to_host();

    const E* lhs_data_on_host  = std::get<0>(lhs_on_host).data();

    const gdf_valid_type* lhs_validity_on_host = std::get<1>(lhs_on_host).data();

    return expect_column_values_are_equal<E>(
        common_size,
        lhs_data_on_host, lhs_validity_on_host, lhs_name,
        rhs.data.get(), rhs.valid.get(), rhs_name,
        print_all_unequal_pairs);
}

template<typename E>
void expect_columns_are_equal(
    const column_wrapper<E>& actual,
    const poor_mans_host_side_column<E>&  expected)
{
    return expect_columns_are_equal(actual, "Actual", expected, "Expected", false);
}

template<typename DataGenerator>
auto make_non_nullable_column_wrapper(
    gdf_size_type size,
    DataGenerator data_generator)
{
    using element_type = decltype(data_generator(0));
    std::vector<element_type> data_on_host;
    data_on_host.reserve(size);
    gdf_size_type i { 0 };
    auto no_params_generator = [&]() { return data_generator(i++); };
    std::generate_n(std::back_inserter(data_on_host), size, no_params_generator);
    return column_wrapper<element_type>{data_on_host};
}



} // namespace

#endif // CUDF_TEST_FILTER_OPS_CUH_
