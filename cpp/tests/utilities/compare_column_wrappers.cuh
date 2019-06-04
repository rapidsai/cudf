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

#ifndef CUDF_TEST_UTILITIES_COMPARE_COLUMNS_CUH_
#define CUDF_TEST_UTILITIES_COMPARE_COLUMNS_CUH_

#include <tests/utilities/cudf_test_fixtures.h> // for GdfTest
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/column_wrapper.cuh>
#include <utilities/bit_util.cuh>
#include <utilities/type_dispatcher.hpp>
#include <utilities/wrapper_types.hpp>
#include <utilities/column_utils.hpp>

#include <cudf.h>

#include <gtest/gtest.h>

#include <iostream>
#include <iomanip>
#include <tuple>

namespace detail {

template <typename T>
std::string to_string(T val) { return std::to_string(val); }

template <> std::string to_string<cudf::bool8>(cudf::bool8 val)
{
    return {unwrap(val) ? "true" : "false"};
}

// TODO: Implement for cudf::category, cudf::nvstring_category, cudf::timestamp, cudf::date32, cudf::date64

} // namespace detail

/**
 * @note Currently ignoring the extra type info, i.e. assuming it's the same for both columns, or can be ignored
 */
template<typename E>
void expect_column_values_are_equal(
    gdf_size_type          common_size,
    const E*               lhs_data_on_host,
    const gdf_valid_type*  lhs_validity_on_host,
    const std::string&     lhs_name,
    const E*               rhs_data_on_host,
    const gdf_valid_type*  rhs_validity_on_host,
    const std::string&     rhs_name,
    bool                   print_all_unequal_pairs = false)
{
    auto lhs_non_nullable = (lhs_validity_on_host == nullptr);
    auto rhs_non_nullable = (rhs_validity_on_host == nullptr);
    auto max_name_length = std::max(lhs_name.length(), rhs_name.length());

    for(gdf_size_type i = 0; i < common_size; i++) {
        auto lhs_element_is_valid = lhs_non_nullable or gdf::util::bit_is_set<gdf_valid_type, gdf_size_type>(lhs_validity_on_host, i);
        auto rhs_element_is_valid = rhs_non_nullable or gdf::util::bit_is_set<gdf_valid_type, gdf_size_type>(rhs_validity_on_host, i);
        auto elements_are_equal =
            (not lhs_element_is_valid and not rhs_element_is_valid) or
            (lhs_element_is_valid == rhs_element_is_valid and lhs_data_on_host[i] == rhs_data_on_host[i]);
        EXPECT_TRUE(elements_are_equal)
            << std::left << std::setw(max_name_length) << lhs_name << std::right << '[' << i << "] = " << (lhs_element_is_valid ? detail::to_string(lhs_data_on_host[i]) : "@") << '\n'
            << std::left << std::setw(max_name_length) << rhs_name << std::right << '[' << i << "] = " << (rhs_element_is_valid ? detail::to_string(rhs_data_on_host[i]) : "@") ;
        if (not print_all_unequal_pairs and not elements_are_equal) { break; }
    }
}


template<typename E>
void expect_columns_are_equal(
    cudf::test::column_wrapper<E> const&  lhs,
    const std::string&                    lhs_name,
    cudf::test::column_wrapper<E> const&  rhs,
    const std::string&                    rhs_name,
    bool                                  print_all_unequal_pairs = false)
{
    const gdf_column& lhs_gdf_column = *(lhs.get());
    const gdf_column& rhs_gdf_column = *(rhs.get());
    cudf::validate(lhs_gdf_column);
    cudf::validate(rhs_gdf_column);
    EXPECT_TRUE(cudf::have_same_type(lhs_gdf_column, rhs_gdf_column));
    if (not cudf::have_same_type(lhs_gdf_column, rhs_gdf_column)) { return; }
    EXPECT_EQ(lhs_gdf_column.size, rhs_gdf_column.size);
    EXPECT_EQ(lhs_gdf_column.null_count, rhs_gdf_column.null_count);
    auto common_size = lhs_gdf_column.size;
    if (common_size == 0) { return; }
    auto non_nullable = (lhs_gdf_column.null_count == 0);

    auto lhs_on_host = lhs.to_host();
    auto rhs_on_host = rhs.to_host();

    const E* lhs_data_on_host  = std::get<0>(lhs_on_host).data();
    const E* rhs_data_on_host  = std::get<0>(rhs_on_host).data();

    const gdf_valid_type * lhs_validity_on_host = non_nullable ? nullptr : std::get<1>(lhs_on_host).data();
    const gdf_valid_type * rhs_validity_on_host = non_nullable ? nullptr : std::get<1>(rhs_on_host).data();

    return expect_column_values_are_equal(
        common_size,
        lhs_data_on_host, lhs_validity_on_host, lhs_name,
        rhs_data_on_host, rhs_validity_on_host, rhs_name,
        print_all_unequal_pairs);
}

template<typename E>
void expect_columns_are_equal(
    cudf::test::column_wrapper<E> const&  actual,
    cudf::test::column_wrapper<E> const&  expected,
    bool                                  print_all_unequal_pairs = false)
{
    return expect_columns_are_equal<E>(expected, "Expected", actual, "Actual", print_all_unequal_pairs);
}

#endif // CUDF_TEST_UTILITIES_COMPARE_COLUMNS_CUH_
