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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/convert/convert_integers.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/strings/utilities.h>

#include <vector>
#include <string>
#include <gmock/gmock.h>


struct StringsConvertTest : public cudf::test::BaseFixture {};

TEST_F(StringsConvertTest, ToInteger)
{
    std::vector<const char*> h_strings{ "eee", "1234", nullptr, "", "-9832", "93.24", "765é", "-1.78e+5", "2147483647" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<int32_t> h_expected{ 0, 1234, 0, 0, -9832, 93, 765, -1, 2147483647 };

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::to_integers(strings_view, cudf::data_type{cudf::INT32} );

    cudf::test::fixed_width_column_wrapper<int32_t> expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsConvertTest, FromInteger)
{
    std::vector<int32_t> h_integers{ 100, 987654321, 0, 0, -12761, 0, 5, -4, 2147483647 };
    std::vector<const char*> h_expected{ "100", "987654321", nullptr, "0", "-12761", "0", "5", "-4", "2147483647" };

    cudf::test::fixed_width_column_wrapper<int32_t> integers( h_integers.begin(), h_integers.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));

    auto results = cudf::strings::from_integers(integers);
    //cudf::strings::print(cudf::strings_column_view(*results));

    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));

    cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsConvertTest, ZeroSizeStringsColumn)
{
    cudf::column_view zero_size_column( cudf::data_type{cudf::INT32}, 0, nullptr, nullptr, 0);
    auto results = cudf::strings::from_integers(zero_size_column);
    cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsConvertTest, ZeroSizeIntegersColumn)
{
    cudf::column_view zero_size_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto results = cudf::strings::to_integers(zero_size_column, cudf::data_type{cudf::INT32});
    EXPECT_EQ(0,results->size());
}

template <typename T>
class StringsIntegerConvertTest : public StringsConvertTest {};

using IntegerTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_CASE(StringsIntegerConvertTest, IntegerTypes);

TYPED_TEST(StringsIntegerConvertTest, FromToInteger)
{
    cudf::size_type size = 255;
    thrust::device_vector<TypeParam> d_integers(size);
    thrust::sequence( thrust::device, d_integers.begin(), d_integers.end(), -(size/2) );
    auto integers = cudf::make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<TypeParam>()}, size);
    auto integers_view = integers->mutable_view();
    cudaMemcpy( integers_view.data<TypeParam>(), d_integers.data().get(), size * sizeof(TypeParam), cudaMemcpyDeviceToDevice );
    integers_view.set_null_count(0);

    // convert to strings
    auto results_strings = cudf::strings::from_integers(integers->view());

    thrust::host_vector<TypeParam> h_integers(d_integers);
    std::vector<std::string> h_strings;
    for( auto itr = h_integers.begin(); itr != h_integers.end(); ++itr )
        h_strings.push_back(std::to_string(*itr));

    cudf::test::strings_column_wrapper expected( h_strings.begin(), h_strings.end() );
    cudf::test::expect_columns_equal(*results_strings,expected);

    // convert back to integers
    auto strings_view = cudf::strings_column_view(results_strings->view());
    auto results_integers = cudf::strings::to_integers(strings_view, cudf::data_type(cudf::experimental::type_to_id<TypeParam>()));
    cudf::test::expect_columns_equal(*results_integers,integers->view());
}

//
template <typename T>
class StringsFloatConvertTest : public StringsConvertTest {};

using FloatTypes = cudf::test::Types<float, double>;
TYPED_TEST_CASE(StringsFloatConvertTest, FloatTypes);

TYPED_TEST(StringsFloatConvertTest, FromToIntegerError)
{
    auto dtype = cudf::data_type{cudf::experimental::type_to_id<TypeParam>()};
    auto column = cudf::make_numeric_column(dtype, 100);
    EXPECT_THROW(cudf::strings::from_integers(column->view()), cudf::logic_error);

    cudf::test::strings_column_wrapper strings{ "this string intentionally left blank" };
    EXPECT_THROW(cudf::strings::to_integers(column->view(),dtype), cudf::logic_error);
}


TEST_F(StringsConvertTest, HexToInteger)
{
    std::vector<const char*> h_strings{ "1234", nullptr, "98BEEF", "1a5", "CAFE", "2face", "0xAABBCCDD", "112233445566" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    std::vector<int32_t> h_expected;
    for( auto itr = h_strings.begin(); itr != h_strings.end(); ++itr )
    {
        if( *itr==nullptr )
            h_expected.push_back(0);
        else
           h_expected.push_back( (int)std::stol(std::string(*itr),0,16) );
    }

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::hex_to_integers(strings_view, cudf::data_type{cudf::INT32} );

    cudf::test::fixed_width_column_wrapper<int32_t> expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results, expected);
}
