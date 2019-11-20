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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/detail/copy_if_else.cuh>
#include <cudf/utilities/error.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringsCopyIfElseTest : public cudf::test::BaseFixture {};

struct filter_test_fn
{
    __host__ __device__ bool operator()(cudf::size_type idx) const
    {
        return static_cast<bool>(idx % 2);
    }
};

TEST_F(StringsCopyIfElseTest, CopyIfElse)
{
    std::vector<const char*> h_strings1{ "eee", "bb", "", "aa", "bbb", "ééé" };
    cudf::test::strings_column_wrapper strings1( h_strings1.begin(), h_strings1.end() );
    auto lhs = cudf::strings_column_view(strings1);
    std::vector<const char*> h_strings2{ "zz",  "", "yyy", "w", "ééé", "ooo" };
    cudf::test::strings_column_wrapper strings2( h_strings2.begin(), h_strings2.end() );
    auto rhs = cudf::strings_column_view(strings2);

    auto results = cudf::strings::detail::copy_if_else(lhs,rhs,filter_test_fn{});

    std::vector<const char*> h_expected;
    for( cudf::size_type idx=0; idx < static_cast<cudf::size_type>(h_strings1.size()); ++idx )
    {
        if( filter_test_fn()(idx) )
            h_expected.push_back( h_strings1[idx] );
        else
            h_expected.push_back( h_strings2[idx] );
    }
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end());
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsCopyIfElseTest, ZeroSizeStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto lhs = cudf::strings_column_view(zero_size_strings_column);
    auto rhs = cudf::strings_column_view(zero_size_strings_column);
    auto results = cudf::strings::detail::copy_if_else( lhs, rhs, filter_test_fn{});
    cudf::test::expect_strings_empty(results->view());
}
