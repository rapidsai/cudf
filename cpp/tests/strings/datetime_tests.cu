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
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringsDatetimeTest : public cudf::test::BaseFixture {};

TEST_F(StringsDatetimeTest, ToTimestamp)
{
    std::vector<const char*> h_strings{ "1974-02-28T01:23:45Z", "2019-07-17T21:34:37Z", nullptr, "", "2019-03-20T12:34:56Z", "2020-02-29T00:00:00Z" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<cudf::timestamp_s> h_expected{ 131246625, 1563399277, 0,0, 1553085296, 1582934400 };

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::to_timestamps(strings_view, cudf::data_type{cudf::TIMESTAMP_SECONDS}, "%Y-%m-%dT%H:%M:%SZ" );

    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s> expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsDatetimeTest, FromTimestamp)
{
    std::vector<cudf::timestamp_s> h_timestamps{ 131246625 , 1563399277, 0, 1553085296, 1582934400 };
    std::vector<const char*> h_expected{ "1974-02-28T01:23:45Z", "2019-07-17T21:34:37Z", nullptr, "2019-03-20T12:34:56Z", "2020-02-29T00:00:00Z" };

    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s> timestamps( h_timestamps.begin(), h_timestamps.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));

    auto results = cudf::strings::from_timestamps(timestamps);
    //cudf::strings::print(cudf::strings_column_view(*results));

    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsDatetimeTest, ZeroSizeStringsColumn)
{
    cudf::column_view zero_size_column( cudf::data_type{cudf::TIMESTAMP_SECONDS}, 0, nullptr, nullptr, 0);
    auto results = cudf::strings::from_timestamps(zero_size_column);
    cudf::test::expect_strings_empty(results->view());
}
