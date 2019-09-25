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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/column_utilities.cuh>
#include "./utilities.h"

#include <vector>


struct AttrsTest : public GdfTest {};


TEST_F(AttrsTest, BytesCounts)
{
    std::vector<const char*> h_test_strings{ "xyz", "", "aé", nullptr, "bbb", "éé" };
    std::vector<int32_t> h_bytes{ 3, 0, 3, 0, 3, 4 };
    std::vector<cudf::bitmask_type> h_nbits{ 0x0037 };

    auto strings = cudf::test::create_strings_column(h_test_strings);
    auto strings_view = cudf::strings_column_view(strings->view());

    auto column = cudf::strings::bytes_counts(strings_view);
    rmm::device_vector<int32_t> d_expected(h_bytes);
    rmm::device_vector<cudf::bitmask_type> d_nbits(h_nbits);
    cudf::column_view column_expected( cudf::data_type{cudf::INT32}, d_expected.size(),
        d_expected.data().get(), d_nbits.data().get(), 1 );
    cudf::test::expect_columns_equal(column->view(), column_expected);
}

TEST_F(AttrsTest, CharactersCounts)
{
    std::vector<const char*> h_test_strings{ "xyz", "", "aé", nullptr, "bbb", "éé" };
    std::vector<int32_t> h_characters{ 3, 0, 2, 0, 3, 2 };
    std::vector<cudf::bitmask_type> h_nbits{ 0x0037 };

    auto strings = cudf::test::create_strings_column(h_test_strings);
    auto strings_view = cudf::strings_column_view(strings->view());

    auto column = cudf::strings::characters_counts(strings_view);
    rmm::device_vector<int32_t> d_expected(h_characters);
    rmm::device_vector<cudf::bitmask_type> d_nbits(h_nbits);
    cudf::column_view column_expected( cudf::data_type{cudf::INT32}, d_expected.size(),
        d_expected.data().get(), d_nbits.data().get(), 1 );
    cudf::test::expect_columns_equal(column->view(), column_expected);
}

TEST_F(AttrsTest, CodePoints)
{
    std::vector<const char*> h_test_strings{ "xyz", "", "aé", nullptr, "bbb", "éé" };
    std::vector<int32_t> h_codepoints{ 120, 121, 122, 97, 50089, 98, 98, 98, 50089, 50089 };

    auto strings = cudf::test::create_strings_column(h_test_strings);
    auto strings_view = cudf::strings_column_view(strings->view());

    auto column = cudf::strings::code_points(strings_view);
    rmm::device_vector<int32_t> d_expected(h_codepoints);
    cudf::column_view column_expected( cudf::data_type{cudf::INT32}, d_expected.size(),
        d_expected.data().get(), nullptr, 0 );
    cudf::test::expect_columns_equal(column->view(), column_expected);
}
