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

#include <cudf/strings/strings_column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/column_utilities.cuh>
#include "./utilities.h"

#include <vector>
#include <cstring>


struct AttrsTest : public GdfTest {};


TEST_F(AttrsTest, BytesCounts)
{
    std::vector<const char*> h_test_strings{ "xyz", "", "aé", nullptr, "bbb", "éé" };
    std::vector<int32_t> h_bytes{ 3, 0, 3, 0, 3, 4 };

    auto strings = cudf::test::create_strings_column(h_test_strings);
    auto strings_view = cudf::strings_column_view(strings->view());
    cudf::size_type count = strings_view.size();
    cudf::strings::print(strings_view);

    auto column = cudf::strings::bytes_counts(strings_view);
    rmm::device_vector<int32_t> d_expected(h_bytes);
    cudf::column_view column_expected( cudf::data_type{cudf::INT32}, count,
        d_expected.data().get(), nullptr, 0 );
    cudf::test::expect_columns_equal(column->view(), column_expected);
}

TEST_F(AttrsTest, CharactersCounts)
{
    std::vector<const char*> h_test_strings{ "xyz", "", "aé", nullptr, "bbb", "éé" };
    std::vector<int32_t> h_characters{ 3, 0, 2, 0, 3, 2 };

    auto strings = cudf::test::create_strings_column(h_test_strings);
    auto strings_view = cudf::strings_column_view(strings->view());
    cudf::size_type count = strings_view.size();

    auto column = cudf::strings::characters_counts(strings_view);
    rmm::device_vector<int32_t> d_expected(h_characters);
    cudf::column_view column_expected( cudf::data_type{cudf::INT32}, count,
        d_expected.data().get(), nullptr, 0 );
    cudf::test::expect_columns_equal(column->view(), column_expected);
}

