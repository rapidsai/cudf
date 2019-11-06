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
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/wrappers/bool.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringsCharsTest : public cudf::test::BaseFixture {};

class StringsCharsTypesTest : public StringsCharsTest,
                       public testing::WithParamInterface<cudf::strings::string_character_types> {};

TEST_P(StringsCharsTypesTest, AllTypes)
{
    std::vector<const char*> h_strings{ "Héllo", "thesé", nullptr, "HERE", "tést strings", "",
        "1.75", "-34", "+9.8", "17¼", "x³", "2³", " 12⅝",
        "1234567890", "de", "\t\r\n\f "};

    cudf::experimental::bool8 expecteds[] = {
                           0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,   // decimal
                           0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,   // numeric
                           0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,   // digit
                           1,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,   // alpha
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,   // space
                           0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,   // upper
                           0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0 }; // lower

    auto is_parm = GetParam();

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    auto results = cudf::strings::all_characters_of_type(strings_view,is_parm);

    int x = static_cast<int>(is_parm);
    int index = 0;
    int strings_count = static_cast<int>(h_strings.size());
    while( x >>= 1 ) ++index;
    cudf::experimental::bool8* sub_expected = &expecteds[index * strings_count];

    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( sub_expected, sub_expected + strings_count,
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

INSTANTIATE_TEST_CASE_P(StringsAllCharsTypes, StringsCharsTypesTest,
    testing::ValuesIn(std::array<cudf::strings::string_character_types,7>
    { cudf::strings::string_character_types::DECIMAL,
      cudf::strings::string_character_types::NUMERIC,
      cudf::strings::string_character_types::DIGIT,
      cudf::strings::string_character_types::ALPHA,
      cudf::strings::string_character_types::SPACE,
      cudf::strings::string_character_types::UPPER,
      cudf::strings::string_character_types::LOWER}));


TEST_F(StringsCharsTest, Alphanumeric)
{
    std::vector<const char*> h_strings{ "Héllo", "thesé", nullptr, "HERE", "tést strings", "",
        "1.75", "-34", "+9.8", "17¼", "x³", "2³", " 12⅝",
        "1234567890", "de", "\t\r\n\f "};

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    auto results = cudf::strings::all_characters_of_type(strings_view,cudf::strings::string_character_types::ALPHANUM);

    std::vector<cudf::experimental::bool8> h_expected{ 1,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0 };
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected.begin(), h_expected.end(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsCharsTest, AlphaNumericSpace)
{
    std::vector<const char*> h_strings{ "Héllo", "thesé", nullptr, "HERE", "tést strings", "",
        "1.75", "-34", "+9.8", "17¼", "x³", "2³", " 12⅝",
        "1234567890", "de", "\t\r\n\f "};

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    auto types = cudf::strings::string_character_types::ALPHANUM | cudf::strings::string_character_types::SPACE;
    auto results = cudf::strings::all_characters_of_type(strings_view, (cudf::strings::string_character_types)types);

    std::vector<cudf::experimental::bool8> h_expected{ 1,1,0,1,1,0,0,0,0,1,1,1,1,1,1,1 };
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected.begin(), h_expected.end(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsCharsTest, Numerics)
{
    std::vector<const char*> h_strings{ "Héllo", "thesé", nullptr, "HERE", "tést strings", "",
        "1.75", "-34", "+9.8", "17¼", "x³", "2³", " 12⅝",
        "1234567890", "de", "\t\r\n\f "};

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    auto types = cudf::strings::string_character_types::DIGIT |
                 cudf::strings::string_character_types::DECIMAL |
                 cudf::strings::string_character_types::NUMERIC;
    auto results = cudf::strings::all_characters_of_type(strings_view, (cudf::strings::string_character_types)types);

    std::vector<cudf::experimental::bool8> h_expected{ 0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0 };
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected.begin(), h_expected.end(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsCharsTest, EmptyStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    auto results = cudf::strings::all_characters_of_type(strings_view,cudf::strings::string_character_types::ALPHANUM);
    auto view = results->view();
    EXPECT_EQ(cudf::BOOL8, view.type().id());
    EXPECT_EQ(0,view.size());
    EXPECT_EQ(0,view.null_count());
    EXPECT_EQ(0,view.num_children());
}
