/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <nvtext/generate_ngrams.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>

#include <vector>


struct TextGenerateNgramsTest : public cudf::test::BaseFixture {};

TEST_F(TextGenerateNgramsTest, Ngrams)
{
    std::vector<const char*> h_strings{ "the", "fox", "jumped", "over", "the", "dog" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    cudf::strings_column_view strings_view( strings );

    {
        cudf::test::strings_column_wrapper expected{ "the_fox","fox_jumped","jumped_over","over_the","the_dog" };
        auto results = nvtext::generate_ngrams(strings_view);
        cudf::test::expect_columns_equal(*results,expected);
    }

    {
        cudf::test::strings_column_wrapper expected{ "the_fox_jumped","fox_jumped_over","jumped_over_the","over_the_dog" };
        auto results = nvtext::generate_ngrams(strings_view,3);
        cudf::test::expect_columns_equal(*results,expected);
    }
}

