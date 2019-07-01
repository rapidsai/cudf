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

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/column_wrapper.cuh>
#include <cudf/search.hpp>

using cudf::test::column_wrapper;

class SearchTest : public GdfTest {};

TEST_F(SearchTest, single_non_null_column_multiple_needles)
{
    using single_element_type = int64_t;

    std::vector<single_element_type> haystack_data       { 10, 20, 30, 40, 50 };
    std::vector<single_element_type> needle_data         {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    std::vector<gdf_size_type> first_greater_or_equal_to {  0,  0,  0,  1,  2,  3,  3,  4,  4,  5 };

    auto single_haystack_column = column_wrapper<single_element_type>(haystack_data);
    auto single_needle_column   = column_wrapper<single_element_type>(needle_data);

    gdf_column haystack_column = *(single_haystack_column.get() );
    gdf_column needle_column   = *(single_needle_column.get()   );
    gdf_column results_column;

    EXPECT_NO_THROW(
        results_column = cudf::search_sorted(
            haystack_column,
            needle_column,
            true)
    );

    auto results = column_wrapper<gdf_index_type>(results_column);
    auto expect = column_wrapper<gdf_index_type>(first_greater_or_equal_to);

    ASSERT_EQ(results, expect) << "  Actual:" << results.to_str()
                               << "Expected:" << expect.to_str();
}