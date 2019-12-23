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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <cudf/groupby.hpp>
#include <cudf/detail/aggregation.hpp>
#include <cudf/table/table.hpp>

namespace cudf {
namespace test {


struct groupby_sum_test : public cudf::test::BaseFixture {
    void run_test(column_view const& keys,
                  column_view const& values,
                  column_view const& expect_keys,
                  column_view const& expect_vals,
                  std::unique_ptr<experimental::aggregation>&& agg)
    {
        std::vector<cudf::experimental::groupby::aggregation_request> requests;
        requests.emplace_back(
            cudf::experimental::groupby::aggregation_request());
        requests[0].values = values;
        
        requests[0].aggregations.push_back(std::move(agg));
        

        cudf::experimental::groupby::groupby gb_obj(table_view({keys}));

        auto result = gb_obj.aggregate(requests);
        expect_tables_equal(table_view({expect_keys}), result.first->view());
        expect_columns_equal(expect_vals, result.second[0].results[0]->view(), true);
    }
};


TEST_F(groupby_sum_test, basic)
{
    fixed_width_column_wrapper<int32_t> keys        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    fixed_width_column_wrapper<float>   vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    fixed_width_column_wrapper<int32_t> expect_keys { 1, 2,  3 };
    fixed_width_column_wrapper<float>   expect_vals { 9, 19, 17};

    auto agg = cudf::experimental::make_sum_aggregation();
    run_test(keys, vals, expect_keys, expect_vals, std::move(agg));
}


} // namespace test
} // namespace cudf
