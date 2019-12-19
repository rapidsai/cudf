#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <cudf/groupby.hpp>
#include <cudf/detail/aggregation.hpp>
#include <cudf/table/table.hpp>

namespace cudf {
namespace test {
    
struct super_basic_test : public cudf::test::BaseFixture {};

TEST_F(super_basic_test, really_basic_mean) {
    fixed_width_column_wrapper<int32_t> keys        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    fixed_width_column_wrapper<float>   vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    fixed_width_column_wrapper<int32_t> expect_keys { 1, 2,    3};
    fixed_width_column_wrapper<double>  expect_vals { 3, 4.75, 17./3};

    auto mean_agg = cudf::experimental::make_mean_aggregation();

    std::vector<cudf::experimental::groupby::aggregation_request> requests;
    requests.emplace_back(
        cudf::experimental::groupby::aggregation_request());
    requests[0].values = vals;
    
    requests[0].aggregations.push_back(std::move(mean_agg));
    

    cudf::experimental::groupby::groupby gb_obj(table_view({keys}));

    auto result = gb_obj.aggregate(requests);
    expect_tables_equal(table_view({expect_keys}), result.first->view());
    expect_columns_equal(expect_vals, result.second[0].results[0]->view());
}

auto all_valid() {
    auto all_valid = cudf::test::make_counting_transform_iterator(
        0, [](auto i) { return true; });
    return all_valid;
}

struct GroupSTDTest : public cudf::test::BaseFixture {};

TEST_F(GroupSTDTest, SingleColumn)
{
    auto keys = fixed_width_column_wrapper<int32_t>        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto vals = fixed_width_column_wrapper<float>          { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

                                                       //  { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3}
    auto expect_keys = fixed_width_column_wrapper<int32_t> { 1,       2,          3      };
                                                       //  { 0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
    auto expect_vals = fixed_width_column_wrapper<double> ({    3,   sqrt(131./12),sqrt(31./3)},
        all_valid());
    
    auto std_agg = cudf::experimental::make_std_aggregation();

    std::vector<cudf::experimental::groupby::aggregation_request> requests;
    requests.emplace_back(
        cudf::experimental::groupby::aggregation_request());
    requests[0].values = vals;
    
    requests[0].aggregations.push_back(std::move(std_agg));
    
    cudf::experimental::groupby::groupby gb_obj(table_view({keys}));

    auto result = gb_obj.aggregate(requests);
    
    expect_columns_equal(expect_vals, result.second[0].results[0]->view(), true);
}

TEST_F(GroupSTDTest, SingleColumnNullable)
{
    fixed_width_column_wrapper<int32_t> keys(   { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                                { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
    fixed_width_column_wrapper<double>  vals(   { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                { 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

                                                    //  { 1, 1,     2, 2, 2,   3, 3,      4}
    fixed_width_column_wrapper<int32_t> expect_keys(    { 1,        2,         3,         4});
                                                    //  { 3, 6,     1, 4, 9,   2, 8,      -}
    fixed_width_column_wrapper<double>  expect_vals(    {3/sqrt(2), 7/sqrt(3), 3*sqrt(2), 0},
                                                        {1,           1,         1,       0});
    

    auto std_agg = cudf::experimental::make_std_aggregation();

    std::vector<cudf::experimental::groupby::aggregation_request> requests;
    requests.emplace_back(
        cudf::experimental::groupby::aggregation_request());
    requests[0].values = vals;
    
    requests[0].aggregations.push_back(std::move(std_agg));
    
    cudf::experimental::groupby::groupby gb_obj(table_view({keys}));

    auto result = gb_obj.aggregate(requests);
    
    expect_columns_equal(expect_vals, result.second[0].results[0]->view(), true);
}

struct group_quantile : public cudf::test::BaseFixture {};

TEST_F(group_quantile, SingleColumn)
{
    auto keys = fixed_width_column_wrapper<int32_t>        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto vals = fixed_width_column_wrapper<float>          { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

                                                       //  { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3}
    auto expect_keys = fixed_width_column_wrapper<int32_t> { 1,       2,          3      };
                                                       //  { 0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
    auto expect_vals = fixed_width_column_wrapper<double> ({    3,        4.5,       7   },
        all_valid());
    
    // auto quantile_agg = cudf::experimental::make_median_aggregation();
    auto quantile_agg = cudf::experimental::make_quantile_aggregation({0.5});

    std::vector<cudf::experimental::groupby::aggregation_request> requests;
    requests.emplace_back(
        cudf::experimental::groupby::aggregation_request());
    requests[0].values = vals;
    
    requests[0].aggregations.push_back(std::move(quantile_agg));
    
    cudf::experimental::groupby::groupby gb_obj(table_view({keys}));
    auto result = gb_obj.aggregate(requests);

    expect_columns_equal(expect_vals, result.second[0].results[0]->view(), true);
}

} // namespace test
} // namespace cudf
