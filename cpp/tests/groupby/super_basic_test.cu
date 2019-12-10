#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/groupby.hpp>
#include <cudf/detail/aggregation.hpp>
#include <cudf/table/table.hpp>

struct super_basic_test : public cudf::test::BaseFixture {};

namespace cudf {
namespace test {
    
TEST_F(super_basic_test, really_basic) {
    fixed_width_column_wrapper<int32_t> keys { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    fixed_width_column_wrapper<float>   vals { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto mean_agg = cudf::experimental::make_mean_aggregation();

    std::vector<cudf::experimental::groupby::aggregation_request> requests;
    requests.emplace_back(
        cudf::experimental::groupby::aggregation_request());
    requests[0].values = vals;
    
    requests[0].aggregations.push_back(std::move(mean_agg));
    

    cudf::experimental::groupby::groupby gb_obj(table_view({keys}));

    auto result = gb_obj.aggregate(requests);
    printf("done");
}

} // namespace test
} // namespace cudf
