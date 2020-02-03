#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <tests/utilities/table_utilities.hpp>

#include <cudf/quantiles.hpp>

using namespace cudf::test;

template <typename T>
struct QuantilesTest : public BaseFixture {
};

// using TestTypes = AllTypes;
using TestTypes = cudf::test::Types<int8_t>;

TYPED_TEST_CASE(QuantilesTest, TestTypes);

TYPED_TEST(QuantilesTest, TestMulticolumn) {

    using T = TypeParam;

    auto input_a = fixed_width_column_wrapper<T>(
        { 0, 1, 2, 3, 4 },
        { 1, 1, 1, 1, 1 });

    auto inut_b = strings_column_wrapper(
        { "a", "b", "c", "d", "e" },
        {  1,   1,   1,   1,   1  });

    auto input = cudf::table_view({ input_a, inut_b });

    auto actual = cudf::experimental::quantiles(input, { 0.0f, 0.5f, 0.5f, 1.0f });

    auto expected_a = fixed_width_column_wrapper<T>(
        { 0, 2, 2, 4 },
        { 1, 1, 1, 1 }
    );

    auto expected_b = strings_column_wrapper(
        { "a", "c", "c", "e" },
        {  1,   1,   1,   1    });

    auto expected = cudf::table_view({ expected_a, expected_b });

    expect_tables_equal(expected, actual->view());
}
