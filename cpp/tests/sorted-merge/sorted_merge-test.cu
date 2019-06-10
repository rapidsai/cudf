#include <cassert>

#include <gtest/gtest.h>

#include "tests/utilities/column_wrapper.cuh"
 #include "tests/utilities/cudf_test_fixtures.h"

#include <cudf.h>
#include <cudf/functions.h>

std::vector<gdf_valid_type> bools_to_valids(const std::vector<uint8_t> & input) {
    std::vector<gdf_valid_type> vec(gdf_valid_allocation_size(input.size()), 0);
    for(size_t i = 0; i < input.size(); ++i){
        if(input[i]){
            gdf::util::turn_bit_on(vec.data(), i);
        } else {
            gdf::util::turn_bit_off(vec.data(), i);
        }
    }
    return vec;
}

template <typename T>
class SortedMergeTest : public GdfTest {};

using SortedMergerTypes = ::testing::Types<int8_t, int32_t, double>;

TYPED_TEST_CASE(SortedMergeTest, SortedMergerTypes);

TYPED_TEST(SortedMergeTest, Merge1KeyColumns) {
    cudf::test::column_wrapper<TypeParam> leftColWrap1({0, 1, 2, 3}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(4),0xFF));
    cudf::test::column_wrapper<TypeParam> leftColWrap2({4, 5, 6, 7}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(4),0xFF));
    gdf_column *leftColumn1 = leftColWrap1.get();
    gdf_column *leftColumn2 = leftColWrap2.get();

    cudf::test::column_wrapper<TypeParam> rightColWrap1({1, 2}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(2),0xFF));
    cudf::test::column_wrapper<TypeParam> rightColWrap2({8, 9}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(2),0xFF));

    gdf_column *rightColumn1 = rightColWrap1.get();
    gdf_column *rightColumn2 = rightColWrap2.get();

    const gdf_size_type outputLength = leftColumn1->size + rightColumn1->size;
    cudf::test::column_wrapper<TypeParam> outputColWrap1(std::vector<TypeParam>(outputLength), std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));
    cudf::test::column_wrapper<TypeParam> outputColWrap2(std::vector<TypeParam>(outputLength), std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));
    gdf_column *outputColumn1 = outputColWrap1.get();
    gdf_column *outputColumn2 = outputColWrap2.get();

    gdf_column *leftColumns[]  = {leftColumn1, leftColumn2};
    gdf_column *rightColumns[] = {rightColumn1, rightColumn2};

    gdf_column *outputColumns[] = {outputColumn1, outputColumn2};

    cudf::test::column_wrapper<int8_t> ordersWrap(std::vector<int8_t>({GDF_ORDER_ASC}));
    gdf_column *orders = ordersWrap.get();

    std::vector<gdf_size_type> sortByCols = {0};

    const gdf_size_type columnsLength = 2;
    gdf_error  gdfError = gdf_sorted_merge(leftColumns,
                                          rightColumns,
                                          columnsLength,
                                          sortByCols.data(),
                                          sortByCols.size(),
                                          orders,
                                          outputColumns);

    // print_gdf_column(leftColWrap1.get());
    // print_gdf_column(leftColWrap2.get());
    // print_gdf_column(rightColWrap1.get());
    // print_gdf_column(rightColWrap2.get());
    // print_gdf_column(outputColWrap1.get());
    // print_gdf_column(outputColWrap2.get());
    // print_gdf_column(expectedDataWrap1.get());
    // print_gdf_column(expectedDataWrap2.get());

    EXPECT_EQ(GDF_SUCCESS, gdfError);

    cudf::test::column_wrapper<TypeParam> expectedDataWrap1({0, 1, 1, 2, 2, 3}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));
    cudf::test::column_wrapper<TypeParam> expectedDataWrap2({4, 5, 8, 6, 9, 7}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));

    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap1.get(), outputColWrap1.get()));
    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap2.get(), outputColWrap2.get()));
}

TYPED_TEST(SortedMergeTest, Merge2KeyColumns) {
    cudf::test::column_wrapper<TypeParam> leftColWrap1({0, 1, 2, 3}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(4),0xFF));
    cudf::test::column_wrapper<TypeParam> leftColWrap2({4, 5, 6, 7}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(4),0xFF));
    gdf_column *leftColumn1 = leftColWrap1.get();
    gdf_column *leftColumn2 = leftColWrap2.get();

    cudf::test::column_wrapper<TypeParam> rightColWrap1({1, 2}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(2),0xFF));
    cudf::test::column_wrapper<TypeParam> rightColWrap2({8, 9}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(2),0xFF));

    gdf_column *rightColumn1 = rightColWrap1.get();
    gdf_column *rightColumn2 = rightColWrap2.get();

    const gdf_size_type outputLength = leftColumn1->size + rightColumn1->size;
    cudf::test::column_wrapper<TypeParam> outputColWrap1(std::vector<TypeParam>(outputLength), std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));
    cudf::test::column_wrapper<TypeParam> outputColWrap2(std::vector<TypeParam>(outputLength), std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));
    gdf_column *outputColumn1 = outputColWrap1.get();
    gdf_column *outputColumn2 = outputColWrap2.get();

    gdf_column *leftColumns[]  = {leftColumn1, leftColumn2};
    gdf_column *rightColumns[] = {rightColumn1, rightColumn2};

    gdf_column *outputColumns[] = {outputColumn1, outputColumn2};

    cudf::test::column_wrapper<int8_t> ordersWrap(std::vector<int8_t>({GDF_ORDER_ASC, GDF_ORDER_DESC}));
    gdf_column *orders = ordersWrap.get();

    std::vector<gdf_size_type> sortByCols = {0, 1};

    const gdf_size_type columnsLength = 2;
    gdf_error  gdfError = gdf_sorted_merge(leftColumns,
                                          rightColumns,
                                          columnsLength,
                                          sortByCols.data(),
                                          sortByCols.size(),
                                          orders,
                                          outputColumns);

    EXPECT_EQ(GDF_SUCCESS, gdfError);

    cudf::test::column_wrapper<TypeParam> expectedDataWrap1({0, 1, 1, 2, 2, 3}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));
    cudf::test::column_wrapper<TypeParam> expectedDataWrap2({4, 8, 5, 9, 6, 7}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));

    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap1.get(), outputColWrap1.get()));
    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap2.get(), outputColWrap2.get()));
}

TYPED_TEST(SortedMergeTest, Merge2KeyNullColumns) {
    cudf::test::column_wrapper<TypeParam> leftColWrap1({0, 1, 2, 3, -1}, bools_to_valids({1, 1, 1, 1, 0}));
    cudf::test::column_wrapper<TypeParam> leftColWrap2({4, 5, 6, 7, 1}, bools_to_valids({1, 1, 0, 1, 1}));
    gdf_column *leftColumn1 = leftColWrap1.get();
    gdf_column *leftColumn2 = leftColWrap2.get();

    cudf::test::column_wrapper<TypeParam> rightColWrap1({1, 2, -1}, bools_to_valids({1, 1, 0}));
    cudf::test::column_wrapper<TypeParam> rightColWrap2({8, 9, 2}, bools_to_valids({0, 1, 1}));

    gdf_column *rightColumn1 = rightColWrap1.get();
    gdf_column *rightColumn2 = rightColWrap2.get();

    const gdf_size_type outputLength = leftColumn1->size + rightColumn1->size;
    cudf::test::column_wrapper<TypeParam> outputColWrap1(std::vector<TypeParam>(outputLength), std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength)));
    cudf::test::column_wrapper<TypeParam> outputColWrap2(std::vector<TypeParam>(outputLength), std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength)));
    gdf_column *outputColumn1 = outputColWrap1.get();
    gdf_column *outputColumn2 = outputColWrap2.get();

    gdf_column *leftColumns[]  = {leftColumn1, leftColumn2};
    gdf_column *rightColumns[] = {rightColumn1, rightColumn2};

    gdf_column *outputColumns[] = {outputColumn1, outputColumn2};

    cudf::test::column_wrapper<int8_t> ordersWrap(std::vector<int8_t>({GDF_ORDER_ASC, GDF_ORDER_DESC}));
    gdf_column *orders = ordersWrap.get();

    std::vector<gdf_size_type> sortByCols = {0, 1};

    const gdf_size_type columnsLength = 2;
    gdf_error  gdfError = gdf_sorted_merge(leftColumns,
                                          rightColumns,
                                          columnsLength,
                                          sortByCols.data(),
                                          sortByCols.size(),
                                          orders,
                                          outputColumns);

    EXPECT_EQ(GDF_SUCCESS, gdfError);

    cudf::test::column_wrapper<TypeParam> expectedDataWrap1({0, 1, 1, 2, 2, 3, -1, -1}, bools_to_valids({1, 1, 1, 1, 1, 1, 0, 0}));
    cudf::test::column_wrapper<TypeParam> expectedDataWrap2({4, 8, 5, 6, 9, 7, 2, 1}, bools_to_valids({1, 0, 1, 0, 1, 1, 1, 1}));

    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap1.get(), outputColWrap1.get()));
    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap2.get(), outputColWrap2.get()));
}
