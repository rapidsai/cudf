#include <cassert>

#include <gtest/gtest.h>

#include "tests/utilities/column_wrapper.cuh"
 #include "tests/utilities/cudf_test_fixtures.h"

#include <cudf.h>
#include <cudf/functions.h>



template <typename T>
class SortedMergeTest : public GdfTest {};

using SortedMergerTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_CASE(SortedMergeTest, SortedMergerTypes);

TYPED_TEST(SortedMergeTest, MergeTwoSortedColumns) {
    cudf::test::column_wrapper<TypeParam> leftColWrap1({0, 1, 2, 3}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(4),0xFF));
    cudf::test::column_wrapper<TypeParam> leftColWrap2({4, 5, 6, 7}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(4),0xFF));
    // cudf::test::column_wrapper<TypeParam> leftColWrap1(std::vector<TypeParam>({0, 1, 2, 3}));
    // cudf::test::column_wrapper<TypeParam> leftColWrap2(std::vector<TypeParam>({4, 5, 6, 7}));
    gdf_column *leftColumn1 = leftColWrap1.get();
    gdf_column *leftColumn2 = leftColWrap2.get();

    cudf::test::column_wrapper<TypeParam> rightColWrap1({1, 2}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(2),0xFF));
    cudf::test::column_wrapper<TypeParam> rightColWrap2({8, 9}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(2),0xFF));
    // cudf::test::column_wrapper<TypeParam> rightColWrap1(std::vector<TypeParam>({1, 2}));
    // cudf::test::column_wrapper<TypeParam> rightColWrap2(std::vector<TypeParam>({8, 9}));

    gdf_column *rightColumn1 = rightColWrap1.get();
    gdf_column *rightColumn2 = rightColWrap2.get();

    const gdf_size_type outputLength = leftColumn1->size + rightColumn1->size;
    cudf::test::column_wrapper<TypeParam> outputColWrap1(std::vector<TypeParam>(outputLength), std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));
    cudf::test::column_wrapper<TypeParam> outputColWrap2(std::vector<TypeParam>(outputLength), std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));
    // cudf::test::column_wrapper<TypeParam> outputColWrap1(std::vector<TypeParam>(0, outputLength));
    // cudf::test::column_wrapper<TypeParam> outputColWrap2(std::vector<TypeParam>(0, outputLength));
    gdf_column *outputColumn1 = outputColWrap1.get();
    gdf_column *outputColumn2 = outputColWrap2.get();

    gdf_column *leftColumns[]  = {leftColumn1, leftColumn2};
    gdf_column *rightColumns[] = {rightColumn1, rightColumn2};

    gdf_column *outputColumns[] = {outputColumn1, outputColumn2};

    cudf::test::column_wrapper<int8_t> ordersWrap(std::vector<int8_t>({GDF_ORDER_ASC, GDF_ORDER_DESC}));
    gdf_column *orders = ordersWrap.get();

    cudf::test::column_wrapper<int32_t> outputIndicesWrap({0});
    gdf_column *outputIndices = outputIndicesWrap.get();

    const gdf_size_type columnsLength = 2;
    gdf_error         gdfError      = gdf_sorted_merge(leftColumns,
                                          rightColumns,
                                          columnsLength,
                                          outputIndices,
                                          orders,
                                          outputColumns);

    print_gdf_column(leftColWrap1.get());
    print_gdf_column(leftColWrap2.get());
    print_gdf_column(rightColWrap1.get());
    print_gdf_column(rightColWrap2.get());
    print_gdf_column(outputColWrap1.get());
    print_gdf_column(outputColWrap2.get());

    EXPECT_EQ(GDF_SUCCESS, gdfError);

    cudf::test::column_wrapper<TypeParam> expectedDataWrap1({0, 1, 1, 2, 2, 3}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));
    cudf::test::column_wrapper<TypeParam> expectedDataWrap2({4, 5, 8, 9, 6, 7}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));
    // cudf::test::column_wrapper<TypeParam> expectedDataWrap1(std::vector<TypeParam>({0, 1, 1, 2, 2, 3}));
    // cudf::test::column_wrapper<TypeParam> expectedDataWrap2(std::vector<TypeParam>({4, 5, 8, 9, 6, 7}));

    print_gdf_column(expectedDataWrap1.get());
    print_gdf_column(expectedDataWrap2.get());


    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap1.get(), outputColWrap1.get()));
    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap2.get(), outputColWrap2.get()));
}
