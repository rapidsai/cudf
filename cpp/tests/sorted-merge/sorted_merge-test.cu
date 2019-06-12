#include <cassert>
#include <vector>
#include <memory>
#include <algorithm>
#include <gtest/gtest.h>
#include <nvstrings/NVCategory.h>
#include <nvstrings/NVStrings.h>

#include "tests/utilities/column_wrapper.cuh"
#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/utilities/nvcategory_utils.cuh"

#include <cudf.h>
#include <cudf/functions.h>
#include <sorted_merge.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <table.hpp>
#include "string/nvcategory_util.hpp"

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

TYPED_TEST(SortedMergeTest, MergeWithEmptyColumn) {
    cudf::test::column_wrapper<TypeParam> leftColWrap1({0, 1, 2, 3}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(4),0xFF));
    gdf_column *leftColumn1 = leftColWrap1.get();

    cudf::test::column_wrapper<TypeParam> rightColWrap1(0);
    gdf_column *rightColumn1 = rightColWrap1.get();

    gdf_column *leftColumns[]  = {leftColumn1};
    gdf_column *rightColumns[] = {rightColumn1};

    std::vector<gdf_size_type> sortByCols = {0};
    rmm::device_vector<int8_t> ordersDeviceVector(std::vector<int8_t>{GDF_ORDER_ASC});

    cudf::table outputTable = cudf::sorted_merge(cudf::table(leftColumns, 1),
                                                cudf::table(rightColumns, 1),
                                                sortByCols,
                                                ordersDeviceVector);

    const gdf_size_type outputLength = leftColumn1->size + rightColumn1->size;
    cudf::test::column_wrapper<TypeParam> expectedDataWrap1({0, 1, 2, 3}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));

    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap1.get(), outputTable.get_column(0)));
}

TYPED_TEST(SortedMergeTest, Merge1KeyColumns) {
    cudf::test::column_wrapper<TypeParam> leftColWrap1({0, 1, 2, 3}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(4),0xFF));
    cudf::test::column_wrapper<TypeParam> leftColWrap2({4, 5, 6, 7}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(4),0xFF));
    gdf_column *leftColumn1 = leftColWrap1.get();
    gdf_column *leftColumn2 = leftColWrap2.get();

    cudf::test::column_wrapper<TypeParam> rightColWrap1({1, 2}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(2),0xFF));
    cudf::test::column_wrapper<TypeParam> rightColWrap2({8, 9}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(2),0xFF));
    gdf_column *rightColumn1 = rightColWrap1.get();
    gdf_column *rightColumn2 = rightColWrap2.get();

    gdf_column *leftColumns[]  = {leftColumn1, leftColumn2};
    gdf_column *rightColumns[] = {rightColumn1, rightColumn2};

    std::vector<gdf_size_type> sortByCols = {0};
    rmm::device_vector<int8_t> ordersDeviceVector(std::vector<int8_t>{GDF_ORDER_ASC});

    cudf::table outputTable = cudf::sorted_merge(cudf::table(leftColumns, 2),
                                                cudf::table(rightColumns, 2),
                                                sortByCols,
                                                ordersDeviceVector);

    const gdf_size_type outputLength = leftColumn1->size + rightColumn1->size;
    cudf::test::column_wrapper<TypeParam> expectedDataWrap1({0, 1, 1, 2, 2, 3}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));
    cudf::test::column_wrapper<TypeParam> expectedDataWrap2({4, 5, 8, 6, 9, 7}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));

    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap1.get(), outputTable.get_column(0)));
    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap2.get(), outputTable.get_column(1)));
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

    gdf_column *leftColumns[]  = {leftColumn1, leftColumn2};
    gdf_column *rightColumns[] = {rightColumn1, rightColumn2};

    std::vector<gdf_size_type> sortByCols = {0, 1};
    rmm::device_vector<int8_t> ordersDeviceVector(std::vector<int8_t>{GDF_ORDER_ASC, GDF_ORDER_DESC});

    cudf::table outputTable = cudf::sorted_merge(cudf::table(leftColumns, 2),
                                                cudf::table(rightColumns, 2),
                                                sortByCols,
                                                ordersDeviceVector);

    const gdf_size_type outputLength = leftColumn1->size + rightColumn1->size;
    cudf::test::column_wrapper<TypeParam> expectedDataWrap1({0, 1, 1, 2, 2, 3}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));
    cudf::test::column_wrapper<TypeParam> expectedDataWrap2({4, 8, 5, 9, 6, 7}, std::vector<gdf_valid_type>(gdf_valid_allocation_size(outputLength),0xFF));

    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap1.get(), outputTable.get_column(0)));
    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap2.get(), outputTable.get_column(1)));
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

    gdf_column *leftColumns[]  = {leftColumn1, leftColumn2};
    gdf_column *rightColumns[] = {rightColumn1, rightColumn2};

    std::vector<gdf_size_type> sortByCols = {0, 1};
    rmm::device_vector<int8_t> ordersDeviceVector(std::vector<int8_t>{GDF_ORDER_ASC, GDF_ORDER_DESC});

    cudf::table outputTable = cudf::sorted_merge(cudf::table(leftColumns, 2),
                                                cudf::table(rightColumns, 2),
                                                sortByCols,
                                                ordersDeviceVector);

    cudf::test::column_wrapper<TypeParam> expectedDataWrap1({0, 1, 1, 2, 2, 3, -1, -1}, bools_to_valids({1, 1, 1, 1, 1, 1, 0, 0}));
    cudf::test::column_wrapper<TypeParam> expectedDataWrap2({4, 8, 5, 6, 9, 7, 2, 1}, bools_to_valids({1, 0, 1, 0, 1, 1, 1, 1}));

    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap1.get(), outputTable.get_column(0)));
    EXPECT_TRUE(gdf_equal_columns<TypeParam>(expectedDataWrap2.get(), outputTable.get_column(1)));
}


class SortedMergeStringTest : public GdfTest {};

TEST_F(SortedMergeStringTest, Merge1KeyColumns) {
    constexpr int STRING_LENGTH = 8;

    const char * leftStrings1[] = {"aaaaaaab", "aaaaaaaf", "aaaaaaak", "aaaaaaap", "aaaaaaat", "aaaaaaax", "aaaaaaaz"};
    gdf_column *leftColumn1 = cudf::test::create_nv_category_column_strings(leftStrings1, 7);

    const char * rightStrings1[] = {"aaaaaaad", "aaaaaaan", "aaaaaaay"};
    gdf_column *rightColumn1 = cudf::test::create_nv_category_column_strings(rightStrings1, 3);

    gdf_column *leftColumns[]  = {leftColumn1};
    gdf_column *rightColumns[] = {rightColumn1};

    std::vector<gdf_size_type> sortByCols = {0};
    rmm::device_vector<int8_t> ordersDeviceVector(std::vector<int8_t>{GDF_ORDER_ASC});

    cudf::table outputTable = cudf::sorted_merge(cudf::table(leftColumns, 1),
                                                cudf::table(rightColumns, 1),
                                                sortByCols,
                                                ordersDeviceVector);

    gdf_column * outCol = outputTable.get_column(0);
    const gdf_size_type outputLength = leftColumn1->size + rightColumn1->size;
    NVStrings * tempNVStrings = static_cast<NVCategory *>(outCol->dtype_info.category)->gather_strings( 
			(nv_category_index_type *) outCol->data, outputLength, DEVICE_ALLOCATED );

    std::vector<std::unique_ptr<char[]>> c_strings(outputLength);
    std::vector<char*> hostStrings(outputLength);
    for(gdf_size_type i = 0; i < outputLength; i++){
        c_strings[i] = std::make_unique<char[]>(STRING_LENGTH+1);
        hostStrings[i] = c_strings[i].get();
    }

    tempNVStrings->to_host(hostStrings.data(), 0, outputLength);
	
    for(gdf_size_type i = 0; i < outputLength; i++){
        hostStrings[i][STRING_LENGTH] = 0;
    }
    std::vector<std::string> outputStrings(hostStrings.begin(), hostStrings.end());

    NVStrings::destroy(tempNVStrings);

    std::vector<std::string> expectedOutput = {"aaaaaaab", "aaaaaaad", "aaaaaaaf", "aaaaaaak", "aaaaaaan", "aaaaaaap", "aaaaaaat", "aaaaaaax", "aaaaaaay", "aaaaaaaz"};

    EXPECT_TRUE(std::equal(expectedOutput.begin(), expectedOutput.end(), outputStrings.begin(), outputStrings.end()));
}
