//#include <nvstrings/NVCategory.h>
//#include <nvstrings/NVStrings.h>

#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf/merge.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/column/column_factories.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/cudf_gtest.hpp>
#include <cudf/utilities/legacy/wrapper_types.hpp>

#include <cassert>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <initializer_list>

#include <gtest/gtest.h>

template <typename T>
class MergeTest_ : public cudf::test::BaseFixture {};

//TODO: confirm if the legacy test_types below can be replaced
//just by cudf::test::NumericTypes
//
//legacy:
//{
///using test_types =
///  ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>
//,cudf::bool8>; //column_wrapper failure //, cudf::nvstring_category>; //string not ready

//TYPED_TEST_CASE(MergeTest_, test_types);
//}

TYPED_TEST_CASE(MergeTest_, cudf::test::NumericTypes);

TYPED_TEST(MergeTest_, MismatchedNumColumns) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    columnFactoryT leftColWrap1{{0,1,2,3}}; 
    columnFactoryT rightColWrap1{{0,1,2,3}};
    columnFactoryT rightColWrap2{{0,1,2,3}};

    std::vector<cudf::size_type> key_cols{0};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1, rightColWrap2}};

    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}



TYPED_TEST(MergeTest_, MismatchedColumnDypes) {
    cudf::test::fixed_width_column_wrapper<int32_t> leftColWrap1{{0,1,2,3}};
    cudf::test::fixed_width_column_wrapper<double> rightColWrap1{{0,1,2,3}};

    std::vector<cudf::size_type> key_cols{0};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1}};


    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}


TYPED_TEST(MergeTest_, EmptyKeyColumns) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    columnFactoryT leftColWrap1{{0,1,2,3}}; 
    columnFactoryT rightColWrap1{{0,1,2,3}};

    std::vector<cudf::size_type> key_cols{}; // empty! this should trigger exception
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1}};

    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}


TYPED_TEST(MergeTest_, TooManyKeyColumns) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    columnFactoryT leftColWrap1{{0,1,2,3}}; 
    columnFactoryT rightColWrap1{{0,1,2,3}};

    std::vector<cudf::size_type> key_cols{0, 1}; // more keys than columns: this should trigger exception
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1}};

    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}


TYPED_TEST(MergeTest_, EmptyOrderTypes) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    columnFactoryT leftColWrap1{{0,1,2,3}}; 
    columnFactoryT rightColWrap1{{0,1,2,3}};

    std::vector<cudf::size_type> key_cols{0};
    std::vector<cudf::order> column_order {}; // empty! this should trigger exception
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1}};

    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}


TYPED_TEST(MergeTest_, TooManyOrderTypes) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    columnFactoryT leftColWrap1{{0,1,2,3}}; 
    columnFactoryT rightColWrap1{{0,1,2,3}};

    std::vector<cudf::size_type> key_cols{0}; 
    std::vector<cudf::order> column_order {cudf::order::ASCENDING, cudf::order::DESCENDING}; // more order types than columns: this should trigger exception
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1}};

    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}

TYPED_TEST(MergeTest_, MismatchedKeyColumnsAndOrderTypes) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    columnFactoryT leftColWrap1{{0,1,2,3}};
    columnFactoryT leftColWrap2{{0,1,2,3}};
    columnFactoryT rightColWrap1{{0,1,2,3}};
    columnFactoryT rightColWrap2{{0,1,2,3}};

    cudf::table_view left_view{{leftColWrap1, leftColWrap2}};
    cudf::table_view right_view{{rightColWrap1, rightColWrap2}};
    
    std::vector<cudf::size_type> key_cols{0, 1};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{};

    std::vector<cudf::size_type> sortByCols = {0, 1};
    std::vector<order_by_type> orderByTypes = {GDF_ORDER_ASC};

    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}

TYPED_TEST(MergeTest_, MergeWithEmptyColumn) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;

    cudf::size_type inputRows = 50000;
    auto unwrap_max = cudf::detail::unwrap(std::numeric_limits<TypeParam>::max()); 
    inputRows = (static_cast<cudf::size_type>(unwrap_max) < inputRows ? 40 : inputRows);

    auto sequence = cudf::test::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });
    columnFactoryT leftColWrap1(sequence, sequence+inputRows);
    columnFactoryT rightColWrap1{};//wrapper of empty column <- this might require a (sequence, sequence) generator 

    std::vector<cudf::size_type> key_cols{0};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1}};

    std::unique_ptr<cudf::experimental::table> p_outputTable;
    EXPECT_NO_THROW(p_outputTable = cudf::experimental::merge(left_view,
                                                              right_view,
                                                              key_cols,
                                                              column_order,
                                                              null_precedence));

    const cudf::size_type outputRows = left_view.num_columns() +
      right_view.num_columns();
    
    columnFactoryT expectedDataWrap1(sequence, sequence+outputRows);

    auto expected_column_view{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    auto output_column_view{p_outputTable->view().column(0)};
    
    cudf::test::expect_columns_equal(expected_column_view, output_column_view);
}
/*
TYPED_TEST(MergeTest_, Merge1KeyColumns) {
    cudf::test::column_wrapper_factory<TypeParam> columnFactory;

    cudf::size_type inputRows = 50000;
    inputRows = (cudf::detail::unwrap(std::numeric_limits<TypeParam>::max()) < inputRows ? 40 : inputRows);

    auto leftColWrap1 = columnFactory.make(inputRows,
                                            [](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return 1;
                                                else return 2 * row; 
                                            });
    auto leftColWrap2 = columnFactory.make(inputRows,
                                            [](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return 0;
                                                else return row;
                                            });

    auto rightColWrap1 = columnFactory.make(inputRows,
                                            [](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return 0;
                                                else return 2 * row + 1;
                                            });
    auto rightColWrap2 = columnFactory.make(inputRows,
                                            [](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return 0;
                                                else return row;
                                            });

    std::vector<cudf::size_type> sortByCols = {0};
    std::vector<order_by_type> orderByTypes = {GDF_ORDER_ASC};

    cudf::table outputTable;
    EXPECT_NO_THROW(outputTable = cudf::merge(cudf::table{leftColWrap1.get(), leftColWrap2.get()},
                                            cudf::table{rightColWrap1.get(), rightColWrap2.get()},
                                            sortByCols,
                                            orderByTypes));

    const cudf::size_type outputRows = leftColWrap1.size() + rightColWrap1.size();
    auto expectedDataWrap1 = columnFactory.make(outputRows,
                                                [=](cudf::size_type row)->cudf::size_type {
                                                    if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return row >= outputRows / 2;
                                                    else return row;
                                                });
    auto expectedDataWrap2 = columnFactory.make(outputRows,
                                                [](cudf::size_type row)->cudf::size_type {
                                                    if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return 0;
                                                    else return row / 2; 
                                                });

    EXPECT_TRUE(gdf_equal_columns(*expectedDataWrap1.get(), *outputTable.get_column(0)));
    EXPECT_TRUE(gdf_equal_columns(*expectedDataWrap2.get(), *outputTable.get_column(1)));
}

TYPED_TEST(MergeTest_, Merge2KeyColumns) {
    cudf::test::column_wrapper_factory<TypeParam> columnFactory;

    cudf::size_type inputRows = 50000;
    inputRows = (cudf::detail::unwrap(std::numeric_limits<TypeParam>::max()) < inputRows ? 40 : inputRows);

    auto leftColWrap1 = columnFactory.make(inputRows,
                                            [=](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return row >= inputRows / 2;
                                                else return row;
                                            });
    auto leftColWrap2 = columnFactory.make(inputRows,
                                            [=](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return (row / (inputRows / 4)) % 2 == 0;
                                                else return 2 * row;
                                            });

    auto rightColWrap1 = columnFactory.make(inputRows,
                                            [=](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return row >= inputRows / 2;
                                                else return row;
                                            });
    auto rightColWrap2 = columnFactory.make(inputRows,
                                            [=](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return (row / (inputRows / 4)) % 2 == 0;
                                                else return 2 * row + 1;
                                            });

    std::vector<cudf::size_type> sortByCols = {0, 1};
    std::vector<order_by_type> orderByTypes = {GDF_ORDER_ASC, GDF_ORDER_DESC};

    cudf::table outputTable;
    EXPECT_NO_THROW(outputTable = cudf::merge(cudf::table{leftColWrap1.get(), leftColWrap2.get()},
                                            cudf::table{rightColWrap1.get(), rightColWrap2.get()},
                                            sortByCols,
                                            orderByTypes));

    const cudf::size_type outputRows = leftColWrap1.size() + rightColWrap1.size();
    auto expectedDataWrap1 = columnFactory.make(outputRows,
                                                [=](cudf::size_type row)->cudf::size_type {
                                                    if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return row >= outputRows / 2;
                                                    else return row / 2;
                                                });
    auto expectedDataWrap2 = columnFactory.make(outputRows,
                                                [=](cudf::size_type row)->cudf::size_type {
                                                    if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return (row / (outputRows / 4)) % 2 == 0;
                                                    else return row % 2 == 0 ? row + 1 : row - 1;
                                                });

    EXPECT_TRUE(gdf_equal_columns(*expectedDataWrap1.get(), *outputTable.get_column(0)));
    EXPECT_TRUE(gdf_equal_columns(*expectedDataWrap2.get(), *outputTable.get_column(1)));
}

TYPED_TEST(MergeTest_, Merge1KeyNullColumns) {
    cudf::test::column_wrapper_factory<TypeParam> columnFactory;

    cudf::size_type inputRows = 50000;
    inputRows = (cudf::detail::unwrap(std::numeric_limits<TypeParam>::max()) < inputRows ? 40 : inputRows);

    // data: 0  2  4  6 | valid: 1 1 1 0
    auto leftColWrap1 = columnFactory.make(inputRows,
                                            [](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return 0;
                                                else return 2 * row;
                                            },
                                            [=](cudf::size_type row) { return row < inputRows - 1; });

    // data: 1  3  5  7 | valid: 1 1 1 0
    auto rightColWrap1 = columnFactory.make(inputRows,
                                            [](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return 1;
                                                else return 2 * row + 1;
                                            },
                                            [=](cudf::size_type row) { return row < inputRows - 1; });

    std::vector<cudf::size_type> sortByCols = {0};
    std::vector<order_by_type> orderByTypes = {GDF_ORDER_ASC};

    cudf::table outputTable;
    EXPECT_NO_THROW(outputTable = cudf::merge(cudf::table{leftColWrap1.get()},
                                            cudf::table{rightColWrap1.get()},
                                            sortByCols,
                                            orderByTypes));

    const cudf::size_type outputRows = leftColWrap1.size() + rightColWrap1.size();
    // data: 0 1 2 3 4 5 6 7 | valid: 1 1 1 1 1 1 0 0
    const cudf::size_type column1TotalNulls = leftColWrap1.null_count() + rightColWrap1.null_count();
    auto expectedDataWrap1 = columnFactory.make(outputRows,
                                                [=](cudf::size_type row)->cudf::size_type {
                                                    if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return row >= (outputRows - column1TotalNulls) / 2;
                                                    else return row;
                                                },
                                                [=](cudf::size_type row) { return row < (outputRows - column1TotalNulls); });

    EXPECT_TRUE(gdf_equal_columns(*expectedDataWrap1.get(), *outputTable.get_column(0)));
}

TYPED_TEST(MergeTest_, Merge2KeyNullColumns) {
    cudf::test::column_wrapper_factory<TypeParam> columnFactory;

    cudf::size_type inputRows = 50000;
    inputRows = (cudf::detail::unwrap(std::numeric_limits<TypeParam>::max()) < inputRows ? 40 : inputRows);

    // data: 0 1 2 3 | valid: 1 1 1 1
    auto leftColWrap1 = columnFactory.make(inputRows,
                                            [=](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return row >= inputRows / 2;
                                                else return row;
                                            });
    // data: 0 2 4 6 | valid: 1 1 1 1
    auto leftColWrap2 = columnFactory.make(inputRows,
                                            [=](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return (row / (inputRows / 4)) % 2 == 0;
                                                else return 2 * row;
                                            },
                                            [](cudf::size_type row) { return true; });

    // data: 0 1 2 3 | valid: 1 1 1 1
    auto rightColWrap1 = columnFactory.make(inputRows,
                                            [=](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return row >= inputRows / 2;
                                                else return row;
                                            });
    // data: 0 1 2 3 | valid: 0 0 0 0
    auto rightColWrap2 = columnFactory.make(inputRows,
                                            [=](cudf::size_type row)->cudf::size_type {
                                                if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return (row / (inputRows / 4)) % 2 == 0;
                                                else return row;
                                            },
                                            [](cudf::size_type row) { return false; });

    std::vector<cudf::size_type> sortByCols = {0, 1};
    std::vector<order_by_type> orderByTypes = {GDF_ORDER_ASC, GDF_ORDER_DESC};

    cudf::table outputTable;
    EXPECT_NO_THROW(outputTable = cudf::merge(cudf::table{leftColWrap1.get(), leftColWrap2.get()},
                                            cudf::table{rightColWrap1.get(), rightColWrap2.get()},
                                            sortByCols,
                                            orderByTypes));

    const cudf::size_type outputRows = leftColWrap1.size() + rightColWrap1.size();
    // data: 0 0 1 1 2 2 3 3 | valid: 1 1 1 1 1 1 1 1
    auto expectedDataWrap1 = columnFactory.make(outputRows,
                                                [=](cudf::size_type row)->cudf::size_type {
                                                    if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return row >= outputRows / 2;
                                                    else return row / 2;
                                                },
                                                [](cudf::size_type row) { return true; });
    // data: 0 0 2 1 4 2 6 3 | valid: 0 1 0 1 0 1 0 1
    auto expectedDataWrap2 = columnFactory.make(outputRows,
                                                [=](cudf::size_type row)->cudf::size_type {
                                                    if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return (row / (outputRows / 8)) % 2 == 0;
                                                    else return row % 2 != 0 ? 2 * (row / 2) : (row / 2);
                                                },
                                                [=](cudf::size_type row) {
                                                    if(cudf::gdf_dtype_of<TypeParam>() == GDF_BOOL8) return (row / (outputRows / 4)) % 2 == 1;
                                                    else return row % 2 != 0;
                                                });

    EXPECT_TRUE(gdf_equal_columns(*expectedDataWrap1.get(), *outputTable.get_column(0)));
    EXPECT_TRUE(gdf_equal_columns(*expectedDataWrap2.get(), *outputTable.get_column(1)));
}
*/
