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
    
    //TODO: figure out solution around compiler error
    //when trying to instantiate
    //fixed_width_column_wrapper<cudf::bool8>
    //
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


/*
TYPED_TEST(MergeTest_, MismatchedColumnDypes) {
    cudf::size_type inputRows = 4;

    cudf::test::column_wrapper<int32_t> leftColWrap1(inputRows, [](cudf::size_type row) { return row; });

    cudf::test::column_wrapper<double> rightColWrap1(inputRows, [](cudf::size_type row) { return row; });

    std::vector<cudf::size_type> sortByCols = {0};
    std::vector<order_by_type> orderByTypes = {GDF_ORDER_ASC};

    EXPECT_THROW(cudf::merge(cudf::table{leftColWrap1.get()},
                            cudf::table{rightColWrap1.get()},
                            sortByCols,
                            orderByTypes), cudf::logic_error);
}

TYPED_TEST(MergeTest_, EmptyKeyColumns) {
    cudf::test::column_wrapper_factory<TypeParam> columnFactory;

    cudf::size_type inputRows = 4;

    auto leftColWrap1 = columnFactory.make(inputRows, [](cudf::size_type row) { return row; });

    auto rightColWrap1 = columnFactory.make(inputRows, [](cudf::size_type row) { return row; });

    std::vector<cudf::size_type> sortByCols;
    std::vector<order_by_type> orderByTypes = {GDF_ORDER_ASC};

    EXPECT_THROW(cudf::merge(cudf::table{leftColWrap1.get()},
                            cudf::table{rightColWrap1.get()},
                            sortByCols,
                            orderByTypes), cudf::logic_error);
}

TYPED_TEST(MergeTest_, TooManyKeyColumns) {
    cudf::test::column_wrapper_factory<TypeParam> columnFactory;

    cudf::size_type inputRows = 4;

    auto leftColWrap1 = columnFactory.make(inputRows, [](cudf::size_type row) { return row; });

    auto rightColWrap1 = columnFactory.make(inputRows, [](cudf::size_type row) { return row; });

    std::vector<cudf::size_type> sortByCols = {0, 1};
    std::vector<order_by_type> orderByTypes = {GDF_ORDER_ASC};

    EXPECT_THROW(cudf::merge(cudf::table{leftColWrap1.get()},
                            cudf::table{rightColWrap1.get()},
                            sortByCols,
                            orderByTypes), cudf::logic_error);
}

TYPED_TEST(MergeTest_, EmptyOrderTypes) {
    cudf::test::column_wrapper_factory<TypeParam> columnFactory;

    cudf::size_type inputRows = 4;

    auto leftColWrap1 = columnFactory.make(inputRows, [](cudf::size_type row) { return row; });

    auto rightColWrap1 = columnFactory.make(inputRows, [](cudf::size_type row) { return row; });

    std::vector<cudf::size_type> sortByCols = {0};
    std::vector<order_by_type> orderByTypes;

    EXPECT_THROW(cudf::merge(cudf::table{leftColWrap1.get()},
                            cudf::table{rightColWrap1.get()},
                            sortByCols,
                            orderByTypes), cudf::logic_error);
}

TYPED_TEST(MergeTest_, TooManyOrderTypes) {
    cudf::test::column_wrapper_factory<TypeParam> columnFactory;

    cudf::size_type inputRows = 4;

    auto leftColWrap1 = columnFactory.make(inputRows, [](cudf::size_type row) { return row; });

    auto rightColWrap1 = columnFactory.make(inputRows, [](cudf::size_type row) { return row; });

    std::vector<cudf::size_type> sortByCols = {0};
    std::vector<order_by_type> orderByTypes = {GDF_ORDER_ASC, GDF_ORDER_DESC};

    EXPECT_THROW(cudf::merge(cudf::table{leftColWrap1.get()},
                            cudf::table{rightColWrap1.get()},
                            sortByCols,
                            orderByTypes), cudf::logic_error);
}

TYPED_TEST(MergeTest_, MismatchedKeyColumnsAndOrderTypes) {
    cudf::test::column_wrapper_factory<TypeParam> columnFactory;

    cudf::size_type inputRows = 4;

    auto leftColWrap1 = columnFactory.make(inputRows, [](cudf::size_type row) { return row; });
    auto leftColWrap2 = columnFactory.make(inputRows, [](cudf::size_type row) { return row; });

    auto rightColWrap1 = columnFactory.make(inputRows, [](cudf::size_type row) { return row; });
    auto rightColWrap2 = columnFactory.make(inputRows, [](cudf::size_type row) { return row; });

    std::vector<cudf::size_type> sortByCols = {0, 1};
    std::vector<order_by_type> orderByTypes = {GDF_ORDER_ASC};

    EXPECT_THROW(cudf::merge(cudf::table{leftColWrap1.get(), leftColWrap2.get()},
                            cudf::table{rightColWrap1.get(), rightColWrap2.get()},
                            sortByCols,
                            orderByTypes), cudf::logic_error);
}

TYPED_TEST(MergeTest_, MergeWithEmptyColumn) {
    cudf::test::column_wrapper_factory<TypeParam> columnFactory;

    cudf::size_type inputRows = 50000;
    inputRows = (cudf::detail::unwrap(std::numeric_limits<TypeParam>::max()) < inputRows ? 40 : inputRows);

    auto leftColWrap1 = columnFactory.make(inputRows, [](cudf::size_type row) { return row; });

    auto rightColWrap1 = columnFactory.make(0, [](cudf::size_type row) { return 0; });

    std::vector<cudf::size_type> sortByCols = {0};
    std::vector<order_by_type> orderByTypes = {GDF_ORDER_ASC};

    cudf::table outputTable;
    EXPECT_NO_THROW(outputTable = cudf::merge(cudf::table{leftColWrap1.get()},
                                            cudf::table{rightColWrap1.get()},
                                            sortByCols,
                                            orderByTypes));
    
    const cudf::size_type outputRows = leftColWrap1.size() + rightColWrap1.size();
    auto expectedDataWrap1 = columnFactory.make(outputRows, [](cudf::size_type row) { return row; });

    EXPECT_TRUE(gdf_equal_columns(*expectedDataWrap1.get(), *outputTable.get_column(0)));
}

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
