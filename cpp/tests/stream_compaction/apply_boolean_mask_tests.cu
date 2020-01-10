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

#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/table_utilities.hpp>

struct ApplyBooleanMask : public cudf::test::BaseFixture {};

TEST_F(ApplyBooleanMask, NonNullBooleanMask) {
    cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false}, {1, 1, 0, 1, 1, 0}};
    cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
    cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
    cudf::table_view input {{col1, col2, col3}};
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> boolean_mask{{true, false, true, false, true, false}};
    cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{true, true, true}, {1, 0, 1}};
    cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{10, 70, 2}, {1, 0, 1}};
    cudf::test::fixed_width_column_wrapper<double> col3_expected{{10, 70, 2}, {1, 0, 1}};
    cudf::table_view expected {{col1_expected, col2_expected, col3_expected}};

    auto got = cudf::experimental::apply_boolean_mask(input, boolean_mask);

    cudf::test::expect_tables_equal(expected, got->view());
}

TEST_F(ApplyBooleanMask, NullBooleanMask) {
    cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false}, {1, 1, 0, 1, 1, 0}};
    cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
    cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
    cudf::table_view input {{col1, col2, col3}};
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> boolean_mask{{true, false, true, false, true, false}, {0, 1, 1, 1, 1, 1}};
    cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{true, true}, {0, 1}};
    cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{70, 2}, {0, 1}};
    cudf::test::fixed_width_column_wrapper<double> col3_expected{{70, 2}, {0, 1}};
    cudf::table_view expected {{col1_expected, col2_expected, col3_expected}};

    auto got = cudf::experimental::apply_boolean_mask(input, boolean_mask);

    cudf::test::expect_tables_equal(expected, got->view());
}

TEST_F(ApplyBooleanMask, EmptyMask) {
    cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false}, {1, 1, 0, 1, 1, 0}};
    cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
    cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
    cudf::table_view input {{col1, col2, col3}};
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> boolean_mask{std::initializer_list<cudf::experimental::bool8>{}};
    cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{std::initializer_list<int16_t>{}};
    cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{std::initializer_list<int32_t>{}};
    cudf::test::fixed_width_column_wrapper<double> col3_expected{std::initializer_list<double>{}};
    cudf::table_view expected {{col1_expected, col2_expected, col3_expected}};

    auto got = cudf::experimental::apply_boolean_mask(input, boolean_mask);

    cudf::test::expect_tables_equal(expected, got->view());
}

TEST_F(ApplyBooleanMask, WrongMaskType) {
    cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false}, {1, 1, 0, 1, 1, 0}};
    cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
    cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
    cudf::table_view input {{col1, col2, col3}};
    cudf::test::fixed_width_column_wrapper<int16_t> boolean_mask{{true, false, true, false, true, false}};

    EXPECT_THROW(cudf::experimental::apply_boolean_mask(input, boolean_mask), cudf::logic_error);
}

TEST_F(ApplyBooleanMask, MaskAndInputSizeMismatch) {
    cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false}, {1, 1, 0, 1, 1, 0}};
    cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
    cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
    cudf::table_view input {{col1, col2, col3}};
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> boolean_mask{{true, false, true, false, true}};

    EXPECT_THROW(cudf::experimental::apply_boolean_mask(input, boolean_mask), cudf::logic_error);
}

TEST_F(ApplyBooleanMask, StringColumnTest) {
    cudf::test::strings_column_wrapper col1{{"This", "is", "the", "a", "k12", "string", "table", "column"}, {1, 1, 1, 1, 1, 0, 1, 1}};
    cudf::table_view input {{col1}};
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> boolean_mask{{true, true, true, true, false, true, false, true}, {1, 1, 0, 1, 1, 1, 1, 1}};
    cudf::test::strings_column_wrapper col1_expected{{"This", "is", "a", "string", "column"}, {1, 1, 1, 0, 1}};
    cudf::table_view expected {{col1_expected}};

    auto got = cudf::experimental::apply_boolean_mask(input, boolean_mask);

    cudf::test::expect_tables_equal(expected, got->view());
}

TEST_F(ApplyBooleanMask, withoutNullString)
{
    cudf::test::strings_column_wrapper col1({"d", "e", "a", "d", "k", "d", "l"});
    cudf::table_view cudf_table_in_view {{col1}};

    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> bool_filter{{1, 1, 0, 0, 1, 0, 0}};
    cudf::column_view bool_filter_col(bool_filter);

    std::unique_ptr<cudf::experimental::table> filteredTable = cudf::experimental::apply_boolean_mask(cudf_table_in_view,bool_filter_col);
    cudf::table_view tableView = filteredTable->view();

    cudf::test::strings_column_wrapper expect_col1({"d", "e", "k"});
    cudf::table_view expect_cudf_table_view {{expect_col1}};

    cudf::test::expect_tables_equal(expect_cudf_table_view, tableView);
}

TEST_F(ApplyBooleanMask, NoNullInput) {
  cudf::test::fixed_width_column_wrapper<int> col(                        {9668,  9590, 9526,  9205,  9434, 9347,  9160, 9569,  9143, 9807,  9606,  9446, 9279,  9822, 9691});
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> mask({false, false, true, false, false, true, false, true, false, true, false, false, true, false, true});
  cudf::table_view input({col});
  cudf::test::fixed_width_column_wrapper<int> col_expected({9526,9347,9569,9807,9279,9691});
  cudf::table_view expected({col_expected});
  auto got = cudf::experimental::apply_boolean_mask(input, mask);
  cudf::test::expect_tables_equal(expected, got->view());
}
