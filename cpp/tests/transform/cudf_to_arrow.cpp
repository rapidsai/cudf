/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <arrow/util/bit_util.h>
#include <arrow/testing/gtest_util.h>
#include <tests/transform/arrow_utils.hpp>


struct CUDFToArrow : public cudf::test::BaseFixture {
};

std::unique_ptr<cudf::table> get_cudf_table(){
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.emplace_back(cudf::test::fixed_width_column_wrapper<int64_t>({1, 2, 3, 4, 5}).release());
    columns.emplace_back(cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 5, 2, 7}, {1, 0, 1, 1, 1}).release());
    columns.emplace_back(cudf::test::strings_column_wrapper({"fff", "aaa", "", "fff", "ccc"}, {1, 1, 1, 0, 1}).release());
    auto col4 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 5, 2, 7}, {1, 0, 1, 1, 1});
    columns.emplace_back(std::move(cudf::dictionary::encode(col4)));
    columns.emplace_back(cudf::test::lists_column_wrapper<int>({{1, 2}, {3, 4}, {}, {6}, {7, 8, 9}}).release());
    return std::make_unique<cudf::table>(std::move(columns));
}

std::shared_ptr<arrow::Table> get_arrow_table(){
    auto int64array = get_arrow_array<int64_t>({1, 2, 3, 4, 5});
    auto int32array = get_arrow_array<int32_t>({1, 2, 5, 2, 7}, {1, 0, 1, 1, 1});
    auto string_array = get_arrow_array<cudf::string_view>({"fff", "aaa", "", "fff", "ccc"}, {1, 1, 1, 0, 1});
    auto dict_array = get_arrow_dict_array({1, 2, 5, 7}, {0, 1, 2, 1, 7}, {1, 0, 1, 1, 1});
    auto list_array = get_arrow_list_array({1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 2, 4, 5, 6, 9}, {1, 1, 0, 1, 1});

    std::vector <std::shared_ptr<arrow::Field>> schema_vector = {arrow::field("a", int32array->type()), arrow::field("b", int64array->type()), arrow::field("c", string_array->type()), arrow::field("d", dict_array->type()), arrow::field("e", list_array->type())};

    auto schema = std::make_shared<arrow::Schema>(schema_vector);
    return arrow::Table::Make(schema, {int32array, int64array, string_array, dict_array, list_array});
}

TEST_F(CUDFToArrow, NormalTable){
    auto cudf_table = get_cudf_table();
    auto expected_arrow_table = get_arrow_table();

    auto got_arrow_table = cudf::to_arrow(cudf_table->view(), {"a", "b", "c", "d", "e"});

    expected_arrow_table->Equals(*got_arrow_table, true);
}

