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
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
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
#include <tests/utilities/table_utilities.hpp>
#include <gtest/gtest.h>

struct ArrowToCUDFTest : public cudf::test::BaseFixture {
};

TEST_F(ArrowToCUDFTest, NormalTable){
    auto expected_cudf_table = get_cudf_table();
    auto arrow_table = get_arrow_table();

    auto got_cudf_table = cudf::arrow_to_table(arrow_table);

    cudf::test::expect_tables_equal(expected_cudf_table->view(), got_cudf_table->view());
}

TEST_F(ArrowToCUDFTest, DateTimeTable){
    auto data = {1, 2, 3, 4, 5, 6};

    auto col1 = cudf::test::fixed_width_column_wrapper<cudf::timestamp_D> (data);
    auto col2 = cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms> (data);

    cudf::table_view expected_table_view({col1, col2});

    std::shared_ptr<arrow::Array> arr1;
    arrow::Date32Builder d32builder;
    d32builder.AppendValues(std::vector<int32_t>(data));
    CUDF_EXPECTS(d32builder.Finish(&arr1).ok(), "Failed to build array");
    std::shared_ptr<arrow::Array> arr2;
    arrow::Date64Builder d64builder;
    d64builder.AppendValues(std::vector<int64_t>{1, 2, 3, 4, 5, 6});
    CUDF_EXPECTS(d64builder.Finish(&arr2).ok(), "Failed to build array");

    std::vector <std::shared_ptr<arrow::Field>> schema_vector ({arrow::field("a", arr1->type()), arrow::field("b", arr2->type())});
    auto schema = std::make_shared<arrow::Schema>(schema_vector);

    auto arrow_table = arrow::Table::Make(schema, {arr1, arr2});

    auto got_cudf_table = cudf::arrow_to_table(arrow_table);
    
    cudf::test::expect_tables_equal(expected_table_view, got_cudf_table->view());

}

TEST_F(ArrowToCUDFTest, NestedList){
    auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i == 2 ? false : true; });
    auto col = cudf::test::lists_column_wrapper<int64_t>({{{1, 2}, {3, 4}, {5}}, {{6}, {7, 8, 9}}});
    cudf::table_view expected_table_view({col});

    auto list_arr = get_arrow_list_array<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 2, 4, 5, 6, 9});
    std::vector<int32_t> offset {0, 3, 5};
    auto nested_list_arr = std::make_shared<arrow::ListArray>(arrow::list(list(arrow::int64())), offset.size()-1, arrow::Buffer::Wrap(offset), list_arr);

    std::vector <std::shared_ptr<arrow::Field>> schema_vector({arrow::field("a", nested_list_arr->type())});
    auto schema = std::make_shared<arrow::Schema>(schema_vector);

    auto arrow_table = arrow::Table::Make(schema, {nested_list_arr});

    auto got_cudf_table = cudf::arrow_to_table(arrow_table);
    cudf::test::expect_tables_equal(expected_table_view, got_cudf_table->view());
}

TEST_F(ArrowToCUDFTest, DictionaryIndicesType){
    auto array1 = get_arrow_dict_array<int64_t, int8_t>({1, 2, 5, 7}, {0, 1, 2, 1, 3}, {1, 0, 1, 1, 1});
    auto array2 = get_arrow_dict_array<int64_t, int16_t>({1, 2, 5, 7}, {0, 1, 2, 1, 3}, {1, 0, 1, 1, 1});

    std::vector <std::shared_ptr<arrow::Field>> schema_vector({arrow::field("a", array1->type()), arrow::field("b", array2->type())});
    auto schema = std::make_shared<arrow::Schema>(schema_vector);

    auto arrow_table = arrow::Table::Make(schema, {array1, array2});

    std::vector<std::unique_ptr<cudf::column>> columns;
    auto col = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 5, 2, 7}, {1, 0, 1, 1, 1});
    columns.emplace_back(std::move(cudf::dictionary::encode(col)));
    columns.emplace_back(std::move(cudf::dictionary::encode(col)));

    cudf::table expected_table(std::move(columns));

    auto got_cudf_table = cudf::arrow_to_table(arrow_table);

    cudf::test::expect_tables_equal(expected_table.view(), got_cudf_table->view());
}


struct ArrowToCUDFTestSlice : public ArrowToCUDFTest,
                              public ::testing::WithParamInterface<
                                  std::tuple<cudf::size_type, cudf::size_type>> {
};

TEST_P(ArrowToCUDFTestSlice, SliceTest){
    auto cudf_table = get_cudf_table();
    auto arrow_table = get_arrow_table();
    auto start = std::get<0>(GetParam());
    auto end = std::get<1>(GetParam());

    auto sliced_cudf_table = cudf::slice(cudf_table->view(), {start, end})[0];
    cudf::table expected_cudf_table{sliced_cudf_table};
    auto sliced_arrow_table = arrow_table->Slice(start, end-start);
    auto got_cudf_table = cudf::arrow_to_table(sliced_arrow_table);

    cudf::test::expect_tables_equal(expected_cudf_table.view(), got_cudf_table->view());
}

INSTANTIATE_TEST_CASE_P(
  ArrowToCUDFTest,
  ArrowToCUDFTestSlice,
  ::testing::Values(std::make_tuple(0, 0), std::make_tuple(0, 2), std::make_tuple(4, 4)));
