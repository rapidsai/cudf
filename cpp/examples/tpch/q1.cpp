/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <iostream>
#include <vector>
#include <cudf/io/parquet.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

void read_parquet_file(std::vector<std::string> column_names) {
    std::string path = "/home/jayjeetc/tpch_sf1/lineitem/part-0.parquet";
    auto source = cudf::io::source_info(path);
    auto builder = cudf::io::parquet_reader_options_builder(source);

    auto col_ref = cudf::ast::column_reference(3);

    auto literal_value = cudf::numeric_scalar<int32_t>(2);
    auto literal = cudf::ast::literal(literal_value);

    auto filter_expr = cudf::ast::operation(
        cudf::ast::ast_operator::LESS, 
        col_ref,
        literal
    );

    builder.columns(column_names);
    builder.filter(filter_expr);

    auto options = builder.build();
    cudf::io::table_with_metadata result = cudf::io::read_parquet(options);

    std::cout << result.tbl->num_columns() << std::endl;
    std::cout << result.tbl->num_rows() << std::endl;
}

int main() {
    std::vector<std::string> column_names = {"l_orderkey", "l_partkey", "l_suppkey", "l_linenumber"};
    read_parquet_file(column_names);
    return 0;
}
