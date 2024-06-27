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
#include <memory>
#include <vector>
#include <cudf/io/parquet.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/groupby.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/sorting.hpp>
#include <cudf/reduction.hpp>
#include <cudf/unary.hpp>
#include <cudf/stream_compaction.hpp>

#include "utils.hpp"

/*
select
    sum(l_extendedprice * l_discount) as revenue
from
    '~/tpch_sf1/lineitem/part-0.parquet'
where
        l_shipdate >= date '1994-01-01'
    and l_shipdate < date '1995-01-01'
    and l_discount >= 0.05 
    and l_discount <= 0.07
    and l_quantity < 24;
*/

std::unique_ptr<cudf::table> scan_filter_project() {
    std::string lineitem = "/home/jayjeetc/tpch_sf1/lineitem/part-0.parquet";
    auto source = cudf::io::source_info(lineitem);
    auto builder = cudf::io::parquet_reader_options_builder(source);

    std::vector<std::string> projection_cols = {
        "l_extendedprice",
        "l_discount",
        "l_shipdate",
        "l_quantity" 
    };

    auto extendedprice = cudf::ast::column_reference(0);
    auto discount = cudf::ast::column_reference(1);
    auto shipdate = cudf::ast::column_reference(2);
    auto quantity = cudf::ast::column_reference(3);

    auto date_scalar_a = cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1994, 1, 1), true);
    auto date_literal_a = cudf::ast::literal(date_scalar_a);

    auto date_scalar_b = cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 1, 1), true);
    auto date_literal_b = cudf::ast::literal(date_scalar_b);

    auto pred_a = cudf::ast::operation(
        cudf::ast::ast_operator::GREATER_EQUAL,
        shipdate,
        date_literal_a
    );

    auto pred_b = cudf::ast::operation(
        cudf::ast::ast_operator::LESS,
        shipdate,
        date_literal_b
    );

    auto pred_ab = cudf::ast::operation(
        cudf::ast::ast_operator::LOGICAL_AND,
        pred_a,
        pred_b
    );

    builder.columns(projection_cols);

    // FIXME: since, ast does not support `fixed_point_scalar` yet,
    // we just push down the date filters while scanning the parquet file.
    builder.filter(pred_ab);

    auto options = builder.build();
    auto result = cudf::io::read_parquet(options);
    return std::move(result.tbl);
}

std::unique_ptr<cudf::table> apply_filters(std::unique_ptr<cudf::table>& table) {
    // NOTE: apply the remaining filters based on the float32 casted columns
    auto l_discount = cudf::ast::column_reference(4);
    auto l_quantity = cudf::ast::column_reference(5);

    auto l_discount_lower = cudf::numeric_scalar<float_t>(0.05);
    auto l_discount_lower_literal = cudf::ast::literal(l_discount_lower);
    auto l_discount_upper = cudf::numeric_scalar<float_t>(0.07);
    auto l_discount_upper_literal = cudf::ast::literal(l_discount_upper);
    auto l_quantity_upper = cudf::numeric_scalar<float_t>(24);
    auto l_quantity_upper_literal = cudf::ast::literal(l_quantity_upper);

    auto l_discount_pred_a = cudf::ast::operation(
            cudf::ast::ast_operator::GREATER_EQUAL,
            l_discount,
            l_discount_lower_literal
        );
    
    auto l_discount_pred_b = cudf::ast::operation(
            cudf::ast::ast_operator::LESS_EQUAL,
            l_discount,
            l_discount_upper_literal
        );

    auto l_discount_pred = cudf::ast::operation(
        cudf::ast::ast_operator::LOGICAL_AND, l_discount_pred_a, l_discount_pred_b
    );

    auto l_quantity_pred = cudf::ast::operation(
        cudf::ast::ast_operator::LESS,
        l_quantity,
        l_quantity_upper_literal
    );

    auto pred = cudf::ast::operation(
        cudf::ast::ast_operator::LOGICAL_AND,
        l_discount_pred,
        l_quantity_pred
    );

    auto boolean_mask = cudf::compute_column(table->view(), pred);
    return cudf::apply_boolean_mask(table->view(), boolean_mask->view());
}

std::unique_ptr<cudf::table> apply_reduction(std::unique_ptr<cudf::table>& table) {
    auto extendedprice = table->view().column(0);
    auto discount = table->view().column(1);

    auto extendedprice_mul_discount_type = cudf::data_type{cudf::type_id::DECIMAL64, -4};
    auto extendedprice_mul_discount = cudf::binary_operation(
        extendedprice,
        discount,
        cudf::binary_operator::MUL,
        extendedprice_mul_discount_type
    );
    
    auto const sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    auto sum = cudf::reduce(extendedprice_mul_discount->view(), *sum_agg, extendedprice_mul_discount->type());

    cudf::size_type len = 1;
    auto col = cudf::make_column_from_scalar(*sum, len);

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(col));
    auto result_table = std::make_unique<cudf::table>(std::move(columns));
    return result_table;
}

int main() {
    auto t1 = scan_filter_project();
    auto discout_float = cudf::cast(t1->view().column(1), cudf::data_type{cudf::type_id::FLOAT32});
    auto quantity_float = cudf::cast(t1->view().column(3), cudf::data_type{cudf::type_id::FLOAT32});
    auto t2 = append_col_to_table(t1, discout_float);
    auto t3 = append_col_to_table(t2, quantity_float);
    auto t4 = apply_filters(t3);
    auto t5 = apply_reduction(t4);
    debug_table(std::move(t5), "q6.parquet");
    return 0;
}
