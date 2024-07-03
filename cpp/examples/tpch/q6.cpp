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

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>

#include "utils.hpp"

/*
create view lineitem as select * from '~/tpch_sf1/lineitem/part-0.parquet';

select
    sum(l_extendedprice * l_discount) as revenue
from
    lineitem
where
        l_shipdate >= date '1994-01-01'
    and l_shipdate < date '1995-01-01'
    and l_discount >= 0.05 
    and l_discount <= 0.07
    and l_quantity < 24;
*/

std::unique_ptr<cudf::column> calc_revenue(std::unique_ptr<table_with_cols>& table) {
    auto extendedprice = table->column("l_extendedprice");
    auto discount = table->column("l_discount");
    auto extendedprice_mul_discount_type = cudf::data_type{cudf::type_id::DECIMAL64, -4};
    auto extendedprice_mul_discount = cudf::binary_operation(
        extendedprice,
        discount,
        cudf::binary_operator::MUL,
        extendedprice_mul_discount_type
    );
    return extendedprice_mul_discount;
}

int main(int argc, char const** argv) {
    auto args = parse_args(argc, argv);

    // Use a memory pool
    auto resource = create_memory_resource(args.use_memory_pool, args.use_managed_memory);
    rmm::mr::set_current_device_resource(resource.get());

    Timer timer;

    // 1. Read out the `lineitem` table from parquet file
    auto shipdate_ref = cudf::ast::column_reference(2);
    auto shipdate_lower = cudf::timestamp_scalar<cudf::timestamp_D>(
        days_since_epoch(1994, 1, 1), true);
    auto shipdate_lower_literal = cudf::ast::literal(shipdate_lower);
    auto shipdate_upper = cudf::timestamp_scalar<cudf::timestamp_D>(
        days_since_epoch(1995, 1, 1), true);
    auto shipdate_upper_literal = cudf::ast::literal(shipdate_upper);
    auto shipdate_pred_a = cudf::ast::operation(
        cudf::ast::ast_operator::GREATER_EQUAL,
        shipdate_ref,
        shipdate_lower_literal
    );
    auto shipdate_pred_b = cudf::ast::operation(
        cudf::ast::ast_operator::LESS,
        shipdate_ref,
        shipdate_upper_literal
    );
    auto shipdate_pred = std::make_unique<cudf::ast::operation>(
        cudf::ast::ast_operator::LOGICAL_AND,
        shipdate_pred_a,
        shipdate_pred_b
    );
    auto lineitem = read_parquet(
        args.dataset_dir + "lineitem/part-0.parquet", 
        {"l_extendedprice", "l_discount", "l_shipdate", "l_quantity"},
        std::move(shipdate_pred)
    );

    // 2. Cast the discount and quantity columns to float32 and append to lineitem table
    auto discout_float = cudf::cast(
        lineitem->column("l_discount"), cudf::data_type{cudf::type_id::FLOAT32});
    auto quantity_float = cudf::cast(
        lineitem->column("l_quantity"), cudf::data_type{cudf::type_id::FLOAT32});
    auto appended_table = lineitem
        ->append(discout_float, "l_discount_float")
        ->append(quantity_float, "l_quantity_float");
    
    // 3. Apply the filters
    auto discount_ref = cudf::ast::column_reference(
        appended_table->col_id("l_discount_float"));
    auto quantity_ref = cudf::ast::column_reference(
        appended_table->col_id("l_quantity_float")
    );

    auto discount_lower = cudf::numeric_scalar<float_t>(0.05);
    auto discount_lower_literal = cudf::ast::literal(discount_lower);
    auto discount_upper = cudf::numeric_scalar<float_t>(0.07);
    auto discount_upper_literal = cudf::ast::literal(discount_upper);
    auto quantity_upper = cudf::numeric_scalar<float_t>(24);
    auto quantity_upper_literal = cudf::ast::literal(quantity_upper);

    auto discount_pred_a = cudf::ast::operation(
        cudf::ast::ast_operator::GREATER_EQUAL,
        discount_ref,
        discount_lower_literal
    );
    
    auto discount_pred_b = cudf::ast::operation(
        cudf::ast::ast_operator::LESS_EQUAL,
        discount_ref,
        discount_upper_literal
    );
    auto discount_pred = cudf::ast::operation(
        cudf::ast::ast_operator::LOGICAL_AND, discount_pred_a, discount_pred_b
    );
    auto quantity_pred = cudf::ast::operation(
        cudf::ast::ast_operator::LESS,
        quantity_ref,
        quantity_upper_literal
    );
    auto discount_quantity_pred = cudf::ast::operation(
        cudf::ast::ast_operator::LOGICAL_AND,
        discount_pred,
        quantity_pred
    );
    auto filtered_table = apply_filter(appended_table, discount_quantity_pred);

    // 4. Calculate the `revenue` column
    auto revenue = calc_revenue(filtered_table);

    // 5. Sum the `revenue` column
    auto revenue_view = revenue->view();
    auto result_table = apply_reduction(
        revenue_view,
        cudf::aggregation::Kind::SUM,
        "revenue"
    );

    timer.print_elapsed_millis();

    // 6. Write query result to a parquet file
    result_table->to_parquet("q6.parquet");
    return 0;
}
