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
create view customer as select * from '~/tpch_sf1/customer/part-0.parquet';
create view orders as select * from '~/tpch_sf1/orders/part-0.parquet';
create view lineitem as select * from '~/tpch_sf1/lineitem/part-0.parquet';
create view supplier as select * from '~/tpch_sf1/supplier/part-0.parquet';
create view nation as select * from '~/tpch_sf1/nation/part-0.parquet';
create view region as select * from '~/tpch_sf1/region/part-0.parquet';

select
    n_name,
    sum(l_extendedprice * (1 - l_discount)) as revenue
from
    customer,
    orders,
    lineitem,
    supplier,
    nation,
    region
where
        c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and l_suppkey = s_suppkey
  and c_nationkey = s_nationkey
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'ASIA'
  and o_orderdate >= date '1994-01-01'
  and o_orderdate < date '1995-01-01'
group by
    n_name
order by
    revenue desc;
*/

std::unique_ptr<cudf::column> calc_revenue(std::unique_ptr<table_with_cols>& table) {
    auto one = cudf::fixed_point_scalar<numeric::decimal64>(1, -2);
    auto disc = table->column("l_discount");
    auto one_minus_disc = cudf::binary_operation(one, disc, cudf::binary_operator::SUB, disc.type());
    auto extended_price = table->column("l_extendedprice");

    auto disc_price_type = cudf::data_type{cudf::type_id::DECIMAL64, -4};
    auto disc_price = cudf::binary_operation(extended_price, one_minus_disc->view(), cudf::binary_operator::MUL, disc_price_type);
    return disc_price;
}

int main() {
    std::string dataset_dir = BASE_DATASET_DIR;
    
    // 1. Read out the tables from parquet files
    auto customer = read_parquet(dataset_dir + "customer/part-0.parquet", {"c_custkey", "c_nationkey"});
    auto orders = read_parquet(dataset_dir + "orders/part-0.parquet", {"o_custkey", "o_orderkey", "o_orderdate"});
    auto lineitem = read_parquet(dataset_dir + "lineitem/part-0.parquet", {"l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"});
    auto supplier = read_parquet(dataset_dir + "supplier/part-0.parquet", {"s_suppkey", "s_nationkey"});
    auto nation = read_parquet(dataset_dir + "nation/part-0.parquet", {"n_nationkey", "n_regionkey", "n_name"});
    auto region = read_parquet(dataset_dir + "region/part-0.parquet", {"r_regionkey", "r_name"});

    // 2. Perform the joins
    auto join_a = apply_inner_join(
        region,
        nation,
        {"r_regionkey"},
        {"n_regionkey"}
    );
    auto join_b = apply_inner_join(
        join_a,
        customer,
        {"n_nationkey"},
        {"c_nationkey"}
    );
    auto join_c = apply_inner_join(
        join_b,
        orders,
        {"c_custkey"},
        {"o_custkey"}
    );
    auto join_d = apply_inner_join(
        join_c,
        lineitem,
        {"o_orderkey"},
        {"l_orderkey"}
    );
    auto joined_table = apply_inner_join(
        supplier,
        join_d,
        {"s_suppkey", "s_nationkey"},
        {"l_suppkey", "n_nationkey"}
    );

    // 3. Apply the filter predicates
    auto o_orderdate_ref = cudf::ast::column_reference(joined_table->col_id("o_orderdate"));
    
    auto o_orderdate_lower = cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1994, 1, 1), true);
    auto o_orderdate_lower_limit = cudf::ast::literal(o_orderdate_lower);
    auto o_orderdate_pred_a = cudf::ast::operation(
        cudf::ast::ast_operator::GREATER_EQUAL,
        o_orderdate_ref,
        o_orderdate_lower_limit
    );
    
    auto o_orderdate_upper = cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 1, 1), true);
    auto o_orderdate_upper_limit = cudf::ast::literal(o_orderdate_upper);
    auto o_orderdate_pred_b = cudf::ast::operation(
        cudf::ast::ast_operator::LESS,
        o_orderdate_ref,
        o_orderdate_upper_limit
    );
    
    auto r_name_ref = cudf::ast::column_reference(joined_table->col_id("r_name")); 
    auto r_name_value = cudf::string_scalar("ASIA");
    auto r_name_literal = cudf::ast::literal(r_name_value);
    auto r_name_pred = cudf::ast::operation(
        cudf::ast::ast_operator::EQUAL,
        r_name_ref,
        r_name_literal
    );

    auto o_orderdate_pred = cudf::ast::operation(
        cudf::ast::ast_operator::LOGICAL_AND,
        o_orderdate_pred_a,
        o_orderdate_pred_b
    );

    auto final_pred = cudf::ast::operation(
        cudf::ast::ast_operator::LOGICAL_AND,
        o_orderdate_pred,
        r_name_pred
    );
    auto filtered_table = apply_filter(joined_table, final_pred);

    // 4. Calcute and append the `revenue` column
    auto revenue = calc_revenue(filtered_table);
    auto appended_table = filtered_table->append(revenue, "revenue");

    // 5. Perform groupby and orderby operations
    groupby_context ctx{
        {"n_name"}, 
        {
            {"revenue", {{cudf::aggregation::Kind::SUM, "revenue"}}},
        }
    };
    auto groupedby_table = apply_groupby(appended_table, ctx);
    auto orderedby_table = apply_orderby(
        groupedby_table, {"revenue"}, {cudf::order::DESCENDING});
    
    // 6. Write query result to a parquet file
    orderedby_table->to_parquet("q5.parquet");
    return 0;
}
