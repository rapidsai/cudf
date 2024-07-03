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
    l_returnflag,
    l_linestatus,
    sum(l_quantity) as sum_qty,
    sum(l_extendedprice) as sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
    avg(l_quantity) as avg_qty,
    avg(l_extendedprice) as avg_price,
    avg(l_discount) as avg_disc,
    count(*) as count_order
from
    lineitem
where
    l_shipdate <= date '1998-09-02'
group by
    l_returnflag,
    l_linestatus
order by
    l_returnflag,
    l_linestatus;
*/

std::unique_ptr<cudf::column> calc_disc_price(std::unique_ptr<table_with_cols>& table) {
    auto one = cudf::fixed_point_scalar<numeric::decimal64>(1);
    auto discount = table->column("l_discount");
    auto one_minus_discount = cudf::binary_operation(one, discount, cudf::binary_operator::SUB, discount.type());
    auto extended_price = table->column("l_extendedprice");
    auto disc_price_type = cudf::data_type{cudf::type_id::DECIMAL64, -4};
    auto disc_price = cudf::binary_operation(extended_price, one_minus_discount->view(), cudf::binary_operator::MUL, disc_price_type);
    return disc_price;
}

std::unique_ptr<cudf::column> calc_charge(std::unique_ptr<table_with_cols>& table, std::unique_ptr<cudf::column>& disc_price) {
    auto one = cudf::fixed_point_scalar<numeric::decimal64>(1);
    auto tax = table->column("l_tax");
    auto one_plus_tax = cudf::binary_operation(one, tax, cudf::binary_operator::ADD, tax.type());
    auto charge_type = cudf::data_type{cudf::type_id::DECIMAL64, -6};
    auto charge = cudf::binary_operation(disc_price->view(), one_plus_tax->view(), cudf::binary_operator::MUL, charge_type);
    return charge;
}

int main(int argc, char const** argv) {
    auto args = parse_args(argc, argv);

    // Use a memory pool
    auto resource = create_memory_resource(args.memory_resource_type);
    rmm::mr::set_current_device_resource(resource.get());

    Timer timer;

    // 1. Read out the `lineitem` table from parquet file
    auto shipdate_ref = cudf::ast::column_reference(5);
    auto shipdate_upper = cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1998, 9, 2), true);
    auto shipdate_upper_literal = cudf::ast::literal(shipdate_upper);
    auto shipdate_pred = std::make_unique<cudf::ast::operation>(
        cudf::ast::ast_operator::LESS_EQUAL,
        shipdate_ref,
        shipdate_upper_literal
    );
    auto lineitem = read_parquet(
        args.dataset_dir + "lineitem/part-0.parquet", 
        {"l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice", "l_discount", "l_shipdate", "l_orderkey", "l_tax"},
        std::move(shipdate_pred)
    );

    // 2. Calculate the discount price and charge columns and append to lineitem table
    auto disc_price = calc_disc_price(lineitem);
    auto charge = calc_charge(lineitem, disc_price);
    auto appended_table = lineitem->append(disc_price, "disc_price")->append(charge, "charge");

    // 3. Perform the group by operation
    auto groupedby_table = apply_groupby(
        appended_table, 
        groupby_context_t {
            {"l_returnflag", "l_linestatus"},
            {
                {
                    "l_extendedprice", 
                    {
                        {cudf::aggregation::Kind::SUM, "sum_base_price"}, 
                        {cudf::aggregation::Kind::MEAN, "avg_price"}    
                    }
                },
                {
                    "l_quantity", 
                    {
                        {cudf::aggregation::Kind::SUM, "sum_qty"},
                        {cudf::aggregation::Kind::MEAN, "avg_qty"}
                    }
                },
                {
                    "l_discount", 
                    {
                        {cudf::aggregation::Kind::MEAN, "avg_disc"},
                    }
                },
                {
                    "disc_price", 
                    {
                        {cudf::aggregation::Kind::SUM, "sum_disc_price"},
                    }
                },
                {
                    "charge", 
                    {
                        {cudf::aggregation::Kind::SUM, "sum_charge"}, 
                        {cudf::aggregation::Kind::COUNT_ALL, "count_order"}
                    }
                },
            }
        }
    );

    // 4. Perform the order by operation
    auto orderedby_table = apply_orderby(groupedby_table, {"l_returnflag", "l_linestatus"}, {cudf::order::ASCENDING, cudf::order::ASCENDING});

    timer.print_elapsed_millis();
    
    // 5. Write query result to a parquet file
    orderedby_table->to_parquet("q1.parquet");
    return 0;
}
