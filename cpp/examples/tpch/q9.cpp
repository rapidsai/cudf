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

#include "utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>

/**
 * @file q9.cpp
 * @brief Implement query 9 of the TPC-H benchmark.
 *
 * create view part as select * from '/tables/scale-1/part.parquet';
 * create view supplier as select * from '/tables/scale-1/supplier.parquet';
 * create view lineitem as select * from '/tables/scale-1/lineitem.parquet';
 * create view partsupp as select * from '/tables/scale-1/partsupp.parquet';
 * create view orders as select * from '/tables/scale-1/orders.parquet';
 * create view nation as select * from '/tables/scale-1/nation.parquet';
 *
 * select
 *    nation,
 *    o_year,
 *    sum(amount) as sum_profit
 * from
 *     (
 *        select
 *            n_name as nation,
 *            extract(year from o_orderdate) as o_year,
 *            l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
 *        from
 *            part,
 *            supplier,
 *            lineitem,
 *            partsupp,
 *            orders,
 *            nation
 *        where
 *           s_suppkey = l_suppkey
 *           and ps_suppkey = l_suppkey
 *           and ps_partkey = l_partkey
 *           and p_partkey = l_partkey
 *           and o_orderkey = l_orderkey
 *           and s_nationkey = n_nationkey
 *           and p_name like '%green%'
 *     ) as profit
 * group by
 *     nation,
 *     o_year
 * order by
 *     nation,
 *     o_year desc;
 */

/**
 * @brief Calculate the amount column
 *
 * @param discount The discount column
 * @param extendedprice The extended price column
 * @param supplycost The supply cost column
 * @param quantity The quantity column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
std::unique_ptr<cudf::column> calc_amount(
  cudf::column_view const& discount,
  cudf::column_view const& extendedprice,
  cudf::column_view const& supplycost,
  cudf::column_view const& quantity,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  auto const one = cudf::numeric_scalar<double>(1);
  auto one_minus_discount =
    cudf::binary_operation(one, discount, cudf::binary_operator::SUB, discount.type());
  auto extendedprice_discounted_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto extendedprice_discounted      = cudf::binary_operation(extendedprice,
                                                         one_minus_discount->view(),
                                                         cudf::binary_operator::MUL,
                                                         extendedprice_discounted_type,
                                                         stream,
                                                         mr);
  auto supplycost_quantity_type      = cudf::data_type{cudf::type_id::FLOAT64};
  auto supplycost_quantity           = cudf::binary_operation(
    supplycost, quantity, cudf::binary_operator::MUL, supplycost_quantity_type);
  auto amount = cudf::binary_operation(extendedprice_discounted->view(),
                                       supplycost_quantity->view(),
                                       cudf::binary_operator::SUB,
                                       extendedprice_discounted->type(),
                                       stream,
                                       mr);
  return amount;
}

int main(int argc, char const** argv)
{
  auto args = parse_args(argc, argv);

  // Use a memory pool
  auto resource = create_memory_resource(args.memory_resource_type);
  rmm::mr::set_current_device_resource(resource.get());

  Timer timer;

  // Read out the table from parquet files
  auto lineitem = read_parquet(
    args.dataset_dir + "/lineitem.parquet",
    {"l_suppkey", "l_partkey", "l_orderkey", "l_extendedprice", "l_discount", "l_quantity"});
  auto nation   = read_parquet(args.dataset_dir + "/nation.parquet", {"n_nationkey", "n_name"});
  auto orders   = read_parquet(args.dataset_dir + "/orders.parquet", {"o_orderkey", "o_orderdate"});
  auto part     = read_parquet(args.dataset_dir + "/part.parquet", {"p_partkey", "p_name"});
  auto partsupp = read_parquet(args.dataset_dir + "/partsupp.parquet",
                               {"ps_suppkey", "ps_partkey", "ps_supplycost"});
  auto supplier =
    read_parquet(args.dataset_dir + "/supplier.parquet", {"s_suppkey", "s_nationkey"});

  // Generating the `profit` table
  // Filter the part table using `p_name like '%green%'`
  auto p_name = part->table().column(1);
  auto mask =
    cudf::strings::like(cudf::strings_column_view(p_name), cudf::string_scalar("%green%"));
  auto part_filtered = apply_mask(part, mask);

  // Perform the joins
  auto join_a = apply_inner_join(supplier, nation, {"s_nationkey"}, {"n_nationkey"});
  auto join_b = apply_inner_join(partsupp, join_a, {"ps_suppkey"}, {"s_suppkey"});
  auto join_c = apply_inner_join(lineitem, part_filtered, {"l_partkey"}, {"p_partkey"});
  auto join_d = apply_inner_join(orders, join_c, {"o_orderkey"}, {"l_orderkey"});
  auto joined_table =
    apply_inner_join(join_d, join_b, {"l_suppkey", "l_partkey"}, {"s_suppkey", "ps_partkey"});

  // Calculate the `nation`, `o_year`, and `amount` columns
  auto n_name = std::make_unique<cudf::column>(joined_table->column("n_name"));
  auto o_year = cudf::datetime::extract_year(joined_table->column("o_orderdate"));
  auto amount = calc_amount(joined_table->column("l_discount"),
                            joined_table->column("l_extendedprice"),
                            joined_table->column("ps_supplycost"),
                            joined_table->column("l_quantity"));

  // Put together the `profit` table
  std::vector<std::unique_ptr<cudf::column>> profit_columns;
  profit_columns.push_back(std::move(n_name));
  profit_columns.push_back(std::move(o_year));
  profit_columns.push_back(std::move(amount));

  auto profit_table = std::make_unique<cudf::table>(std::move(profit_columns));
  auto profit       = std::make_unique<table_with_names>(
    std::move(profit_table), std::vector<std::string>{"nation", "o_year", "amount"});

  // Perform the groupby operation
  auto groupedby_table = apply_groupby(
    profit,
    groupby_context_t{{"nation", "o_year"},
                      {{"amount", {{cudf::groupby_aggregation::SUM, "sum_profit"}}}}});

  // Perform the orderby operation
  auto orderedby_table = apply_orderby(
    groupedby_table, {"nation", "o_year"}, {cudf::order::ASCENDING, cudf::order::DESCENDING});

  timer.print_elapsed_millis();

  // Write query result to a parquet file
  orderedby_table->to_parquet("q9.parquet");
}
