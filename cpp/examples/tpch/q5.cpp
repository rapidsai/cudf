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
#include <chrono>
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


int main() {
    std::string lineitem_path = "~/tpch_sf1/lineitem/part-0.parquet";
    std::string orders_path = "~/tpch_sf1/orders/part-0.parquet";
    std::string customer_path = "~/tpch_sf1/customer/part-0.parquet";
    std::string supplier_path = "~/tpch_sf1/supplier/part-0.parquet";
    std::string nation_path = "~/tpch_sf1/nation/part-0.parquet";
    std::string region_path = "~/tpch_sf1/region/part-0.parquet";

    auto lineitem = read_parquet(lineitem_path);
    auto orders = read_parquet(orders_path);
    auto customer = read_parquet(customer_path);
    auto supplier = read_parquet(supplier_path);
    auto nation = read_parquet(nation_path);
    auto region = read_parquet(region_path);
}