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

#include "tpch_datagen.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/parquet.hpp>

// namespace {
// const std::vector<std::string> ORDERS   = {"o_orderkey",
//                                            "o_custkey",
//                                            "o_orderdate",
//                                            "o_orderpriority",
//                                            "o_clerk",
//                                            "o_shippriority",
//                                            "o_comment",
//                                            "o_totalprice",
//                                            "o_orderstatus"};
// const std::vector<std::string> LINEITEM = {"l_orderkey",
//                                            "l_partkey",
//                                            "l_suppkey",
//                                            "l_linenumber",
//                                            "l_quantity",
//                                            "l_discount",
//                                            "l_tax",
//                                            "l_shipdate",
//                                            "l_commitdate",
//                                            "l_receiptdate",
//                                            "l_returnflag",
//                                            "l_linestatus",
//                                            "l_shipinstruct",
//                                            "l_shipmode",
//                                            "l_comment",
//                                            "l_extendedprice"};
// const std::vector<std::string> PART     = {"p_partkey",
//                                            "p_name",
//                                            "p_mfgr",
//                                            "p_brand",
//                                            "p_type",
//                                            "p_size",
//                                            "p_container",
//                                            "p_retailprice",
//                                            "p_comment"};
// const std::vector<std::string> PARTSUPP = {
//   "ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"};
// const std::vector<std::string> SUPPLIER = {
//   "s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"};
// const std::vector<std::string> CUSTOMER = {"c_custkey",
//                                            "c_name",
//                                            "c_address",
//                                            "c_nationkey",
//                                            "c_phone",
//                                            "c_acctbal",
//                                            "c_mktsegment",
//                                            "c_comment"};
// const std::vector<std::string> NATION   = {"n_nationkey", "n_name", "n_regionkey", "n_comment"};
// const std::vector<std::string> REGION   = {"r_regionkey", "r_name", "r_comment"};

// }  // namespace

// /**
//  * @brief Write a `cudf::table` to a parquet file
//  *
//  * @param table The cudf::table to write
//  * @param path The path to write the parquet file to
//  * @param col_names The names of the columns in the table
//  */
// void write_parquet(std::unique_ptr<cudf::table> table,
//                    std::string const& path,
//                    std::vector<std::string> const& col_names)
// {
//   CUDF_FUNC_RANGE();
//   cudf::io::table_metadata metadata;
//   std::vector<cudf::io::column_name_info> col_name_infos;
//   for (auto& col_name : col_names) {
//     col_name_infos.push_back(cudf::io::column_name_info(col_name));
//   }
//   metadata.schema_info            = col_name_infos;
//   auto const table_input_metadata = cudf::io::table_input_metadata{metadata};
//   auto builder = cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(path));
//   builder.metadata(table_input_metadata);
//   auto const options = builder.build();
//   cudf::io::parquet_chunked_writer(options).write(table->view());
// }

void use_arr(cudf::host_span<const char* const> s) { std::cout << s[1]; }

int main(int argc, char** argv)
{
  constexpr std::array a{"hello", "world", "nvidia"};
  use_arr(cudf::host_span<const char* const>(a.data(), a.size()));
}
// if (argc < 2) {
//   std::cerr << "Usage: " << argv[0] << " [scale_factor]" << std::endl;
//   return 1;
// }

// double scale_factor = std::atof(argv[1]);
// std::cout << "Generating scale factor: " << scale_factor << std::endl;

// auto [orders, lineitem, part] = cudf::datagen::generate_orders_lineitem_part(
//   scale_factor, cudf::get_default_stream(), rmm::mr::get_current_device_resource());
// write_parquet(std::move(orders), "orders.parquet", ORDERS);
// write_parquet(std::move(lineitem), "lineitem.parquet", LINEITEM);
// write_parquet(std::move(part), "part.parquet", PART);

// auto partsupp = cudf::datagen::generate_partsupp(
//   scale_factor, cudf::get_default_stream(), rmm::mr::get_current_device_resource());
// write_parquet(std::move(partsupp), "partsupp.parquet", PARTSUPP);

// auto supplier = cudf::datagen::generate_supplier(
//   scale_factor, cudf::get_default_stream(), rmm::mr::get_current_device_resource());
// write_parquet(std::move(supplier), "supplier.parquet", SUPPLIER);

// auto customer = cudf::datagen::generate_customer(
//   scale_factor, cudf::get_default_stream(), rmm::mr::get_current_device_resource());
// write_parquet(std::move(customer), "customer.parquet", CUSTOMER);

// auto nation = cudf::datagen::generate_nation(cudf::get_default_stream(),
//                                              rmm::mr::get_current_device_resource());
// write_parquet(std::move(nation), "nation.parquet", NATION);

// auto region = cudf::datagen::generate_region(cudf::get_default_stream(),
//                                              rmm::mr::get_current_device_resource());
// write_parquet(std::move(region), "region.parquet", REGION);
