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
#include <string>
#include <vector>

auto const schema_orders = std::vector<std::string>{"o_orderkey",
                                                    "o_custkey",
                                                    "o_orderdate",
                                                    "o_orderpriority",
                                                    "o_clerk",
                                                    "o_shippriority",
                                                    "o_comment"};

auto const schema_lineitem = std::vector<std::string>{"l_partkey",
                                                      "l_suppkey",
                                                      "l_linenumber",
                                                      "l_quantity",
                                                      "l_discount",
                                                      "l_tax",
                                                      "l_shipdate",
                                                      "l_commitdate",
                                                      "l_receiptdate",
                                                      "l_returnflag",
                                                      "l_linestatus",
                                                      "l_shipinstruct",
                                                      "l_shipmode",
                                                      "l_comment"};

auto const schema_customer = std::vector<std::string>{"c_custkey",
                                                      "c_name",
                                                      "c_address",
                                                      "c_nationkey",
                                                      "c_phone",
                                                      "c_acctbal",
                                                      "c_mktsegment",
                                                      "c_comment"};

auto const schema_region = std::vector<std::string>{"r_regionkey", "r_name", "r_comment"};

auto const schema_supplier = std::vector<std::string>{
  "s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"};

auto const schema_nation =
  std::vector<std::string>{"n_nationkey", "n_name", "n_regionkey", "n_comment"};

auto const schema_part = std::vector<std::string>{"p_partkey",
                                                  "p_name",
                                                  "p_mfgr",
                                                  "p_brand",
                                                  "p_type",
                                                  "p_size",
                                                  "p_container",
                                                  "p_retailprice",
                                                  "p_comment"};

auto const schema_partsupp = std::vector<std::string>{
  "ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"};
