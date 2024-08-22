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

#pragma once

#include <cudf/table/table.hpp>

#include <array>

namespace CUDF_EXPORT cudf {
namespace datagen {
namespace schema {

const std::vector<std::string> ORDERS   = {"o_orderkey",
                                           "o_custkey",
                                           "o_orderdate",
                                           "o_orderpriority",
                                           "o_clerk",
                                           "o_shippriority",
                                           "o_comment",
                                           "o_totalprice",
                                           "o_orderstatus"};
const std::vector<std::string> LINEITEM = {"l_orderkey",
                                           "l_partkey",
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
                                           "l_comment",
                                           "l_extendedprice"};
const std::vector<std::string> PART     = {"p_partkey",
                                           "p_name",
                                           "p_mfgr",
                                           "p_brand",
                                           "p_type",
                                           "p_size",
                                           "p_container",
                                           "p_retailprice",
                                           "p_comment"};
const std::vector<std::string> PARTSUPP = {
  "ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"};
const std::vector<std::string> SUPPLIER = {
  "s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"};
const std::vector<std::string> CUSTOMER = {"c_custkey",
                                           "c_name",
                                           "c_address",
                                           "c_nationkey",
                                           "c_phone",
                                           "c_acctbal",
                                           "c_mktsegment",
                                           "c_comment"};
const std::vector<std::string> NATION   = {"n_nationkey", "n_name", "n_regionkey", "n_comment"};
const std::vector<std::string> REGION   = {"r_regionkey", "r_name", "r_comment"};

}  // namespace schema

/**
 * @brief Generate the `orders`, `lineitem`, and `part` tables
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
generate_orders_lineitem_part(
  double scale_factor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate the `partsupp` table
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_partsupp(
  double scale_factor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate the `supplier` table
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_supplier(
  double scale_factor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate the `customer` table
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_customer(
  double scale_factor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate the `nation` table
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_nation(
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate the `region` table
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_region(
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

}  // namespace datagen
}  // namespace CUDF_EXPORT cudf
