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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <vector>

/**
 * @brief Add a column of days to a column of timestamp_days
 *
 * @param timestamp_days The column of timestamp_days
 * @param days The column of days to add
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> add_calendrical_days(
  cudf::column_view const& timestamp_days,
  cudf::column_view const& days,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Perform a left join operation between two tables
 *
 * @param left_input The left table
 * @param right_input The right table
 * @param left_on The indices of the columns to join on in the left table
 * @param right_on The indices of the columns to join on in the right table
 * @param compare_nulls The null equality comparison
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 */
std::unique_ptr<cudf::table> perform_left_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate the `p_retailprice` column of the `part` table
 *
 * @param p_partkey The `p_partkey` column of the `part` table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> calc_p_retailprice(
  cudf::column_view const& p_partkey,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate the `l_suppkey` column of the `lineitem` table
 *
 * @param l_partkey The `l_partkey` column of the `lineitem` table
 * @param scale_factor The scale factor to use
 * @param num_rows The number of rows in the `lineitem` table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> calc_l_suppkey(
  cudf::column_view const& l_partkey,
  cudf::size_type const& scale_factor,
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate the `ps_suppkey` column of the `partsupp` table
 *
 * @param ps_partkey The `ps_partkey` column of the `partsupp` table
 * @param scale_factor The scale factor to use
 * @param num_rows The number of rows in the `partsupp` table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> calc_ps_suppkey(
  cudf::column_view const& ps_partkey,
  cudf::size_type const& scale_factor,
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());
/**
 * @brief Calculate the cardinality of the `lineitem` table
 *
 * @param o_rep_freqs The frequency of each `o_orderkey` value in the `lineitem` table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] cudf::size_type calc_l_cardinality(
  cudf::column_view const& o_rep_freqs,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());
/**
 * @brief Calculate the charge column for the `lineitem` table
 *
 * @param extendedprice The `l_extendedprice` column
 * @param tax The `l_tax` column
 * @param discount The `l_discount` column
 * @param stream The CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> calc_charge(
  cudf::column_view const& extendedprice,
  cudf::column_view const& tax,
  cudf::column_view const& discount,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate a column of random addresses according to TPC-H specification clause 4.2.2.7
 *
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> generate_address_column(
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate a phone number column according to TPC-H specification clause 4.2.2.9
 *
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> generate_phone_column(
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());
