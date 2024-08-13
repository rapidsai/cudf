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

#include <string>
#include <vector>

/**
 * @brief Generate a column of random strings
 *
 * @param lower The lower bound of the length of the strings
 * @param upper The upper bound of the length of the strings
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> gen_rand_str_col(
  cudf::size_type const& lower,
  cudf::size_type const& upper,
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate a column of random numbers
 * @param lower The lower bound of the random numbers
 * @param upper The upper bound of the random numbers
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
template <typename T>
std::unique_ptr<cudf::column> gen_rand_num_col(
  T const& lower,
  T const& upper,
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate a primary key column
 *
 * @param start The starting value of the primary key
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
template <typename T>
std::unique_ptr<cudf::column> gen_primary_key_col(
  T const& start,
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate a column where all the rows have the same string value
 *
 * @param value The string value to fill the column with
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> gen_rep_str_col(
  std::string const& value,
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate a column by randomly choosing from set of strings
 *
 * @param set The set of strings to choose from
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> gen_rand_str_col_from_set(
  std::vector<std::string> set,
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate a column consisting of a repeating sequence of integers
 *
 * @param limit The upper limit of the repeating sequence
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
template <typename T>
std::unique_ptr<cudf::column> gen_rep_seq_col(
  T const& seq_length,
  bool zero_indexed,
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());
