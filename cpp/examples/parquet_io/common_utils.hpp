/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>
#include <string>

/**
 * @file common_utils.hpp
 * @brief Common utilities for `parquet_io` examples
 *
 */

/**
 * @brief Get encoding type from the keyword
 *
 * @param name encoding keyword name
 * @return corresponding column encoding type
 */
[[nodiscard]] cudf::io::column_encoding get_encoding_type(std::string name);

/**
 * @brief Get compression type from the keyword
 *
 * @param name compression keyword name
 * @return corresponding compression type
 */
[[nodiscard]] cudf::io::compression_type get_compression_type(std::string name);

/**
 * @brief Get boolean from they keyword
 *
 * @param input keyword affirmation string such as: Y, T, YES, TRUE, ON
 * @return true or false
 */
[[nodiscard]] bool get_boolean(std::string input);

/**
 * @brief Concatenate a vector of tables and return the resultant table
 *
 * @param tables Vector of tables to concatenate
 * @param stream CUDA stream to use
 *
 * @return Unique pointer to the resultant concatenated table.
 */
std::unique_ptr<cudf::table> concatenate_tables(std::vector<std::unique_ptr<cudf::table>> tables,
                                                rmm::cuda_stream_view stream);

/**
 * @brief Returns a string containing current date and time
 *
 */
std::string current_date_and_time();
