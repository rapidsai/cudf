/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "thrust/iterator/counting_iterator.h"
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/transform.h>

#include <memory>
#include <type_traits>
#include <vector>

/**
 * @file random_table_generator.cuh
 * @brief Random table and column generation utilities for external sorting example
 *
 * This header provides utilities to generate random tables and columns with various
 * data types for testing and demonstration purposes. The implementation uses thrust
 * algorithms for GPU-accelerated generation.
 */

namespace cudf {
namespace examples {

/**
 * @brief Generate a random numeric column using thrust
 *
 * Creates a column filled with random values of the specified numeric type.
 * Uses thrust::uniform_int_distribution for integral types and 
 * thrust::uniform_real_distribution for floating-point types.
 *
 * @tparam T The numeric data type (int32_t, int64_t, float, double, etc.)
 * @param lower Lower bound of the random values (inclusive)
 * @param upper Upper bound of the random values (inclusive for int, exclusive for float)
 * @param num_rows Number of rows in the output column
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource for allocations
 * @return Unique pointer to the generated column
 */
template <typename T>
std::unique_ptr<cudf::column> generate_random_numeric_column(T lower,
                                                             T upper,
                                                             cudf::size_type num_rows,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr)
{
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_to_id<T>()}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);

  // Generate random numbers on device using thrust
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::counting_iterator(0),
    thrust::counting_iterator(num_rows),
    col->mutable_view().begin<T>(),
    [lower, upper] __device__(cudf::size_type idx) -> T {
      thrust::minstd_rand engine;
      engine.discard(idx);  // Deterministic seeding based on index
      if constexpr (std::is_integral_v<T>) {
        thrust::uniform_int_distribution<T> dist(lower, upper);
        return dist(engine);
      } else {
        thrust::uniform_real_distribution<T> dist(lower, upper);
        return dist(engine);
      }
    });

  return col;
}

/**
 * @brief Generate a table with n_columns random columns and m_rows
 *
 * Creates a table with a mix of different data types to demonstrate libcudf's
 * type system and provide realistic test data. The column types cycle through:
 * INT32, FLOAT64, INT64, FLOAT32, INT16.
 *
 * Each column type uses different value ranges to create diverse datasets:
 * - INT32: Large integers with column-specific offsets
 * - FLOAT64: Double precision floats in range [0, 1000+offset]
 * - INT64: Large long integers around 1e9-1e10 range
 * - FLOAT32: Single precision floats with small ranges
 * - INT16: Short integers with moderate ranges
 *
 * @param n_columns Number of columns to generate
 * @param m_rows Number of rows in each column
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource for allocations
 * @return Unique pointer to the generated table
 */
std::unique_ptr<cudf::table> generate_random_table(cudf::size_type n_columns,
                                                    cudf::size_type m_rows,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(n_columns);

  // Create a mix of different data types for more realistic testing
  std::vector<cudf::data_type> types = {
    cudf::data_type{cudf::type_id::INT32},    // Integer column for primary sorting
    cudf::data_type{cudf::type_id::FLOAT64},  // Double precision floating point
    cudf::data_type{cudf::type_id::INT64},    // Long integer
    cudf::data_type{cudf::type_id::FLOAT32},  // Single precision floating point
    cudf::data_type{cudf::type_id::INT16}     // Short integer
  };

  for (cudf::size_type i = 0; i < n_columns; ++i) {
    auto type_idx = i % types.size();
    switch (types[type_idx].id()) {
      case cudf::type_id::INT32:
        columns.emplace_back(generate_random_numeric_column<int32_t>(
          1000000 + i * 100000, 9999999 + i * 100000, m_rows, stream, mr));
        break;
      case cudf::type_id::FLOAT64:
        columns.emplace_back(
          generate_random_numeric_column<double>(0.0, 1000.0 + i * 100, m_rows, stream, mr));
        break;
      case cudf::type_id::INT64:
        columns.emplace_back(generate_random_numeric_column<int64_t>(
          static_cast<int64_t>(1e9) + i * 1000000,
          static_cast<int64_t>(1e10) + i * 1000000,
          m_rows,
          stream,
          mr));
        break;
      case cudf::type_id::FLOAT32:
        columns.emplace_back(
          generate_random_numeric_column<float>(10.0f + i, 1000.0f + i * 50, m_rows, stream, mr));
        break;
      case cudf::type_id::INT16:
        columns.emplace_back(generate_random_numeric_column<int16_t>(
          static_cast<int16_t>(100 + i * 10),
          static_cast<int16_t>(30000 + i * 1000),
          m_rows,
          stream,
          mr));
        break;
      default: break;  // Should not reach here with current types
    }
  }

  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace examples
}  // namespace cudf
