/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace cudf {
namespace test {

/**
 * @brief Verifies the property equality of two columns.
 *
 * @param lhs The first column
 * @param rhs The second column
 */
void expect_column_properties_equal(cudf::column_view lhs, cudf::column_view rhs);

/**
 * @brief Verifies the element-wise equality of two columns.
 *
 * Treats null elements as equivalent.
 *
 * @param lhs The first column
 * @param rhs The second column
 */
void expect_columns_equal(cudf::column_view lhs, cudf::column_view rhs);

/**
 * @brief Verifies the bitwise equality of two device memory buffers.
 *
 * @param lhs The first buffer
 * @param rhs The second buffer
 * @param size_bytes The number of bytes to check for equality
 */
void expect_equal_buffers(void const* lhs, void const* rhs,
                          std::size_t size_bytes);

/**
 * @brief Copies the data and bitmask of a `column_view` to the host.
 *
 * @tparam T The data type of the elements of the `column_view`
 * @param c the `column_view` to copy from
 * @return std::pair<std::vector<T>, std::vector<bitmask_type>> first is the
 *  `column_view`'s data, and second is the column's bitmask.
 */
template <typename T>
std::pair<std::vector<T>, std::vector<bitmask_type>> to_host(column_view c) {
  std::vector<T> host_data;
  std::vector<bitmask_type> host_bitmask;

  auto col = column(c);

  if (col.size() > 0) {
    host_data.resize(col.size());
    CUDA_TRY(cudaMemcpy(host_data.data(), col.view().head<T>(),
                        col.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  }

  if (col.nullable()) {
    size_t mask_allocation_bytes = bitmask_allocation_size_bytes(c.size());
    host_bitmask.resize(mask_allocation_bytes);
    size_t num_mask_elements = num_bitmask_words(c.size());
    CUDA_TRY(cudaMemcpy(host_bitmask.data(), col.view().null_mask(),
                        num_mask_elements * sizeof(bitmask_type),
                        cudaMemcpyDeviceToHost));
  }

  return std::make_pair(host_data, host_bitmask);
}

/**
 * @brief Copies the data and bitmask of a `column_view` of strings
 * column to the host.
 *
 * @throw cudf::logic_error if c is not strings column.
 *
 * @param c the `column_view` of strings to copy from
 * @return std::pair first is `std::vector` of `std::string`
 * and second is the column's bitmask.
 */
template <>
inline std::pair<std::vector<std::string>, std::vector<bitmask_type>> to_host(column_view c) {
  std::vector<std::string> host_data;
  std::vector<bitmask_type> host_bitmask;

  auto strings = strings_column_view(c);
  auto strings_data = cudf::strings::create_offsets(strings);
  thrust::host_vector<char> h_chars(strings_data.first);         // copies vectors to
  thrust::host_vector<size_type> h_offsets(strings_data.second); // host automatically

  // copy nulls to host bitmask
  if( c.has_nulls() )
  {
    auto num_bitmasks = num_bitmask_words(c.size());
    host_bitmask.resize(num_bitmasks);
    CUDA_TRY(cudaMemcpy(host_bitmask.data(), c.null_mask(),
                        num_bitmasks*sizeof(bitmask_type), cudaMemcpyDeviceToHost));
  }

  // build std::string vector from chars and offsets
  if( !h_chars.empty() ) // check for all nulls case
  {
    for( size_type idx=0; idx < strings.size(); ++idx )
    {
        auto offset = h_offsets[idx];
        auto length = h_offsets[idx+1] - offset;
        host_data.push_back(std::string( h_chars.data()+offset, length));
    }
  }

  return std::make_pair(host_data, host_bitmask);
}

}  // namespace test
}  // namespace cudf
