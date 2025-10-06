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

#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace detail::row::primitive {
class row_equality_comparator;

template <template <typename> class Hash>
class row_hasher;
}  // namespace detail::row::primitive

namespace detail::row::equality {

namespace hash {
class row_hasher;
}

/**
 * @brief Preprocessed table for use with row equality comparison or row hashing
 *
 */
struct preprocessed_table {
  /**
   * @brief Factory to construct preprocessed_table for use with
   * row equality comparison or row hashing
   *
   * Sets up the table for use with row equality comparison or row hashing. The resulting
   * preprocessed table can be passed to the constructor of `equality::self_comparator` to
   * avoid preprocessing again.
   *
   * @param table The table to preprocess
   * @param stream The cuda stream to use while preprocessing.
   * @return A preprocessed table as shared pointer
   */
  static std::shared_ptr<preprocessed_table> create(table_view const& table,
                                                    rmm::cuda_stream_view stream);

  /**
   * @brief Implicit conversion operator to a `table_device_view` of the preprocessed table.
   *
   * @return table_device_view
   */
  operator table_device_view() { return *_t; }

 private:
  friend class self_comparator;
  friend class two_table_comparator;
  friend class hash::row_hasher;
  friend class ::cudf::detail::row::primitive::row_equality_comparator;

  template <template <typename> class Hash>
  friend class ::cudf::detail::row::primitive::row_hasher;

  using table_device_view_owner =
    std::invoke_result_t<decltype(table_device_view::create), table_view, rmm::cuda_stream_view>;

  preprocessed_table(table_device_view_owner&& table,
                     std::vector<rmm::device_buffer>&& null_buffers,
                     std::vector<std::unique_ptr<column>>&& tmp_columns)
    : _t(std::move(table)),
      _null_buffers(std::move(null_buffers)),
      _tmp_columns(std::move(tmp_columns))
  {
  }

  table_device_view_owner _t;
  std::vector<rmm::device_buffer> _null_buffers;
  std::vector<std::unique_ptr<column>> _tmp_columns;
};

}  // namespace detail::row::equality

namespace detail::row::hash {

using preprocessed_table = row::equality::preprocessed_table;

}  // namespace detail::row::hash
}  // namespace CUDF_EXPORT cudf
