/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>

namespace cudf {

namespace detail {

/**
 * @brief Given a column-device-view, an instance of this class provides a
 * wrapper on this compound column for struct operations.
 * Analogous to list_column_view.
 */
class structs_column_device_view {
 public:
  ~structs_column_device_view()                                 = default;
  structs_column_device_view(structs_column_device_view const&) = default;
  structs_column_device_view(structs_column_device_view&&)      = default;
  structs_column_device_view& operator=(structs_column_device_view const&) = default;
  structs_column_device_view& operator=(structs_column_device_view&&) = default;

  CUDF_HOST_DEVICE structs_column_device_view(column_device_view const& underlying_)
    : underlying(underlying_)
  {
#ifdef __CUDA_ARCH__
    cudf_assert(underlying.type().id() == type_id::STRUCT and
                "structs_column_device_view only supports structs");
#else
    CUDF_EXPECTS(underlying_.type().id() == type_id::STRUCT,
                 "structs_column_device_view only supports structs");
#endif
  }

  /**
   * @brief Fetches number of rows in the struct column
   */
  [[nodiscard]] CUDF_HOST_DEVICE inline cudf::size_type size() const { return underlying.size(); }

  /**
   * @brief Fetches the child column of the underlying struct column.
   */
  [[nodiscard]] __device__ inline column_device_view child(size_type idx) const
  {
    return underlying.child(idx);
  }

  /**
   * @brief Fetches the child column of the underlying struct column.
   */
  [[nodiscard]] __device__ inline column_device_view sliced_child(size_type idx) const
  {
    return child(idx).slice(offset(), size());
  }

  /**
   * @brief Indicates whether the struct column is nullable.
   */
  [[nodiscard]] __device__ inline bool nullable() const { return underlying.nullable(); }

  /**
   * @brief Indicates whether the row (i.e. struct) at the specified
   * index is null.
   */
  [[nodiscard]] __device__ inline bool is_null(size_type idx) const
  {
    return underlying.is_null(idx);
  }

  /**
   * @brief Fetches the offset of the underlying column_device_view,
   *        in case it is a sliced/offset column.
   */
  [[nodiscard]] __device__ inline size_type offset() const { return underlying.offset(); }

 private:
  column_device_view underlying;
};

}  // namespace detail

}  // namespace cudf
