/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>
#include <cudf/column/column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>

namespace cudf {

namespace detail {

/**
 * @brief Given a column-device-view, an instance of this class provides a
 * wrapper on this compound column for list operations.
 * Analogous to list_column_view.
 */
class lists_column_device_view {
 public:
  ~lists_column_device_view()                               = default;
  lists_column_device_view(lists_column_device_view const&) = default;
  lists_column_device_view(lists_column_device_view&&)      = default;
  lists_column_device_view& operator=(lists_column_device_view const&) = default;
  lists_column_device_view& operator=(lists_column_device_view&&) = default;

  lists_column_device_view(column_device_view const& underlying_) : underlying(underlying_)
  {
    CUDF_EXPECTS(underlying_.type().id() == type_id::LIST,
                 "lists_column_device_view only supports lists");
  }

  /**
   * @brief Fetches number of rows in the lists column
   */
  CUDA_HOST_DEVICE_CALLABLE cudf::size_type size() const { return underlying.size(); }

  /**
   * @brief Fetches the offsets column of the underlying list column.
   */
  CUDA_DEVICE_CALLABLE column_device_view offsets() const
  {
    return underlying.child(lists_column_view::offsets_column_index);
  }

  /**
   * @brief Fetches the child column of the underlying list column.
   */
  CUDA_DEVICE_CALLABLE column_device_view child() const
  {
    return underlying.child(lists_column_view::child_column_index);
  }

  /**
   * @brief Indicates whether the list column is nullable.
   */
  CUDA_DEVICE_CALLABLE bool nullable() const { return underlying.nullable(); }

  /**
   * @brief Indicates whether the row (i.e. list) at the specified
   * index is null.
   */
  CUDA_DEVICE_CALLABLE bool is_null(size_type idx) const { return underlying.is_null(idx); }

 private:
  column_device_view underlying;
};

}  // namespace detail

}  // namespace cudf
