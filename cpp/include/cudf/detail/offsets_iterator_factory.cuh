/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cudf/detail/offsets_iterator.cuh>

namespace cudf {
namespace detail {

/**
 * @brief Use this class to create an offsetalator instance.
 */
struct offsetalator_factory {
  /**
   * @brief A type_dispatcher functor to create an input iterator from an offsets column
   */
  struct input_offsetalator_fn {
    template <typename T,
              std::enable_if_t<std::is_same_v<T, int32_t> or std::is_same_v<T, int64_t>>* = nullptr>
    input_offsetalator operator()(column_view const& indices)
    {
      return input_offsetalator(indices.data<IndexType>(), indices.type());
    }
    template <typename T,
              typename... Args,
              std::enable_if_t<not std::is_same_v<T, int32_t> and not std::is_same_v<T, int64_t>>* =
                nullptr>
    input_offsetalator operator()(Args&&... args)
    {
      CUDF_FAIL("offsets must be int32 or int64 type");
    }
  };

  /**
   * @brief Create an input offsetalator instance from an offsets column
   */
  static input_offsetalator make_input_iterator(column_view const& offsets)
  {
    return type_dispatcher(offsets.type(), input_offsetalator_fn{}, offsets);
  }

  /**
   * @brief A type_dispatcher functor to create an output iterator from an offsets column
   */
  struct output_offsetalator_fn {
    template <typename T,
              std::enable_if_t<std::is_same_v<T, int32_t> or std::is_same_v<T, int64_t>>* = nullptr>
    output_offsetalator operator()(mutable_column_view const& indices)
    {
      return output_offsetalator(indices.data<IndexType>(), indices.type());
    }
    template <typename T,
              typename... Args,
              std::enable_if_t<not std::is_same_v<T, int32_t> and not std::is_same_v<T, int64_t>>* =
                nullptr>
    output_offsetalator operator()(Args&&... args)
    {
      CUDF_FAIL("offsets must be int32 or int64 type");
    }
  };

  /**
   * @brief Create an output offsetalator instance from an offsets column
   */
  static output_offsetalator make_output_iterator(mutable_column_view const& offsets)
  {
    return type_dispatcher(offsets.type(), output_offsetalator_fn{}, offsets);
  }
};

}  // namespace detail
}  // namespace cudf
