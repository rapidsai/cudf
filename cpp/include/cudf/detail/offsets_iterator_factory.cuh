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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/offsets_iterator.cuh>

namespace cudf {
namespace detail {

/**
 * @brief Use this class to create an offsetalator instance.
 */
struct offsetalator_factory {
  /**
   * @brief Create an input offsetalator instance from an offsets column
   */
  static input_offsetalator make_input_iterator(column_view const& offsets)
  {
    return input_offsetalator(offsets.head(), offsets.type());
  }

  /**
   * @brief Create an output offsetalator instance from an offsets column
   */
  static output_offsetalator make_output_iterator(mutable_column_view const& offsets)
  {
    return output_offsetalator(offsets.head(), offsets.type());
  }
};

}  // namespace detail
}  // namespace cudf
