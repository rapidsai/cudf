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

#include "hostdevice_vector.hpp"

#include <cudf/cudf.h>
#include <utilities/legacy/column_utils.hpp>
#include <cudf/utilities/error.hpp>

#include <nvstrings/NVStrings.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

/**
 * @brief A helper class that wraps a gdf_column and any associated memory.
 *
 * This abstraction initializes and manages a gdf_column (fields and memory)
 * while still allowing direct access. Memory is automatically deallocated
 * unless ownership is transferred via releasing and assigning the raw pointer.
 **/
class gdf_column_wrapper {
  using str_pair = std::pair<const char *, size_t>;
  using str_ptr = std::unique_ptr<NVStrings, decltype(&NVStrings::destroy)>;

 public:
  gdf_column_wrapper(cudf::size_type size, gdf_dtype dtype,
                     gdf_dtype_extra_info dtype_info, const std::string name) {
    col = static_cast<gdf_column *>(malloc(gdf_column_sizeof()));
    gdf_column_view_augmented(col, nullptr, nullptr, size, dtype, 0, dtype_info,
                              name.c_str());
  }

  ~gdf_column_wrapper() {
    if (col) {
      RMM_FREE(col->data, 0);
      RMM_FREE(col->valid, 0);
      free(col->col_name);
    }
    free(col);
  }

  gdf_column_wrapper(const gdf_column_wrapper &other) = delete;
  gdf_column_wrapper(gdf_column_wrapper &&other) : col(other.col) {
    other.col = nullptr;
  }

  void allocate() {
    const auto num_rows = std::max(col->size, 1);
    const auto valid_size = gdf_valid_allocation_size(num_rows);

    // For strings, just store the <ptr, length>. Eventually, the column's data
    // ptr is replaced with an `NvString` instance created from these pairs.
    // NvStrings does not use the valid mask so it expects invalid rows to be
    // <nullptr, 0> initialized.
    if (col->dtype == GDF_STRING) {
      RMM_TRY(RMM_ALLOC(&col->data, num_rows * sizeof(str_pair), 0));
      CUDA_TRY(cudaMemsetAsync(col->data, 0, num_rows * sizeof(str_pair)));
    } else {
      RMM_TRY(RMM_ALLOC(&col->data, num_rows * cudf::byte_width(*col), 0));
    }
    RMM_TRY(RMM_ALLOC(&col->valid, valid_size, 0));
    CUDA_TRY(cudaMemsetAsync(col->valid, 0, valid_size));
  }

  void finalize() {
    // Create and initialize an `NvStrings` instance from <ptr, length> data.
    // The container copies the string data to its internal memory so the source
    // memory must not be released prior to calling this method.
    if (col->dtype == GDF_STRING) {
      auto str_list = static_cast<str_pair *>(col->data);
      str_ptr str_data(NVStrings::create_from_index(str_list, col->size),
                       &NVStrings::destroy);
      CUDF_EXPECTS(str_data != nullptr, "Cannot create `NvStrings` instance");
      RMM_FREE(std::exchange(col->data, str_data.release()), 0);
    }
  }

  gdf_column *operator->() const noexcept { return col; }
  gdf_column *get() const noexcept { return col; }
  gdf_column *release() noexcept {
    auto temp = col;
    col = nullptr;
    return temp;
  }

 private:
  gdf_column *col = nullptr;
};
