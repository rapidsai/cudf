/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef COLUMN_WRAPPER_H
#define COLUMN_WRAPPER_H

#include "cudf.h"
#include "rmm/rmm.h"
#include "utilities/bit_util.cuh"
#include "utilities/type_dispatcher.hpp"

namespace cudf {
namespace test {

template <typename ColumnType>
struct column_wrapper {
  column_wrapper(std::vector<ColumnType> const& host_data,
                 std::vector<gdf_valid_type> const& host_bitmask) {
    initialize_with_host_data(host_data, host_bitmask);
  }

  template <typename BitInitializerType>
  column_wrapper(std::vector<ColumnType> const& host_data,
                 BitInitializerType bit_initializer) {
    const size_t num_masks = gdf_get_num_chars_bitmask(host_data.size());

    // Initialize the valid mask for this column using the initializer
    std::vector<gdf_valid_type> host_bitmask(num_masks, 0);
    for (gdf_index_type row = 0; row < host_data.size(); ++row) {
      if (true == bit_initializer(row)) {
        gdf::util::turn_bit_on(host_bitmask.data(), row);
      }
    }
    initialize_with_host_data(host_data, host_bitmask);
  }

  template <typename ValueInitializerType, typename BitInitializerType>
  column_wrapper(gdf_size_type column_size,
                 ValueInitializerType value_initalizer,
                 BitInitializerType bit_initializer) {
    const size_t num_masks = gdf_get_num_chars_bitmask(column_size);

    // Initialize the values and bitmask using the initializers
    std::vector<ColumnType> host_data(column_size);
    std::vector<gdf_valid_type> host_bitmask(num_masks, 0);

    for (gdf_index_type row = 0; row < column_size; ++row) {
      host_data[row] = value_initalizer(row);

      if (true == bit_initializer(row)) {
        gdf::util::turn_bit_on(host_bitmask.data(), row);
      }
    }
    initialize_with_host_data(host_data, host_bitmask);
  }

  ~column_wrapper() {
    RMM_FREE(the_column.data, 0);
    RMM_FREE(the_column.valid, 0);
    the_column.size = 0;
  }

 private:
  void initialize_with_host_data(
      std::vector<ColumnType> const& host_data,
      std::vector<gdf_valid_type> const& host_bitmask) {
    // Allocate device storage for gdf_column and copy contents from host_data
    RMM_ALLOC(&(the_column.data), host_data.size() * sizeof(ColumnType), 0);
    cudaMemcpy(the_column.data, host_data.data(),
               host_data.size() * sizeof(ColumnType), cudaMemcpyHostToDevice);

    // Fill the gdf_column members
    the_column.size = host_data.size();
    the_column.dtype = cudf::type_to_gdf_dtype<ColumnType>::value;
    gdf_dtype_extra_info extra_info;
    extra_info.time_unit = TIME_UNIT_NONE;
    the_column.dtype_info = extra_info;

    // If a validity bitmask vector was passed in, allocate device storage
    // and copy its contents from the host vector
    if (host_bitmask.size() > 0) {
      RMM_ALLOC(&(the_column.valid),
                host_bitmask.size() * sizeof(gdf_valid_type), 0);
      cudaMemcpy(the_column.valid, host_bitmask.data(),
                 host_bitmask.size() * sizeof(gdf_valid_type),
                 cudaMemcpyHostToDevice);
    } else {
      the_column.valid = nullptr;
    }
    set_null_count(&the_column);
  }
  gdf_column the_column;
};

}  // namespace test
}  // namespace cudf
#endif
