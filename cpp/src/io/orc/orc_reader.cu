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

#include "orc.hpp"

#include "cudf.h"
#include "io/comp/gpuinflate.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"

#include <cuda_runtime.h>
#include <nvstrings/NVStrings.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include <array>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/**
 * @brief Helper class for memory mapping a file source
 **/
class DataSource {
 public:
  explicit DataSource(const char *filepath) {
    fd = open(filepath, O_RDONLY);
    CUDF_EXPECTS(fd > 0, "Cannot open file");

    struct stat st {};
    CUDF_EXPECTS(fstat(fd, &st) == 0, "Cannot query file size");

    mapped_size = st.st_size;
    CUDF_EXPECTS(mapped_size > 0, "Unexpected zero-sized file");

    mapped_data = mmap(NULL, mapped_size, PROT_READ, MAP_PRIVATE, fd, 0);
    CUDF_EXPECTS(mapped_data != MAP_FAILED, "Cannot memory-mapping file");
  }

  ~DataSource() {
    if (mapped_data) {
      munmap(mapped_data, mapped_size);
    }
    if (fd) {
      close(fd);
    }
  }

  const uint8_t *data() const { return static_cast<uint8_t *>(mapped_data); }
  size_t size() const { return mapped_size; }

 private:
  void *mapped_data = nullptr;
  size_t mapped_size = 0;
  int fd = 0;
};

/**
 * @brief A helper class that wraps a gdf_column and any associated memory.
 *
 * This abstraction initializes and manages a gdf_column (fields and memory)
 * while still allowing direct access. Memory is automatically deallocated
 * unless ownership is transferred via releasing and assigning the raw pointer.
 **/
class gdf_column_wrapper {
 public:
  gdf_column_wrapper(gdf_size_type size, gdf_dtype dtype,
                     gdf_dtype_extra_info dtype_info, const std::string name) {
    col = (gdf_column *)malloc(sizeof(gdf_column));
    col->col_name = (char *)malloc(name.length() + 1);
    strcpy(col->col_name, name.c_str());
    gdf_column_view_augmented(col, nullptr, nullptr, size, dtype, 0, dtype_info);
  }

  ~gdf_column_wrapper() {
    if (col) {
      RMM_FREE(col->data, 0);
      RMM_FREE(col->valid, 0);
      free(col->col_name);
    }
    free(col);
  };

  gdf_column_wrapper(const gdf_column_wrapper &other) = delete;
  gdf_column_wrapper(gdf_column_wrapper &&other) : col(other.col) {
    other.col = nullptr;
  }

  gdf_error allocate() {
    // For strings, just store the ptr + length. Eventually, column's data ptr
    // is replaced with an NvString instance created from these pairs.
    const auto num_rows = std::max(col->size, 1);
    const auto column_byte_width = (col->dtype == GDF_STRING)
                                       ? sizeof(std::pair<const char*, int>)
                                       : gdf_dtype_size(col->dtype);

    RMM_TRY(RMM_ALLOC(&col->data, num_rows * column_byte_width, 0));
    RMM_TRY(RMM_ALLOC(&col->valid, gdf_valid_allocation_size(num_rows), 0));
    CUDA_TRY(cudaMemset(col->valid, 0, gdf_valid_allocation_size(num_rows)));

    return GDF_SUCCESS;
  }

  gdf_column *operator->() const { return col; }
  gdf_column *get() const { return col; }
  gdf_column *release() {
    auto temp = col;
    col = nullptr;
    return temp;
  }

 private:
  gdf_column *col = nullptr;
};

/**
 * @brief Reads Apache ORC data and returns an array of gdf_columns.
 *
 * @param[in,out] args Structure containing input and output args
 *
 * @return gdf_error GDF_SUCCESS if successful, otherwise an error code.
 **/
gdf_error read_orc(orc_read_arg *args) {

  std::vector<gdf_column_wrapper> columns;
  int num_columns = 0;
  int num_rows = 0;
  int index_col = -1;

  DataSource input(args->source);

  // Select columns

  // Obtain list of stripe chunks

  // Transfer ownership to raw pointer output arguments
  args->data = (gdf_column **)malloc(sizeof(gdf_column *) * num_columns);
  for (int i = 0; i < num_columns; ++i) {
    args->data[i] = columns[i].release();

    // For string dtype, allocate and return an NvStrings container instance,
    // deallocating the original string list memory in the process.
    // This container takes a list of string pointers and lengths, and copies
    // into its own memory so the source memory must not be released yet.
    if (args->data[i]->dtype == GDF_STRING) {
      using str_pair = std::pair<const char *, size_t>;

      auto str_list = static_cast<str_pair *>(args->data[i]->data);
      auto str_data = NVStrings::create_from_index(str_list, num_rows);
      RMM_FREE(std::exchange(args->data[i]->data, str_data), 0);
    }
  }
  args->num_cols_out = num_columns;
  args->num_rows_out = num_rows;
  if (index_col != -1) {
    args->index_col = (int *)malloc(sizeof(int));
    *args->index_col = index_col;
  }

  return GDF_SUCCESS;
}
