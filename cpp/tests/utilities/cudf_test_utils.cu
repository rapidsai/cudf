/*
 * Copyright 2018 BlazingDB, Inc.
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

#include "cudf_test_utils.cuh"

void print_gdf_column(gdf_column const * the_column)
{
  const size_t num_rows = the_column->size;
  const gdf_dtype gdf_col_type = the_column->dtype;
  switch(gdf_col_type)
  {
    case GDF_INT8:
      {
        using col_type = int8_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_INT16:
      {
        using col_type = int16_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_INT32:
      {
        using col_type = int32_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_INT64:
      {
        using col_type = int64_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_FLOAT32:
      {
        using col_type = float;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_FLOAT64:
      {
        using col_type = double;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    default:
      {
        std::cout << "Attempted to print unsupported type.\n";
      }
  }
}

void print_valid_data(const gdf_valid_type *validity_mask,
                      const size_t num_rows)
{
  cudaError_t error;
  cudaPointerAttributes attrib;
  cudaPointerGetAttributes(&attrib, validity_mask);
  error = cudaGetLastError();

  const size_t num_masks = gdf_get_num_chars_bitmask(num_rows);
  std::vector<gdf_valid_type> h_mask(num_masks);
  if (error != cudaErrorInvalidValue && attrib.memoryType == cudaMemoryTypeDevice)
    cudaMemcpy(h_mask.data(), validity_mask, num_masks * sizeof(gdf_valid_type), cudaMemcpyDeviceToHost);
  else
    memcpy(h_mask.data(), validity_mask, num_masks * sizeof(gdf_valid_type));

  std::transform(h_mask.begin(), h_mask.end(), std::ostream_iterator<std::string>(std::cout, " "), 
                 [](gdf_valid_type x){ 
                   auto bits = std::bitset<GDF_VALID_BITSIZE>(x).to_string('@'); 
                   return std::string(bits.rbegin(), bits.rend());  
                 });
  std::cout << std::endl;
}

