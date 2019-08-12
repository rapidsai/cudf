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

#include "copy_range.cuh"
#include <cudf/copying.hpp>

namespace cudf {

namespace detail {

struct column_range_factory {
  gdf_column column;
  gdf_index_type begin;

  template <typename T>
  struct column_range {
    T const * column_data;
    bit_mask_t const * bitmask;
    gdf_index_type begin;

    __device__
    T data(gdf_index_type index) { 
      return column_data[begin + index]; }

    __device__
    bool valid(gdf_index_type index) {
      return bit_mask::is_valid(bitmask, begin + index);
    }
  };

  template <typename T>
  column_range<T> make() {
    return column_range<T>{
      static_cast<T*>(column.data),
      reinterpret_cast<bit_mask_t*>(column.valid),
      begin
    };
  }
};

}; // namespace detail

void copy_range(gdf_column *out_column, gdf_column const &in_column,
                gdf_index_type out_begin, gdf_index_type out_end, 
                gdf_index_type in_begin)
{
  gdf_size_type num_elements = out_end - out_begin;
  if (num_elements != 0) { // otherwise no-op
    validate(in_column);
    validate(out_column);
    
    CUDF_EXPECTS(out_column->dtype == in_column.dtype, "Data type mismatch");
    
    if (cudf::has_nulls(in_column)) {
      CUDF_EXPECTS(cudf::is_nullable(*out_column),
                   "Expected nullable output column");
    }
    
    // out range validated by detail::copy_range
    CUDF_EXPECTS((in_begin >= 0) && (in_begin + num_elements <= in_column.size),
                 "Range is out of bounds");

    if (out_column->dtype == GDF_STRING_CATEGORY) {
      // if the columns are string types then we need to combine categories
      // before copying to ensure the strings referred to by the new indices
      // are included in the destination column

      // make temporary columns which will have synced categories
      // TODO: these copies seem excessively expensive, but 
      // sync_column_categories doesn't copy the valid mask
      gdf_column temp_out = cudf::copy(*out_column);
      gdf_column temp_in  = cudf::copy(in_column);

      gdf_column * input_cols[2] = {&temp_out,
                                    const_cast<gdf_column*>(&in_column)};
      gdf_column * temp_cols[2] = {out_column, &temp_in};

      // sync categories
      CUDF_EXPECTS(GDF_SUCCESS ==
        sync_column_categories(input_cols, temp_cols, 2),
        "Failed to synchronize NVCategory");

      detail::copy_range(out_column,
                         detail::column_range_factory{temp_in, in_begin},
                         out_begin, out_end);
      
      gdf_column_free(&temp_out);
      gdf_column_free(&temp_in);
    }
    else {
      detail::copy_range(out_column,
                         detail::column_range_factory{in_column, in_begin},
                         out_begin, out_end);
    }
  }
}

}; // namespace cudf