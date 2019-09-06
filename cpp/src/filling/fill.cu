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

#include <copying/copy_range.cuh>
#include <cudf/copying.hpp>

namespace cudf {

namespace detail {

struct scalar_factory {
  gdf_scalar value;

  template <typename T>
  struct scalar {
    T value;
    bool is_valid;

    __device__
    T data(gdf_index_type index) { return value; }

    __device__
    bool valid(gdf_index_type index) { return is_valid; }
  };

  template <typename T>
  scalar<T> make() {
    T val{}; // Safe type pun, compiler should optimize away the memcpy
    memcpy(&val, &value.data, sizeof(T));
    return scalar<T>{val, value.is_valid};
  }
};

struct column_accessor_factory {
  gdf_column column;

  template <typename T>
  struct column_accessor {
    T const * column_data;
    bit_mask_t const * bitmask;
    gdf_index_type index;

    __device__
    T data(gdf_index_type) { 
      return column_data[index]; }

    __device__
    bool valid(gdf_index_type) {
      return bit_mask::is_valid(bitmask, index);
    }
  };

  template <typename T>
  column_accessor<T> make(gdf_index_type index) {
    return column_accessor<T>{
      static_cast<T*>(column.data),
      reinterpret_cast<bit_mask_t*>(column.valid),
      index
    };
  }
};

}; // namespace detail

void fill(gdf_column *column, gdf_scalar const& value, 
          gdf_index_type begin, gdf_index_type end)
{ 
  if (end != begin) { // otherwise no-op   
    validate(column);
    // TODO: once gdf_scalar supports string scalar values we can add support
    CUDF_EXPECTS(column->dtype != GDF_STRING_CATEGORY,
                 "cudf::fill() does not support GDF_STRING_CATEGORY columns");
    CUDF_EXPECTS(column->dtype == value.dtype, "Data type mismatch");
    detail::copy_range(column, detail::scalar_factory{value}, begin, end);
  }
}

void fill(gdf_column *column, gdf_column const& values, 
          std::vector<gdf_index_type> const& beginings,
          std::vector<gdf_index_type> const& ends)
{ 
  validate(column);
  CUDF_EXPECTS(column->dtype == values.dtype, "Data type mismatch");
  if (column->dtype == GDF_STRING_CATEGORY) {
    // if the columns are string types then we need to combine categories
    // before copying to ensure the strings referred to by the new indices
    // are included in the destination column

    // make temporary columns which will have synced categories
    // TODO: these copies seem excessively expensive, but 
    // sync_column_categories doesn't copy the valid mask
    gdf_column temp_out = cudf::copy(*column);
    gdf_column temp_in  = cudf::copy(values);

    gdf_column * input_cols[2] = {&temp_out,
                                  const_cast<gdf_column*>(&values)};
    gdf_column * temp_cols[2] = {column, &temp_in};

    // sync categories
    CUDF_EXPECTS(GDF_SUCCESS ==
      sync_column_categories(input_cols, temp_cols, 2),
      "Failed to synchronize NVCategory");

    detail::copy_ranges(column,
                        detail::column_accessor_factory{temp_in},
                        beginings, ends);
    
    gdf_column_free(&temp_out);
    gdf_column_free(&temp_in);
  } else {
    detail::copy_ranges(column,
                        detail::column_accessor_factory{values},
                        beginings, ends);
  }
}

}; // namespace cudf