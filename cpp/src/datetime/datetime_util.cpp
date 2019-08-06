/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include <iostream>
#include <cudf/cudf.h>
#include <cudf/unary.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>

namespace cudf {
namespace datetime {

namespace detail {

gdf_time_unit common_resolution(gdf_time_unit lhs_unit, gdf_time_unit rhs_unit) {
  if (lhs_unit == TIME_UNIT_NONE || rhs_unit == TIME_UNIT_NONE) {
    return TIME_UNIT_NONE;
  } else if (lhs_unit == TIME_UNIT_ns) {
    return lhs_unit;
  } else if (rhs_unit == TIME_UNIT_ns) {
    return rhs_unit;
  } else if (lhs_unit == TIME_UNIT_us) {
    return lhs_unit;
  } else if (rhs_unit == TIME_UNIT_us) {
    return rhs_unit;
  } else if (lhs_unit == TIME_UNIT_ms) {
    return lhs_unit;
  } else if (rhs_unit == TIME_UNIT_ms) {
    return rhs_unit;
  } else if (lhs_unit == TIME_UNIT_s) {
    return lhs_unit;
  } else if (rhs_unit == TIME_UNIT_s) {
    return rhs_unit;
  }
  return TIME_UNIT_NONE;
}

} // namespace detail

std::pair<gdf_column, gdf_column> cast_to_common_resolution(gdf_column const& lhs,
                                                            gdf_column const& rhs) {

  gdf_column lhs_out = cudf::empty_like(lhs);
  gdf_column rhs_out = cudf::empty_like(rhs);

  if ((lhs.dtype == GDF_TIMESTAMP || lhs.dtype == GDF_DATE64) &&
      (rhs.dtype == GDF_TIMESTAMP || rhs.dtype == GDF_DATE64)) {

    gdf_time_unit lhs_unit = lhs.dtype_info.time_unit;
    gdf_time_unit rhs_unit = rhs.dtype_info.time_unit;

    // GDF_DATE64 and TIME_UNIT_NONE are always int64_t milliseconds
    if (lhs.dtype == GDF_DATE64 || lhs_unit == TIME_UNIT_NONE) {
      lhs_unit = TIME_UNIT_ms;
    }
    if (rhs.dtype == GDF_DATE64 || rhs_unit == TIME_UNIT_NONE) {
      rhs_unit = TIME_UNIT_ms;
    }
    if (lhs_unit != rhs_unit) {
      auto lcd_unit = detail::common_resolution(lhs_unit, rhs_unit);
      if (lcd_unit == lhs_unit) {
        rhs_out = cudf::cast(rhs, GDF_TIMESTAMP, gdf_dtype_extra_info{lcd_unit});
      } else if (lcd_unit == rhs_unit) {
        lhs_out = cudf::cast(lhs, GDF_TIMESTAMP, gdf_dtype_extra_info{lcd_unit});
      }
    }
  }

  return std::make_pair(lhs_out, rhs_out);
}

}  // namespace datetime
}  // namespace cudf
