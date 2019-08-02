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

#include <cudf/cudf.h>
#include <cudf/unary.hpp>
#include <cudf/datetime.hpp>

namespace cudf {
namespace datetime {

std::pair<gdf_col_pointer, gdf_col_pointer> resolve_common_time_unit(gdf_column const& lhs,
																  	 gdf_column const& rhs) {

	auto gdf_col_deleter = [](gdf_column* col) { gdf_column_free(col); };

	gdf_col_pointer lhs_out, rhs_out;

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
			gdf_column out_col;
			gdf_dtype_extra_info out_info{std::max(lhs_unit, rhs_unit)};
			if (lhs_unit > rhs_unit) {
				out_col = cudf::cast(rhs, GDF_TIMESTAMP, out_info);
				rhs_out = {&out_col, gdf_col_deleter};
			} else {
				out_col = cudf::cast(lhs, GDF_TIMESTAMP, out_info);
				lhs_out = {&out_col, gdf_col_deleter};
			}
		}
	}

	return std::make_pair(std::move(lhs_out), std::move(rhs_out));
}

}  // namespace datetime
}  // namespace cudf
