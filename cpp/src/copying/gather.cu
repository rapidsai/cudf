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
#include "gather.cuh"
#include <cudf/copying.hpp>
#include <cudf/cudf.h>
#include <utilities/bit_util.cuh>
#include <utilities/cudf_utils.h>
#include <cudf/utilities/legacy/type_dispatcher.hpp>

#include <cudf/legacy/table.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <algorithm>

#include <table/legacy/device_table.cuh>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <nvstrings/NVCategory.h>
#include <nvstrings/NVStrings.h>
#include <utilities/column_utils.hpp>
#include <utilities/cuda_utils.hpp>

#include <bitmask/valid_if.cuh>

using bit_mask::bit_mask_t;

namespace cudf {
namespace detail {

struct dispatch_map_type {
  template <typename map_type, std::enable_if_t<std::is_integral<map_type>::value>* = nullptr>
  void operator()(table const *source_table, gdf_column const& gather_map,
      table *destination_table, bool check_bounds,
      bool ignore_out_of_bounds, bool sync_nvstring_category = false,
      bool allow_negative_indices = false)
  {

    map_type const * typed_gather_map = static_cast<map_type const*>(gather_map.data);

    if (check_bounds) {
      gdf_index_type begin = (allow_negative_indices) ? -source_table->num_rows() : 0;
      CUDF_EXPECTS(
	  destination_table->num_rows() == thrust::count_if(
	      rmm::exec_policy()->on(0),
	      typed_gather_map,
	      typed_gather_map + destination_table->num_rows(),
	      bounds_checker<map_type>{begin, source_table->num_rows()}),
	  "Index out of bounds.");
    }

    if (allow_negative_indices) {
      gather(source_table,
	     thrust::make_transform_iterator(
		 typed_gather_map,
		 index_converter<map_type,index_conversion::NEGATIVE_TO_POSITIVE>{source_table->num_rows()}),
	     destination_table,
	     check_bounds,
	     ignore_out_of_bounds,
	     sync_nvstring_category,
	     allow_negative_indices
	     );
    }
    else {
      gather(source_table,
	     thrust::make_transform_iterator(
		 typed_gather_map,
		 index_converter<map_type>{source_table->num_rows()}),
	     destination_table,
	     check_bounds,
	     ignore_out_of_bounds,
	     sync_nvstring_category,
	     allow_negative_indices
	     );
    }
  }

  template <typename map_type, std::enable_if_t<not std::is_integral<map_type>::value>* = nullptr>
  void operator()(table const *source_table, gdf_column const& gather_map,
                  table *destination_table, bool check_bounds,
		  bool ignore_out_of_bounds, bool sync_nvstring_category = false,
		  bool allow_negative_indices = false) {
   CUDF_FAIL("Gather map must be an integral type.");
  }
};

void gather(table const *source_table, gdf_column const& gather_map,
            table *destination_table, bool check_bounds, bool ignore_out_of_bounds,
            bool sync_nvstring_category, bool allow_negative_indices) {
  CUDF_EXPECTS(nullptr != source_table, "source table is null");
  CUDF_EXPECTS(nullptr != destination_table, "destination table is null");

  // If the destination is empty, return immediately as there is nothing to
  // gather
  if (0 == destination_table->num_rows()) {
    return;
  }

  CUDF_EXPECTS(nullptr != gather_map.data, "gather_map data is null");
  CUDF_EXPECTS(cudf::has_nulls(gather_map) == false, "gather_map contains nulls");
  CUDF_EXPECTS(source_table->num_columns() == destination_table->num_columns(),
               "Mismatched number of columns");

  cudf::type_dispatcher(gather_map.dtype, dispatch_map_type{},
			source_table, gather_map, destination_table, check_bounds,
			ignore_out_of_bounds,
			sync_nvstring_category, allow_negative_indices);
}

void gather(table const *source_table, gdf_index_type const gather_map[],
	    table *destination_table, bool check_bounds, bool ignore_out_of_bounds,
	    bool sync_nvstring_category, bool allow_negative_indices) {
  gdf_column gather_map_column{};
  gdf_column_view(&gather_map_column,
		  const_cast<gdf_index_type*>(gather_map),
		  nullptr,
		  destination_table->num_rows(),
		  gdf_dtype_of<gdf_index_type>());
  gather(source_table, gather_map_column, destination_table, check_bounds,
	 ignore_out_of_bounds, sync_nvstring_category,
	 allow_negative_indices);
}


} // namespace detail

table gather(table const *source_table, gdf_column const& gather_map, bool check_bounds) {
  table destination_table = cudf::allocate_like(*source_table,
						gather_map.size);
  detail::gather(source_table, gather_map, &destination_table,
		 check_bounds, false, false, true);
  nvcategory_gather_table(*source_table, destination_table);
  return destination_table;
}

void gather(table const *source_table, gdf_column const& gather_map,
	    table *destination_table, bool check_bounds) {
  detail::gather(source_table, gather_map, destination_table, check_bounds, false, false, true);
  nvcategory_gather_table(*source_table, *destination_table);
}

void gather(table const *source_table, gdf_index_type const gather_map[],
	    table *destination_table, bool check_bounds) {
  detail::gather(source_table, gather_map, destination_table, check_bounds, false, false, true);
  nvcategory_gather_table(*source_table, *destination_table);
}

} // namespace cudf
