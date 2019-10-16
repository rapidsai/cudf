#include "gather.cuh"
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/copying.hpp>
#include <utilities/error_utils.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/count.h>

#include <memory>


namespace cudf {
namespace experimental {
namespace detail {


struct dispatch_map_type {
  template <typename map_type, std::enable_if_t<std::is_integral<map_type>::value>* = nullptr>
  void operator()(table_view source_table, column_view gather_map,
		  mutable_table_view destination_table, bool check_bounds,
		  bool ignore_out_of_bounds, bool allow_negative_indices = false)
  {

    map_type const * typed_gather_map = gather_map.data<map_type>();

    if (check_bounds) {
      gdf_index_type begin = (allow_negative_indices) ? -source_table.num_rows() : 0;
      CUDF_EXPECTS(
	  destination_table.num_rows() == thrust::count_if(
	      rmm::exec_policy()->on(0),
	      typed_gather_map,
	      typed_gather_map + destination_table.num_rows(),
	      bounds_checker<map_type>{begin, source_table.num_rows()}),
	  "Index out of bounds.");
    }

    if (allow_negative_indices) {
      gather(source_table,
	     thrust::make_transform_iterator(
		 typed_gather_map,
		 index_converter<map_type,index_conversion::NEGATIVE_TO_POSITIVE>{source_table.num_rows()}),
	     destination_table,
	     check_bounds,
	     ignore_out_of_bounds,
	     allow_negative_indices
	     );
    }
    else {
      gather(source_table,
	     thrust::make_transform_iterator(
		 typed_gather_map,
		 index_converter<map_type>{source_table.num_rows()}),
	     destination_table,
	     check_bounds,
	     ignore_out_of_bounds,
	     allow_negative_indices
	     );
    }
  }

  template <typename map_type, std::enable_if_t<not std::is_integral<map_type>::value>* = nullptr>
  void operator()(table_view source_table, column_view gather_map,
                  mutable_table_view destination_table, bool check_bounds,
		  bool ignore_out_of_bounds, bool allow_negative_indices = false) {
   CUDF_FAIL("Gather map must be an integral type.");
  }
};

void gather(table_view source_table, column_view gather_map,
	    mutable_table_view destination_table, bool check_bounds = false,
	    bool ignore_out_of_bounds = false, bool allow_negative_indices = false,
	    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
  // If the destination is empty, return immediately as there is nothing to
  // gather
  if (0 == destination_table.num_rows()) {
    return;
  }

  CUDF_EXPECTS(gather_map.has_nulls() == false, "gather_map contains nulls");
  CUDF_EXPECTS(source_table.num_columns() == destination_table.num_columns(),
               "Mismatched number of columns");

  cudf::experimental::type_dispatcher(gather_map.type(), dispatch_map_type{},
				      source_table, gather_map, destination_table,
				      check_bounds, ignore_out_of_bounds,
				      allow_negative_indices);
}


}  // namespace detail
}  // namespace exp
}  // namespace cudf
