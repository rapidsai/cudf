#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/copying.hpp>

#include <memory>


namespace cudf {
namespace experimental {
namespace detail {


struct dispatch_map_type {
  template <typename map_type, std::enable_if_t<std::is_integral<map_type>::value>* = nullptr>
  void operator()(table_view source_table, column_view gather_map,
      mutable_table_view destination_table, bool check_bounds,
      bool ignore_out_of_bounds, bool sync_nvstring_category = false,
      bool allow_negative_indices = false)
  {

    map_type const * typed_gather_map = static_cast<map_type const*>(gather_map.data());

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
	     sync_nvstring_category,
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
	     sync_nvstring_category,
	     allow_negative_indices
	     );
    }
  }

  template <typename map_type, std::enable_if_t<not std::is_integral<map_type>::value>* = nullptr>
  void operator()(table_view source_table, column_view gather_map,
                  mutable_table_view destination_table, bool check_bounds,
		  bool ignore_out_of_bounds, bool sync_nvstring_category = false,
		  bool allow_negative_indices = false) {
   CUDF_FAIL("Gather map must be an integral type.");
  }
};

void gather(table_view source_table, column_view gather_map,
	    mutable_table_view destination_table, bool check_bounds = false,
	    bool ignore_out_of_bounds = false, bool sync_nvstring_category = false,
	    bool allow_negative_indices = false,
	    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
  // If the destination is empty, return immediately as there is nothing to
  // gather
  if (0 == destination_table.num_rows()) {
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


}  // namespace detail
}  // namespace exp
}  // namespace cudf


      