#include "gather.cuh"
#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/copying.hpp>
#include <utilities/legacy/error_utils.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/count.h>

#include <memory>


namespace cudf {
namespace experimental {
namespace detail {


struct dispatch_map_type {
  template <typename map_type, std::enable_if_t<std::is_integral<map_type>::value>* = nullptr>
  std::unique_ptr<table> operator()(table_view const& source_table,
				    column_view const& gather_map,
				    size_type num_destination_rows, bool check_bounds,
				    bool ignore_out_of_bounds,
				    bool allow_negative_indices = false)
  {
    std::unique_ptr<table> destination_table;

    if (check_bounds) {
      cudf::size_type begin = (allow_negative_indices) ? -source_table.num_rows() : 0;
      CUDF_EXPECTS(
	  num_destination_rows == thrust::count_if(
	      rmm::exec_policy()->on(0),
	      gather_map.begin<map_type>(),
	      gather_map.end<map_type>(),
	      bounds_checker<map_type>{begin, source_table.num_rows()}),
	  "Index out of bounds.");
    }

    if (allow_negative_indices) {
      destination_table =
	gather(source_table,
	       thrust::make_transform_iterator(
					       gather_map.begin<map_type>(),
					       index_converter<map_type>{source_table.num_rows()}),
	       thrust::make_transform_iterator(
					       gather_map.end<map_type>(),
					       index_converter<map_type>{source_table.num_rows()}),
	       check_bounds,
	       ignore_out_of_bounds,
	       allow_negative_indices
	     );
    }
    else {
      destination_table =
	gather(source_table,
	       gather_map.begin<map_type>(),
	       gather_map.end<map_type>(),
	       check_bounds,
	       ignore_out_of_bounds,
	       allow_negative_indices
	       );
    }

    return destination_table;
  }

  template <typename map_type, std::enable_if_t<not std::is_integral<map_type>::value>* = nullptr>
  std::unique_ptr<table> operator()(table_view const& source_table, column_view const& gather_map,
				    size_type num_destination_rows, bool check_bounds,
				    bool ignore_out_of_bounds, bool allow_negative_indices = false) {
    CUDF_FAIL("Gather map must be an integral type.");
  }
};

std::unique_ptr<table> gather(table_view const& source_table, column_view const& gather_map,
			      bool check_bounds = false, bool ignore_out_of_bounds = false,
			      bool allow_negative_indices = false,
			      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {


  CUDF_EXPECTS(gather_map.has_nulls() == false, "gather_map contains nulls");

  std::unique_ptr<table> destination_table =
    cudf::experimental::type_dispatcher(gather_map.type(), dispatch_map_type{},
					source_table, gather_map,
					gather_map.size(),
					check_bounds, ignore_out_of_bounds,
					allow_negative_indices);

  return destination_table;
}


}  // namespace detail

std::unique_ptr<table> gather(table_view const& source_table, column_view const& gather_map,
			      bool check_bounds, rmm::mr::device_memory_resource* mr) {
  return detail::gather(source_table, gather_map, check_bounds, false, true, mr);
}

}  // namespace exp
}  // namespace cudf
