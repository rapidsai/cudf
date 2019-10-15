
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>

#include <memory>


namespace cudf {
namespace experimental {

std::unique_ptr<table> gather(table_view source_table, column_view gather_map,
			      bool check_bounds = false,
			      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

std::unique_ptr<table> scatter(table_view source, column_view scatter_map,
			       table_view target, bool check_bounds = false,
			       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace exp
}  // namespace cudf

			       
