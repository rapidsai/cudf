#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>

#include <memory>


namespace cudf {
namespace exp {
namespace detail {

void scatter(table_view source_table, column_view gather_map,
	     mutable_table_view destination_table, bool check_bounds = false,
	     bool allow_negative_indices = false,
	     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace detail
}  // namespace exp
}  // namespace cudf


#endif  // GATHER_HPP
