#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf/table/table.hpp>

#include <memory>


namespace cudf {
namespace experimental {
namespace detail {

std::unique_ptr<table> gather(table_view const& source_table, column_view const& gather_map,
	     bool check_bounds = false, bool ignore_out_of_bounds = false,
	     bool allow_negative_indices = false,
	     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
}  // namespace detail
}  // namespace exp
}  // namespace cudf


#endif  // GATHER_HPP
