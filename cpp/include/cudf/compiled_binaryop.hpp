// For experimentation only 
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>

#include <memory>

namespace cudf {
namespace experimental {

std::unique_ptr<column> experimental_binary_operation1(
      column_view const& lhs,
      column_view const& rhs,
      data_type output_type,
      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
std::unique_ptr<column> experimental_binary_operation2(
      column_view const& lhs,
      column_view const& rhs,
      data_type output_type,
      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
std::unique_ptr<column> experimental_binary_operation3(
      column_view const& lhs,
      column_view const& rhs,
      data_type output_type,
      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
}  // namespace experimental
}  // namespace cudf

