#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <jit/type.h>

namespace cudf {
namespace experimental {
namespace detail {

packed_table::serialized_column serialize_column(column_view const& col,
                                                 rmm::device_buffer const& table_data)
{
  auto all_data_buffer_ptr = static_cast<uint8_t const*>(table_data.data());
  size_t data_offset = col.data<uint8_t>() - all_data_buffer_ptr;
  size_t null_mask_offset = reinterpret_cast<uint8_t const*>(col.null_mask()) - all_data_buffer_ptr;
  return packed_table::serialized_column{col.type(), col.size(), data_offset, null_mask_offset, col.num_children()};
}

void add_columns(std::vector<column_view> const& cols,
                 rmm::device_buffer const& table_data,
                 std::vector<packed_table::serialized_column> * table_metadata)
{
  for (auto &&col : cols) {
    table_metadata->emplace_back(serialize_column(col, table_data));
    std::vector<column_view> children;
    for (size_t i = 0; i < col.num_children(); i++) {
      children.push_back(col.child(i));
    }
    
    add_columns(children, table_data, table_metadata);
  }
}

packed_table pack(cudf::table_view const& input,
                  cudaStream_t stream,
                  rmm::mr::device_memory_resource* mr)
{
  contiguous_split_result contiguous_data = std::move(contiguous_split(input, {0})[0]);

  packed_table result{{}, std::move(contiguous_data.all_data)};
  
  std::vector<column_view> table_columns(input.begin(), input.end());

  add_columns(table_columns, *contiguous_data.all_data, &result.table_metadata);

  return result;
}

} // namespace detail

packed_table pack(cudf::table_view const& input,
                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE()
  return detail::pack(input, 0, mr);
}

} // namespace experimental  
} // namespace cudf
