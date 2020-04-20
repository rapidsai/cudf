#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <jit/type.h>

namespace cudf {
namespace experimental {
namespace detail {

namespace {

packed_table::serialized_column serialize_column(column_view const& col,
                                                 rmm::device_buffer const& table_data)
{
  auto all_data_buffer_ptr = static_cast<uint8_t const*>(table_data.data());
  size_t data_offset = col.data<uint8_t>() - all_data_buffer_ptr;
  // TODO: fix for non-nullable
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

} // namespace anonymous

packed_table pack(cudf::table_view const& input,
                  cudaStream_t stream,
                  rmm::mr::device_memory_resource* mr)
{
  contiguous_split_result contiguous_data = std::move(contiguous_split(input, {})[0]);

  packed_table::serialized_column table_element = {{}, 0, 0, 0, contiguous_data.table.num_columns()};

  packed_table result{{table_element}, std::move(contiguous_data.all_data)};
  
  std::vector<column_view> table_columns(contiguous_data.table.begin(), contiguous_data.table.end());

  add_columns(table_columns, *result.table_data, &result.table_metadata);

  return result;
}

namespace {

column_view deserialize_column(packed_table::serialized_column serial_column,
                               std::vector<column_view> const& children,
                               rmm::device_buffer const& table_data)
{
  auto all_data_buffer_ptr = static_cast<uint8_t const*>(table_data.data());

  auto data_ptr = all_data_buffer_ptr + serial_column._data_offset;

  auto null_mask_ptr = reinterpret_cast<bitmask_type const*>(
    all_data_buffer_ptr + serial_column._null_mask_offset);

  return column_view(
    serial_column._type,
    serial_column._size,
    data_ptr,
    null_mask_ptr,
    UNKNOWN_NULL_COUNT,
    0,
    children);
}

std::vector<column_view> get_columns(cudf::size_type num_columns,
                                     std::vector<packed_table::serialized_column> const& serialized_columns,
                                     rmm::device_buffer const& table_data,
                                     size_t * current_index)
{
  std::vector<column_view> cols;
  for (size_t i = 0; i < num_columns; i++)
  {
    auto serial_column = serialized_columns[*current_index];
    (*current_index)++;

    std::vector<column_view> children = get_columns(
      serial_column._num_children,
      serialized_columns,
      table_data,
      current_index);

    cols.emplace_back(deserialize_column(serial_column, children, table_data));
  }
  
  return cols;
}

} // namespace anonymous

contiguous_split_result unpack(packed_table & input,
                               cudaStream_t stream,
                               rmm::mr::device_memory_resource* mr)
{
  cudf::size_type num_columns = input.table_metadata[0]._num_children;
  size_t current_index = 1;

  std::vector<column_view> table_columns = get_columns(num_columns,
                                                       input.table_metadata,
                                                       *input.table_data,
                                                       &current_index);

  return contiguous_split_result{table_view(table_columns), std::move(input.table_data)};
}

} // namespace detail

packed_table pack(cudf::table_view const& input,
                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::pack(input, 0, mr);
}

contiguous_split_result unpack(packed_table & input,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::unpack(input, 0, mr);
}

} // namespace experimental  
} // namespace cudf
