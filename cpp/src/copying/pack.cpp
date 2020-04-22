#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <jit/type.h>

namespace cudf {
namespace experimental {
namespace detail {

namespace {

struct serialized_column {
  cudf::data_type _type;
  cudf::size_type _size;
  size_t _data_offset;
  size_t _null_mask_offset;
  cudf::size_type _num_children;
};

inline void add_column_to_vector(serialized_column const& column,
                                 std::vector<uint8_t> * table_metadata)
{
  auto bytes = reinterpret_cast<uint8_t const*>(&column);
  std::copy(bytes, bytes + sizeof(serialized_column), std::back_inserter(*table_metadata));
}

serialized_column serialize_column(column_view const& col,
                                                 uint8_t const* base_ptr)
{
  // There are columns types that don't have data in parent e.g. strings
  size_t data_offset = col.data<uint8_t>()
                        ? col.data<uint8_t>() - base_ptr
                        : -1;
  size_t null_mask_offset = col.nullable()
                             ? reinterpret_cast<uint8_t const*>(col.null_mask()) - base_ptr
                             : -1;

  return serialized_column{col.type(), col.size(), data_offset, null_mask_offset, col.num_children()};
}

void add_columns(std::vector<column_view> const& cols,
                 uint8_t const* base_ptr,
                 std::vector<uint8_t> * table_metadata)
{
  for (auto &&col : cols) {
    add_column_to_vector(serialize_column(col, base_ptr), table_metadata);
    std::vector<column_view> children;
    for (size_t i = 0; i < col.num_children(); i++) {
      children.push_back(col.child(i));
    }
    
    add_columns(children, base_ptr, table_metadata);
  }
}

} // namespace anonymous

packed_table pack(cudf::table_view const& input,
                  cudaStream_t stream,
                  rmm::mr::device_memory_resource* mr)
{
  contiguous_split_result contiguous_data = 
    std::move(detail::contiguous_split(input, {}, mr, stream).front());

  serialized_column table_element = {{}, 0, 0, 0, contiguous_data.table.num_columns()};

  auto result = packed_table({}, std::move(contiguous_data.all_data));
  add_column_to_vector(table_element, &result.table_metadata);
  
  std::vector<column_view> table_columns(contiguous_data.table.begin(), contiguous_data.table.end());

  add_columns(table_columns,
              static_cast<uint8_t const*>(result.table_data->data()),
              &result.table_metadata);

  return result;
}

namespace {

column_view deserialize_column(serialized_column serial_column,
                               std::vector<column_view> const& children,
                               uint8_t const* base_ptr)
{
  auto data_ptr = serial_column._data_offset != -1
                  ? base_ptr + serial_column._data_offset
                  : 0;

  // size_t is an unsigned int so -1 is the max value of size_t. If the offset
  // is UINT64_MAX then just assume there's no null mask instead of thinking
  // what if there IS a null mask but the buffer is just -1u sized. This translates
  // to 16 EB of memory. No GPU has that amount of memory and it'll be a while
  // before anyone does. By that time, we'll have bigger problems because all code
  // that exists will need to be re-written to consider memory > 16 EB. It'll be
  // bigger than Y2K; and I'll be prepared with a cottage in Waknaghat and a lifetime
  // supply of soylent and shotgun ammo.
  // TODO: Replace above with better reasoning
  auto null_mask_ptr = serial_column._null_mask_offset != -1
                        ? reinterpret_cast<bitmask_type const*>(
                            base_ptr + serial_column._null_mask_offset)
                        : 0;

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
                                     serialized_column const* serialized_columns,
                                     uint8_t const* base_ptr,
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
      base_ptr,
      current_index);

    cols.emplace_back(deserialize_column(serial_column, children, base_ptr));
  }
  
  return cols;
}

} // namespace anonymous

contiguous_split_result unpack(packed_table & input)
{
  auto serialized_columns = reinterpret_cast<serialized_column const*>(input.table_metadata.data());
  cudf::size_type num_columns = serialized_columns[0]._num_children;
  size_t current_index = 1;

  std::vector<column_view> table_columns = get_columns(num_columns,
    serialized_columns,
                                                       static_cast<uint8_t const*>(input.table_data->data()),
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

contiguous_split_result unpack(packed_table & input)
{
  CUDF_FUNC_RANGE();
  return detail::unpack(input);
}

} // namespace experimental  
} // namespace cudf
