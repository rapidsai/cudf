#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>

#include <jit/type.h>

namespace cudf {
namespace detail {

namespace {

struct serialized_column {
  data_type _type;
  size_type _size;
  size_t _data_offset;
  size_t _null_mask_offset;
  size_type _num_children;
};

inline void add_column_to_vector(serialized_column const& column, std::vector<uint8_t>* metadata)
{
  auto bytes = reinterpret_cast<uint8_t const*>(&column);
  std::copy(bytes, bytes + sizeof(serialized_column), std::back_inserter(*metadata));
}

serialized_column serialize(column_view const& col, uint8_t const* base_ptr)
{
  // There are columns types that don't have data in parent e.g. strings
  size_t data_offset = col.data<uint8_t>() ? col.data<uint8_t>() - base_ptr : -1;
  size_t null_mask_offset =
    col.nullable() ? reinterpret_cast<uint8_t const*>(col.null_mask()) - base_ptr : -1;

  return serialized_column{
    col.type(), col.size(), data_offset, null_mask_offset, col.num_children()};
}

void serialize_columns(std::vector<column_view> const& cols,
                       uint8_t const* base_ptr,
                       std::vector<uint8_t>* metadata)
{
  for (auto&& col : cols) {
    add_column_to_vector(serialize(col, base_ptr), metadata);
    std::vector<column_view> children;
    for (size_t i = 0; i < col.num_children(); i++) { children.push_back(col.child(i)); }

    serialize_columns(children, base_ptr, metadata);
  }
}

}  // namespace

packed_columns pack(std::vector<column_view> const& input,
                    cudaStream_t stream,
                    rmm::mr::device_memory_resource* mr)
{
  unpack_result contiguous_data = std::move(detail::alloc_and_copy(input, mr, stream));

  serialized_column first_element = {
    {}, 0, 0, 0, static_cast<size_type>(contiguous_data.columns.size())};

  auto result =
    packed_columns(std::make_unique<std::vector<uint8_t>>(), std::move(contiguous_data.all_data));
  add_column_to_vector(first_element, result.metadata.get());

  std::vector<column_view> table_columns(contiguous_data.columns.begin(),
                                         contiguous_data.columns.end());

  serialize_columns(
    table_columns, static_cast<uint8_t const*>(result.data->data()), result.metadata.get());

  return result;
}

namespace {

column_view deserialize_column(serialized_column serial_column,
                               std::vector<column_view> const& children,
                               uint8_t const* base_ptr)
{
  auto data_ptr = serial_column._data_offset != -1 ? base_ptr + serial_column._data_offset : 0;

  // size_t is an unsigned int so -1 is the max value of size_t. If the offset
  // is UINT64_MAX then just assume there's no null mask instead of thinking
  // what if there IS a null mask but the buffer is just -1u sized. This translates
  // to 16 EB of memory. No GPU has that amount of memory and it'll be a while
  // before anyone does. By that time, we'll have bigger problems because all code
  // that exists will need to be re-written to consider memory > 16 EB. It'll be
  // bigger than Y2K; and I'll be prepared with a cottage in Waknaghat and a lifetime
  // supply of soylent and shotgun ammo.
  // TODO: Replace above with better reasoning
  auto null_mask_ptr =
    serial_column._null_mask_offset != -1
      ? reinterpret_cast<bitmask_type const*>(base_ptr + serial_column._null_mask_offset)
      : 0;

  return column_view(serial_column._type,
                     serial_column._size,
                     data_ptr,
                     null_mask_ptr,
                     UNKNOWN_NULL_COUNT,
                     0,
                     children);
}

}  // namespace

unpack_result unpack(std::unique_ptr<packed_columns> input)
{
  auto serialized_columns = reinterpret_cast<serialized_column const*>(input->metadata->data());
  uint8_t const* base_ptr = static_cast<uint8_t const*>(input->data->data());
  size_t current_index    = 1;

  std::function<std::vector<column_view>(size_type)> get_columns;
  get_columns = [&serialized_columns, &current_index, base_ptr, &get_columns](size_t num_columns) {
    std::vector<column_view> cols;
    for (size_t i = 0; i < num_columns; i++) {
      auto serial_column = serialized_columns[current_index];
      current_index++;

      std::vector<column_view> children = get_columns(serial_column._num_children);

      cols.emplace_back(deserialize_column(serial_column, children, base_ptr));
    }

    return cols;
  };

  std::vector<column_view> table_columns = get_columns(serialized_columns[0]._num_children);

  return unpack_result{table_columns, std::move(input->data)};
}

}  // namespace detail

packed_columns pack(std::vector<column_view> const& input, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::pack(input, 0, mr);
}

unpack_result unpack(std::unique_ptr<packed_columns> input)
{
  CUDF_FUNC_RANGE();
  return detail::unpack(std::move(input));
}

}  // namespace cudf
