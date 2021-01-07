#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>

#include <jit/type.h>

namespace cudf {
namespace detail {

namespace {

column_view deserialize_column(serialized_column serial_column,
                               std::vector<column_view> const& children,
                               uint8_t const* base_ptr)
{
  auto data_ptr = serial_column.data_offset != -1 ? base_ptr + serial_column.data_offset : 0;

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
    serial_column.null_mask_offset != -1
      ? reinterpret_cast<bitmask_type const*>(base_ptr + serial_column.null_mask_offset)
      : 0;

  return column_view(serial_column.type,
                     serial_column.size,
                     data_ptr,
                     null_mask_ptr,
                     UNKNOWN_NULL_COUNT,
                     0,
                     children);
}

}  // anonymous namespace

packed_columns pack(cudf::table_view const& input,
                    cudaStream_t stream,
                    rmm::mr::device_memory_resource* mr)
{
  // do a contiguous_split with no splits to get the memory for the table
  // arranged as we want it
  auto contig_split_result = cudf::detail::contiguous_split(input, {}, stream, mr);
  return std::move(contig_split_result[0].data);
}

table_view unpack(packed_columns const& input)
{
  CUDF_EXPECTS(input.metadata != nullptr && input.gpu_data != nullptr,
               "Encountered invalid packed column input");
  auto serialized_columns = reinterpret_cast<serialized_column const*>(input.metadata->data());
  uint8_t const* base_ptr = static_cast<uint8_t const*>(input.gpu_data->data());
  // first entry is a stub where size == the total # of top level columns (see contiguous_split.cu)
  auto const num_columns = serialized_columns[0].size;
  size_t current_index   = 1;

  std::function<std::vector<column_view>(size_type)> get_columns;
  get_columns = [&serialized_columns, &current_index, base_ptr, &get_columns](size_t num_columns) {
    std::vector<column_view> cols;
    for (size_t i = 0; i < num_columns; i++) {
      auto serial_column = serialized_columns[current_index];
      current_index++;

      std::vector<column_view> children = get_columns(serial_column.num_children);

      cols.emplace_back(deserialize_column(serial_column, children, base_ptr));
    }

    return cols;
  };

  return table_view{get_columns(num_columns)};
}

}  // namespace detail

packed_columns pack(cudf::table_view const& input, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::pack(input, 0, mr);
}

table_view unpack(packed_columns const& input)
{
  CUDF_FUNC_RANGE();
  return detail::unpack(input);
}

}  // namespace cudf