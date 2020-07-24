#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> encode(
  column_view const& input_column, rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  // side effects of this function we are now dependent on:
  // - resulting column elements are sorted ascending
  // - nulls are sorted to the beginning
  auto table_keys = cudf::detail::drop_duplicates(table_view{{input_column}},
                                                  std::vector<size_type>{0},
                                                  duplicate_keep_option::KEEP_FIRST,
                                                  null_equality::EQUAL,
                                                  mr,
                                                  stream)
                      ->release();
  std::unique_ptr<column> keys_column(std::move(table_keys.front()));

  if (input_column.has_nulls()) {
    // the single null entry should be at the beginning -- side effect from drop_duplicates
    // copy the column without the null entry
    keys_column = std::make_unique<column>(
      slice(keys_column->view(), std::vector<size_type>{1, keys_column->size()}).front(),
      stream,
      mr);
    keys_column->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);  // remove the null-mask
  }

  // this returns a column with no null entries
  // - it appears to ignore the null entries in the input and tries to place the value regardless
  auto indices_column = cudf::detail::lower_bound(table_view{{keys_column->view()}},
                                                  table_view{{input_column}},
                                                  std::vector<order>{order::ASCENDING},
                                                  std::vector<null_order>{null_order::AFTER},
                                                  mr,
                                                  stream);

  return std::make_pair(std::move(keys_column), std::move(indices_column));
}
}  // namespace detail

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> encode(
  cudf::column_view const& input, rmm::mr::device_memory_resource* mr)
{
  return detail::encode(input, mr, 0);
}

}  // namespace cudf
